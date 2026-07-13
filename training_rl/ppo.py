"""PPO actor-critic for Overcooked, initialized from the BC network.

The actor trunk mirrors training/train_bc.build_model() exactly, so BC
weights load directly and the trained actor exports to the same .npz layout
consumed by policies/student_agent_bc.py (w0..w4).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.common import IN_CHANNELS, N_ACTIONS, PAD_H, PAD_W


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(IN_CHANNELS, 96, 3, padding=1)
        self.c2 = nn.Conv2d(96, 96, 3, padding=1)
        self.c3 = nn.Conv2d(96, 64, 3, padding=1)
        self.fc = nn.Linear(64 * PAD_H * PAD_W, 256)
        self.actor = nn.Linear(256, N_ACTIONS)
        self.critic = nn.Linear(256, 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def trunk(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = h.flatten(1)
        return F.relu(self.fc(h))

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def load_bc_state(self, bc_state_dict):
        """Map weights from the BC nn.Sequential (indices 0,2,4,7,9)."""
        mapping = {"0": self.c1, "2": self.c2, "4": self.c3, "7": self.fc, "9": self.actor}
        for idx, layer in mapping.items():
            layer.weight.data.copy_(bc_state_dict[f"{idx}.weight"])
            layer.bias.data.copy_(bc_state_dict[f"{idx}.bias"])


class ActorOnly(nn.Module):
    """Logits-only view: probe-compatible and exportable like the BC model."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    def forward(self, x):
        logits, _ = self.ac(x)
        return logits


def export_actor_npz(ac: ActorCritic, out_path):
    arrays = {}
    for i, layer in enumerate([ac.c1, ac.c2, ac.c3, ac.fc, ac.actor]):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(out_path, **arrays)
    return arrays


class PPOUpdater:
    def __init__(self, ac: ActorCritic, frozen_actor: ActorOnly, device,
                 actor_lr=2e-5, critic_lr=3e-4, clip=0.15, vcoef=0.5,
                 entcoef=0.003, klcoef=0.15):
        self.ac = ac
        self.frozen = frozen_actor.eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.device = device
        policy_params = (
            list(ac.c1.parameters()) + list(ac.c2.parameters()) + list(ac.c3.parameters())
            + list(ac.fc.parameters()) + list(ac.actor.parameters())
        )
        self.opt = torch.optim.Adam([
            {"params": policy_params, "lr": actor_lr},
            {"params": ac.critic.parameters(), "lr": critic_lr},
        ])
        self.clip = clip
        self.vcoef = vcoef
        self.entcoef = entcoef
        self.klcoef = klcoef

    def update(self, obs_u8, actions, old_logp, advantages, returns,
               epochs=2, minibatch=2048, critic_only=False):
        """All args are torch tensors on CPU; obs_u8 is uint8 (N,C,H,W)."""
        n = len(actions)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl_bc": 0.0, "count": 0}
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = perm[start : start + minibatch]
                x = obs_u8[idx].to(self.device).float()
                a = actions[idx].to(self.device)
                lp_old = old_logp[idx].to(self.device)
                ad = adv[idx].to(self.device)
                ret = returns[idx].to(self.device)

                if critic_only:
                    # Warmup: fit ONLY the value head (trunk frozen) before
                    # touching the policy — a fresh critic gives noisy
                    # advantages that wreck the BC initialization.
                    with torch.no_grad():
                        h = self.ac.trunk(x)
                    value = self.ac.critic(h).squeeze(-1)
                    v_loss = F.mse_loss(value, ret)
                    loss = self.vcoef * v_loss
                    pi_loss = torch.zeros(())
                    entropy = torch.zeros(())
                    kl_bc = torch.zeros(())
                else:
                    logits, value = self.ac(x)
                    v_loss = F.mse_loss(value, ret)
                    dist = torch.distributions.Categorical(logits=logits)
                    lp = dist.log_prob(a)
                    ratio = (lp - lp_old).exp()
                    pi_loss = -torch.min(
                        ratio * ad, ratio.clamp(1 - self.clip, 1 + self.clip) * ad
                    ).mean()
                    entropy = dist.entropy().mean()

                    with torch.no_grad():
                        frozen_logits = self.frozen(x)
                    kl_bc = F.kl_div(
                        F.log_softmax(frozen_logits, dim=-1),
                        F.log_softmax(logits, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )

                    loss = pi_loss + self.vcoef * v_loss - self.entcoef * entropy + self.klcoef * kl_bc
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()

                stats["pi_loss"] += float(pi_loss)
                stats["v_loss"] += float(v_loss)
                stats["entropy"] += float(entropy)
                stats["kl_bc"] += float(kl_bc)
                stats["count"] += 1
        c = max(1, stats.pop("count"))
        return {k: v / c for k, v in stats.items()}
