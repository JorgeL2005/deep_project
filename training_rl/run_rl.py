"""PPO self-play fine-tuning from the BC policy (alternative agent, v-RL).

Method (grounded in the Overcooked literature):
  - BC-initialized PPO: start from the distilled CNN (v3) so RL refines
    instead of learning from scratch (BC init stabilizes PPO post-training).
  - Population-style partners (FCP-lite): each parallel env samples a partner
    per episode: shared-policy self-play (60%), the frozen BC v3 net (20%),
    greedy scripted (10%), random_motion (10%).
  - Reward: team sparse reward + 0.5 x per-agent shaped reward (pot/dish/soup
    pickups), scaled by 1/20.
  - KL anchor to the frozen BC policy prevents catastrophic forgetting.
  - Gameplay probe every PROBE_EVERY iterations; best checkpoint wins and the
    BC v3 baseline competes, so the exported agent is never worse.

Nothing existing is modified: output goes to policies/rl_weights.npz and
training_rl/logs/. The exported npz is directly loadable by
policies/student_agent_bc.py via config weights: rl_weights.npz.

Usage:  vt\\Scripts\\python.exe -m training_rl.run_rl [iterations]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from training.common import FrameStacker, PROJECT_ROOT, build_layout_env, pad_obs
from training import rollouts as R
from training.run_pipeline import MODEL_DIR, usable_layouts, short
from training.run_pipeline_v3 import probe_v3
from training_rl.ppo import ActorCritic, ActorOnly, PPOUpdater, export_actor_npz

from src.constants import action_index_to_overcooked_action

LOG_DIR = PROJECT_ROOT / "training_rl" / "logs"
RL_MODEL_DIR = PROJECT_ROOT / "training_rl" / "models"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_ENVS = 16
SEG_T = 160          # steps per env per iteration
HORIZON = 400
GAMMA = 0.99
LAM = 0.95
PROBE_EVERY = 30
PARTNER_MODES = ["selfplay", "selfplay", "selfplay", "frozen_bc", "greedy", "random"]

_log_file = open(LOG_DIR / "rl.log", "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")
    _log_file.flush()


class EnvSlot:
    """One environment with per-episode layout/partner sampling."""

    def __init__(self, layouts, rng, frozen_policy_fn):
        self.layouts = layouts
        self.rng = rng
        self.frozen_policy_fn = frozen_policy_fn
        self.episode_returns = []
        self.reset_episode()

    def reset_episode(self):
        self.layout = self.layouts[int(self.rng.integers(0, len(self.layouts)))]
        self.env = build_layout_env(self.layout, horizon=HORIZON)
        self.env.reset(regen_mdp=False)
        self.mode = PARTNER_MODES[int(self.rng.integers(0, len(PARTNER_MODES)))]
        self.stackers = [FrameStacker(), FrameStacker()]
        self.ep_return = 0.0
        if self.mode == "selfplay":
            self.learners = [0, 1]
            self.partner = None
        else:
            li = int(self.rng.integers(0, 2))
            self.learners = [li]
            pi = 1 - li
            if self.mode == "frozen_bc":
                self.partner = ("frozen", pi)
            else:
                name = "greedy_full_task" if self.mode == "greedy" else "random_motion"
                pol = R.BuiltinPolicy(name, seed=int(self.rng.integers(0, 10_000)))
                pol.start(self.env)
                pol.set_index(pi, self.env)
                self.partner = ("scripted", pi, pol)

    def obs_for(self, idx):
        raw = np.asarray(self.env.lossless_state_encoding_mdp(self.env.state)[idx])
        stacked = self.stackers[idx].push(np.clip(raw, 0, 255).astype(np.uint8))
        return pad_obs(stacked.astype(np.float32)).astype(np.float32)

    def obs_peek(self, idx):
        """Stacked observation WITHOUT mutating the frame history (bootstrap)."""
        raw = np.clip(
            np.asarray(self.env.lossless_state_encoding_mdp(self.env.state)[idx]), 0, 255
        ).astype(np.uint8)
        fr = self.stackers[idx].frames
        frames = ([raw] * 3) if not fr else fr[1:] + [raw]
        return pad_obs(np.concatenate(frames, axis=2).astype(np.float32)).astype(np.float32)


def collect_segment(slots, ac, frozen_actor, device, rng):
    """Run SEG_T lockstep steps over all slots; return flat PPO batch."""
    streams = {}  # (slot_id, agent_idx) -> dict of lists

    def stream(sid, ai):
        key = (sid, ai)
        if key not in streams:
            streams[key] = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
        return streams[key]

    for _ in range(SEG_T):
        # Batch observations: learners (current policy) and frozen partners.
        learner_keys, learner_obs = [], []
        frozen_keys, frozen_obs = [], []
        for sid, slot in enumerate(slots):
            for ai in slot.learners:
                learner_keys.append((sid, ai))
                learner_obs.append(slot.obs_for(ai))
            if slot.partner and slot.partner[0] == "frozen":
                ai = slot.partner[1]
                frozen_keys.append((sid, ai))
                frozen_obs.append(slot.obs_for(ai))

        x = torch.from_numpy(np.stack(learner_obs)).to(device)
        with torch.no_grad():
            logits, values = ac(x)
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)
        acts_np = acts.cpu().numpy()
        logps_np = logps.cpu().numpy()
        vals_np = values.cpu().numpy()

        frozen_acts = {}
        if frozen_keys:
            xf = torch.from_numpy(np.stack(frozen_obs)).to(device)
            with torch.no_grad():
                flogits = frozen_actor(xf)
            fa = flogits.argmax(1).cpu().numpy()
            frozen_acts = {k: int(a) for k, a in zip(frozen_keys, fa)}

        # Step every env.
        for sid, slot in enumerate(slots):
            joint = [4, 4]
            for k, (lsid, ai) in enumerate(learner_keys):
                if lsid == sid:
                    joint[ai] = int(acts_np[k])
            if slot.partner:
                if slot.partner[0] == "frozen":
                    joint[slot.partner[1]] = frozen_acts[(sid, slot.partner[1])]
                else:
                    _, pi, pol = slot.partner
                    joint[pi] = pol.act(slot.env, slot.env.state, pi)
            oc_joint = tuple(action_index_to_overcooked_action(a) for a in joint)
            _, sparse, done, info = slot.env.step(oc_joint)
            shaped = info.get("shaped_r_by_agent", [0, 0])
            slot.ep_return += float(sparse)

            for k, (lsid, ai) in enumerate(learner_keys):
                if lsid != sid:
                    continue
                st = stream(sid, ai)
                st["obs"].append(learner_obs[k])
                st["act"].append(int(acts_np[k]))
                st["logp"].append(float(logps_np[k]))
                st["val"].append(float(vals_np[k]))
                st["rew"].append(float(sparse) / 20.0 + 0.5 * float(shaped[ai]) / 20.0)
                st["done"].append(bool(done))

            if done:
                slot.episode_returns.append(slot.ep_return)
                slot.reset_episode()

    # Bootstrap values for unfinished streams and compute GAE.
    boot_keys, boot_obs = [], []
    for (sid, ai), st in streams.items():
        if not st["done"][-1]:
            slot = slots[sid]
            if ai in slot.learners:  # same episode continuing
                boot_keys.append((sid, ai))
                boot_obs.append(slot.obs_peek(ai))
    boot_vals = {}
    if boot_keys:
        xb = torch.from_numpy(np.stack(boot_obs)).to(device)
        with torch.no_grad():
            _, vb = ac(xb)
        boot_vals = {k: float(v) for k, v in zip(boot_keys, vb.cpu().numpy())}
    # NOTE: after a mid-segment reset the (sid, ai) stream mixes episodes, but
    # done=True at the boundary stops advantage flow across them.

    all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []
    for key, st in streams.items():
        rew = np.asarray(st["rew"], dtype=np.float64)
        val = np.asarray(st["val"], dtype=np.float64)
        done = np.asarray(st["done"], dtype=np.float64)
        n = len(rew)
        adv = np.zeros(n)
        next_val = boot_vals.get(key, 0.0)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nonterm = 1.0 - done[t]
            delta = rew[t] + GAMMA * next_val * nonterm - val[t]
            gae = delta + GAMMA * LAM * nonterm * gae
            adv[t] = gae
            next_val = val[t]
        ret = adv + val
        all_obs.extend(st["obs"])
        all_act.extend(st["act"])
        all_logp.extend(st["logp"])
        all_adv.extend(adv.tolist())
        all_ret.extend(ret.tolist())

    batch = (
        torch.from_numpy(np.stack(all_obs).astype(np.uint8)),
        torch.tensor(all_act, dtype=torch.long),
        torch.tensor(all_logp, dtype=torch.float32),
        torch.tensor(all_adv, dtype=torch.float32),
        torch.tensor(all_ret, dtype=torch.float32),
    )
    return batch


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== PPO fine-tuning start (device={device}, iters={iterations}) ===")
    rng = np.random.default_rng(123)

    layouts = usable_layouts()
    log(f"usable layouts: {len(layouts)}")

    bc_state = torch.load(MODEL_DIR / "bc_f3_iter6.pt", map_location="cpu")  # v3 winner
    ac = ActorCritic()
    ac.load_bc_state(bc_state)
    ac = ac.to(device)

    frozen = ActorOnly(ActorCritic())
    frozen.ac.load_bc_state(bc_state)
    frozen = frozen.to(device).eval()

    updater = PPOUpdater(ac, frozen, device)
    slots = [EnvSlot(layouts, rng, None) for _ in range(N_ENVS)]

    best = {"score": -1e9, "state": None, "tag": None}

    def consider(tag):
        actor = ActorOnly(ac).eval()
        ret, idle = probe_v3(actor, device, rng)
        score = ret - 2.0 * idle
        log(f"  probe [{tag}] mean_return={ret:.1f} mean_max_idle={idle:.1f} score={score:.1f}")
        if score > best["score"]:
            best.update(score=score, tag=tag,
                        state={k: v.cpu().clone() for k, v in ac.state_dict().items()})
        ac.train()

    consider("bc-v3-init")

    WARMUP = 25  # critic-head-only iterations before any policy update
    t0 = time.time()
    for it in range(1, iterations + 1):
        batch = collect_segment(slots, ac, frozen, device, rng)
        stats = updater.update(*batch, critic_only=(it <= WARMUP))
        if it % 5 == 0:
            recent = [r for s in slots for r in s.episode_returns[-8:]]
            mean_ep = float(np.mean(recent)) if recent else 0.0
            log(
                f"iter {it}/{iterations} ep_return~{mean_ep:.1f} "
                f"pi={stats['pi_loss']:.3f} v={stats['v_loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl_bc={stats['kl_bc']:.4f} "
                f"({(time.time() - t0) / it:.1f}s/it)"
            )
        if it % PROBE_EVERY == 0:
            consider(f"iter{it}")
            torch.save(ac.state_dict(), RL_MODEL_DIR / f"ppo_iter{it}.pt")

    consider("final")
    log(f"best checkpoint: {best['tag']} (score {best['score']:.1f})")

    ac_best = ActorCritic()
    ac_best.load_state_dict(best["state"])
    weights_path = PROJECT_ROOT / "policies" / "rl_weights.npz"
    export_actor_npz(ac_best, weights_path)
    log(f"exported actor -> {weights_path}")

    # Final official + custom comparison using the numpy deliverable path.
    from training.run_pipeline_v4 import eval_matrix
    from training.common import OFFICIAL_LAYOUTS
    import json

    log("final evaluation of RL agent (official layouts)...")
    off = eval_matrix(OFFICIAL_LAYOUTS, weights_path, "rl-official",
                      partners=("self", "greedy_full_task", "random_motion"))
    (LOG_DIR / "final_eval_rl.json").write_text(json.dumps(off, indent=1))
    means = [v["mean"] for v in off.values()]
    log(f"official matrix mean: {float(np.mean(means)):.1f}")
    log("=== RL pipeline done ===")


if __name__ == "__main__":
    main()
