"""Rollout and data-collection machinery for BC/DAgger distillation.

Teacher = planning agent (policies/student_agent.py). Student = CNN policy.
Each recorded sample is (unpadded lossless grid encoding uint8, teacher action).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from training.common import (  # noqa: F401  (sets sys.path)
    N_ACTIONS,
    PAD_H,
    PAD_W,
    PROJECT_ROOT,
    FrameStacker,
    build_layout_env,
    pad_obs,
)

from src.constants import action_index_to_overcooked_action, overcooked_action_to_index
from src.policy_loader import build_builtin_agent


def _load_teacher_class():
    path = PROJECT_ROOT / "policies" / "student_agent.py"
    spec = importlib.util.spec_from_file_location("teacher_agent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StudentAgent


TeacherClass = _load_teacher_class()


class TeacherPolicy:
    """The planning agent, used as expert/label source."""

    def __init__(self, seed=0):
        self.agent = TeacherClass({"seed": int(seed)})

    def start(self, env):
        self.agent.reset()

    def act(self, env, state, idx) -> int:
        return int(self.agent.act({"state": state, "mdp": env.mdp, "agent_index": idx}))


class BuiltinPolicy:
    """Wraps the runner's builtin baseline agents (greedy, random_motion, stay)."""

    def __init__(self, name, seed=0):
        self.name = name
        self.seed = seed
        self.agent = None

    def start(self, env):
        self.agent = build_builtin_agent(self.name, env, {"seed": self.seed})
        self.agent.reset()

    def set_index(self, idx, env):
        self.agent.set_agent_index(idx)
        self.agent.set_mdp(env.mdp)

    def act(self, env, state, idx) -> int:
        action, _ = self.agent.action(state)
        return overcooked_action_to_index(action)


class NetPolicy:
    """Torch CNN student policy (used during DAgger rollouts)."""

    def __init__(self, model, device):
        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.stacker = FrameStacker()

    def start(self, env):
        self.stacker = FrameStacker()

    def act(self, env, state, idx) -> int:
        obs = np.asarray(env.lossless_state_encoding_mdp(state)[idx])
        x = pad_obs(self.stacker.push(obs))
        if x is None:
            return 4  # stay on oversized layouts (should not happen)
        with self.torch.no_grad():
            t = self.torch.from_numpy(x[None]).float().to(self.device)
            logits = self.model(t)[0]
        return int(logits.argmax().item())


def rollout_collect(
    layout: str,
    policies,
    record_idx,
    rng: np.random.Generator,
    horizon: int = 400,
    exec_eps: float = 0.0,
    shadow_teachers=None,
    stuck_boost: int = 0,
):
    """Run one episode, return (samples, sparse_return).

    policies: [policy_for_agent0, policy_for_agent1]
    record_idx: agent indices to record samples for.
    shadow_teachers: dict idx -> TeacherPolicy giving DAgger labels when the
        acting policy for that index is not the teacher itself.
    exec_eps: prob of replacing the executed action with a random one
        (state-coverage noise; labels stay clean).
    """
    env = build_layout_env(layout, horizon=horizon)
    env.reset(regen_mdp=False)

    for i, pol in enumerate(policies):
        pol.start(env)
        if isinstance(pol, BuiltinPolicy):
            pol.set_index(i, env)
    shadow_teachers = shadow_teachers or {}
    for st in shadow_teachers.values():
        st.start(env)
    rec_stackers = {i: FrameStacker() for i in record_idx}
    pos_history = {i: [] for i in record_idx}

    samples = []
    total = 0.0
    done = False
    while not done:
        state = env.state
        actions = []
        for i, pol in enumerate(policies):
            a = pol.act(env, state, i)
            label = a
            if i in record_idx:
                if i in shadow_teachers:
                    label = shadow_teachers[i].act(env, state, i)
                obs = np.asarray(env.lossless_state_encoding_mdp(state)[i])
                if obs.shape[0] <= PAD_W and obs.shape[1] <= PAD_H:
                    stacked = rec_stackers[i].push(np.clip(obs, 0, 255).astype(np.uint8))
                    samples.append((stacked, int(label)))
                    # Oversample "frozen" states: the rollout policy has not
                    # moved for a while but the teacher says act. These are
                    # exactly the states where the student idles in play.
                    if stuck_boost > 0:
                        hist = pos_history[i]
                        hist.append(state.players[i].position)
                        if len(hist) > 4:
                            hist.pop(0)
                        if len(hist) >= 4 and len(set(hist)) == 1 and label != 4:
                            samples.extend([(stacked, int(label))] * stuck_boost)
            if exec_eps > 0 and rng.random() < exec_eps:
                a = int(rng.integers(0, N_ACTIONS))
            actions.append(a)

        joint = tuple(action_index_to_overcooked_action(a) for a in actions)
        _, reward, done, _ = env.step(joint)
        total += float(reward)
    return samples, total


def save_samples(samples, out_path: Path):
    """Group variable-shape observations by shape and save compressed."""
    by_shape = {}
    for obs, label in samples:
        by_shape.setdefault(obs.shape, [[], []])
        by_shape[obs.shape][0].append(obs)
        by_shape[obs.shape][1].append(label)
    arrays = {}
    for k, (obs_list, labels) in enumerate(by_shape.values()):
        arrays[f"obs_{k}"] = np.stack(obs_list)
        arrays[f"lab_{k}"] = np.asarray(labels, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def load_sample_files(paths):
    """Return list of (obs_batch uint8 (N,W,H,26), labels (N,))."""
    groups = []
    for p in paths:
        with np.load(p) as data:
            keys = sorted(k for k in data.files if k.startswith("obs_"))
            for k in keys:
                idx = k.split("_")[1]
                groups.append((data[k], data[f"lab_{idx}"]))
    return groups
