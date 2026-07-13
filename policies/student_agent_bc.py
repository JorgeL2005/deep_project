"""Trained agent for the Overcooked-AI competition (deliverable).

This is a convolutional neural network policy trained by behavioral cloning +
DAgger: an expert planner generated millions of (observation, action) pairs by
playing on all official and custom layouts with diverse partners, and the CNN
was trained to imitate it (policy distillation). At evaluation time only this
network runs - no planning, no heuristics, no external dependencies beyond
numpy. Weights live in policies/bc_weights.npz.

Input: the standard Overcooked "lossless" grid encoding (W, H, 26 channels),
zero-padded to 16x24. The network is layout-agnostic: it was trained across
~50 different kitchens, both player roles, and multiple partner types.

Architecture: Conv(26-64)-ReLU-Conv(64-64)-ReLU-Conv(64-32)-ReLU-FC(256)-FC(6).

Runner interface:
    StudentAgent(config) ; reset() ; act(obs) -> int in {0..5}

Preferred YAML observation config:

    observation:
      type: lossless_grid
      include_agent_index: true

(observation.type: state also works - the encoding is computed internally.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N_CHANNELS = 26
N_FRAMES = 3
PAD_H = 16
PAD_W = 24
STAY = 4

WEIGHTS_PATH = Path(__file__).resolve().parent / "bc_weights.npz"


class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self._warned = False
        weights_file = Path(self.config.get("weights", WEIGHTS_PATH))
        if not weights_file.is_absolute():
            candidate = Path(__file__).resolve().parent / weights_file.name
            if candidate.exists():
                weights_file = candidate
        data = np.load(weights_file)
        self.w = {k: data[k].astype(np.float32) for k in data.files}
        # Small tie-breaking noise: prevents two deterministic copies of the
        # policy from mirror-locking each other in narrow corridors.
        self.epsilon = float(self.config.get("epsilon", 0.02))
        self.rng = np.random.default_rng(int(self.config.get("seed", 0) or 0))
        self._frames = []

    def reset(self):
        self._frames = []

    # ------------------------------------------------------------------

    def act(self, obs) -> int:
        try:
            x = self._extract(obs)
            if x is None:
                return STAY
            logits = self._forward(x)
            if self.epsilon > 0 and self.rng.random() < self.epsilon:
                return int(self.rng.integers(0, 6))
            return int(np.argmax(logits))
        except Exception as exc:
            if not self._warned:
                print(f"[student_agent_bc] error, defaulting to stay: {exc!r}", file=sys.stderr)
                self._warned = True
            return STAY

    # ------------------------------------------------------------------

    def _extract(self, obs):
        """Accept lossless array, {'obs': array}, or {'state','mdp'} dicts."""
        arr = None
        if isinstance(obs, np.ndarray):
            arr = obs
        elif isinstance(obs, dict):
            if "obs" in obs:
                arr = np.asarray(obs["obs"])
            elif "state" in obs and "mdp" in obs:
                idx = int(obs.get("agent_index", 0))
                arr = np.asarray(obs["mdp"].lossless_state_encoding(obs["state"], 400)[idx])
        if arr is None:
            if not self._warned:
                print(
                    "[student_agent_bc] need observation.type 'lossless_grid' or 'state'",
                    file=sys.stderr,
                )
                self._warned = True
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != N_CHANNELS:
            return None
        w, h, _ = arr.shape
        if w > PAD_W or h > PAD_H:
            return None
        # Frame stacking (last N_FRAMES observations, oldest first).
        if not self._frames:
            self._frames = [arr] * N_FRAMES
        else:
            self._frames = self._frames[1:] + [arr]
        stacked = np.concatenate(self._frames, axis=2)
        out = np.zeros((N_CHANNELS * N_FRAMES, PAD_H, PAD_W), dtype=np.float32)
        out[:, :h, :w] = stacked.transpose(2, 1, 0)
        return out

    def _forward(self, x):
        w = self.w
        h = self._conv(x, w["w0"], w["b0"])
        np.maximum(h, 0, out=h)
        h = self._conv(h, w["w1"], w["b1"])
        np.maximum(h, 0, out=h)
        h = self._conv(h, w["w2"], w["b2"])
        np.maximum(h, 0, out=h)
        v = h.reshape(-1)
        v = w["w3"] @ v + w["b3"]
        np.maximum(v, 0, out=v)
        return w["w4"] @ v + w["b4"]

    @staticmethod
    def _conv(x, w, b):
        c_out, c_in, kh, kw = w.shape
        _, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        cols = np.empty((c_in * kh * kw, h * wd), dtype=np.float32)
        idx = 0
        for ci in range(c_in):
            for i in range(kh):
                for j in range(kw):
                    cols[idx] = xp[ci, i : i + h, j : j + wd].ravel()
                    idx += 1
        out = w.reshape(c_out, -1) @ cols + b[:, None]
        return out.reshape(c_out, h, wd)
