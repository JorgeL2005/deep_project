"""Shared utilities for the BC/DAgger training pipeline.

The teacher is the planning agent in policies/student_agent.py. The student
is a CNN over the lossless grid encoding (26 channels, padded to a fixed
size), trained with behavioral cloning + DAgger and exported to .npz for
dependency-free numpy inference at evaluation time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERCOOKED_REPO = PROJECT_ROOT.parent / "Overcooked-AI"
for p in (str(PROJECT_ROOT), str(OVERCOOKED_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Fixed input geometry for the CNN (grids are padded bottom/right with zeros).
PAD_W = 24
PAD_H = 16
N_CHANNELS = 26
N_FRAMES = 3  # frame stacking: captures partner motion / own recent history
IN_CHANNELS = N_CHANNELS * N_FRAMES
N_ACTIONS = 6

OFFICIAL_LAYOUTS = [
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "forced_coordination",
    "asymmetric_advantages",
    "large_room",
    "simple_o",
    "simple_tomato",
    "small_corridor",
    "soup_coordination",
    "tutorial_0",
    "tutorial_1",
    "tutorial_2",
    "tutorial_3",
]

CUSTOM_LAYOUT_DIR = PROJECT_ROOT / "overcooked_dataset" / "layouts_custom"


def pad_obs(obs: np.ndarray) -> np.ndarray | None:
    """(W, H, C) lossless encoding (possibly frame-stacked) -> (C, PAD_H, PAD_W)."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 3 or obs.shape[2] % N_CHANNELS != 0:
        return None
    w, h, c = obs.shape
    if w > PAD_W or h > PAD_H:
        return None
    out = np.zeros((c, PAD_H, PAD_W), dtype=np.float32)
    out[:, :h, :w] = obs.transpose(2, 1, 0)
    return out


class FrameStacker:
    """Keep the last N_FRAMES single-frame encodings; oldest first channel block."""

    def __init__(self):
        self.frames = []

    def push(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if not self.frames:
            self.frames = [obs] * N_FRAMES
        else:
            self.frames = self.frames[1:] + [obs]
        return np.concatenate(self.frames, axis=2)


def list_custom_layouts() -> list[str]:
    return sorted(str(p) for p in CUSTOM_LAYOUT_DIR.glob("*.layout"))


def build_layout_env(layout: str, horizon: int = 400):
    """layout: builtin name or path to a .layout file."""
    from src.environment import build_env

    cfg = {"horizon": horizon}
    if layout.endswith(".layout"):
        cfg["layout_file"] = layout
        cfg["layout_name"] = None
    else:
        cfg["layout_name"] = layout
    return build_env(cfg)
