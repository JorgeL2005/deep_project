"""Minimal student policy template.

Students may replace this file with their own policy. The runner expects a class
called StudentAgent with:

    __init__(self, config: dict)
    reset(self)
    act(self, obs) -> int

Action convention:
    0 = north/up
    1 = south/down
    2 = east/right
    3 = west/left
    4 = stay
    5 = interact
"""

from __future__ import annotations

import numpy as np


class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.fixed_action = str(self.config.get("action", "stay")).lower()
        self.action_map = {
            "north": 0,
            "up": 0,
            "south": 1,
            "down": 1,
            "east": 2,
            "right": 2,
            "west": 3,
            "left": 3,
            "stay": 4,
            "interact": 5,
            "random": -1,
        }
        if self.fixed_action not in self.action_map:
            raise ValueError(f"Unknown fixed action: {self.fixed_action}")
        self.rng = np.random.default_rng(self.config.get("seed", None))

    def reset(self):
        pass

    def act(self, obs):
        """Return an action index in {0, 1, 2, 3, 4, 5}."""
        action = self.action_map[self.fixed_action]
        if action == -1:
            return int(self.rng.integers(0, 6))
        return action
