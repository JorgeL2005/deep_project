"""Observation builders exposed to student policies."""

from __future__ import annotations

from typing import Any

import numpy as np


class ObservationConfigError(ValueError):
    """Raised when the observation configuration is invalid."""


class ObservationBuilder:
    """Build per-agent observations from an Overcooked state.

    Supported types:
    - state: returns a small dictionary with the raw Overcooked state.
    - featurized: returns env.featurize_state_mdp(state)[agent_index].
    - lossless_grid: returns env.lossless_state_encoding_mdp(state)[agent_index].
    """

    def __init__(self, env, config: dict[str, Any] | None = None):
        self.env = env
        self.config = config or {}
        self.obs_type = str(self.config.get("type", "featurized"))
        self.include_agent_index = bool(self.config.get("include_agent_index", True))

        if self.obs_type not in {"state", "featurized", "lossless_grid"}:
            raise ObservationConfigError(
                f"Unknown observation.type='{self.obs_type}'. Use: state, featurized, lossless_grid"
            )

    def __call__(self, state, agent_index: int):
        if self.obs_type == "state":
            obs = {
                "state": state,
                "mdp": self.env.mdp,
                "agent_index": agent_index,
            }
            return obs if self.include_agent_index else {"state": state, "mdp": self.env.mdp}

        if self.obs_type == "featurized":
            obs_pair = self.env.featurize_state_mdp(state)
            obs = obs_pair[agent_index]
        elif self.obs_type == "lossless_grid":
            obs_pair = self.env.lossless_state_encoding_mdp(state)
            obs = obs_pair[agent_index]
        else:
            raise ObservationConfigError(f"Unsupported observation type: {self.obs_type}")

        obs = np.asarray(obs, dtype=np.float32)
        if self.include_agent_index:
            return {"obs": obs, "agent_index": agent_index}
        return obs
