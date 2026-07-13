"""Logging utilities for episodes and step-level traces."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import ensure_dir


@dataclass
class StepRecord:
    episode_id: int
    timestep: int
    layout_name: str
    role_swap: bool
    action_0: str
    action_1: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class CompetitionLogger:
    def __init__(self, logging_config: dict[str, Any] | None = None):
        self.config = logging_config or {}
        self.output_dir = ensure_dir(self.config.get("output_dir", "outputs/run"))
        self.save_step_log = bool(self.config.get("save_step_log", True))
        self.save_episode_summary = bool(self.config.get("save_episode_summary", True))
        self.save_trajectory_pickle = bool(self.config.get("save_trajectory_pickle", False))
        self.step_records: list[dict[str, Any]] = []
        self.episode_summaries: list[dict[str, Any]] = []

    def log_step(self, record: StepRecord):
        if not self.save_step_log:
            return
        self.step_records.append(
            {
                "episode_id": record.episode_id,
                "timestep": record.timestep,
                "layout_name": record.layout_name,
                "role_swap": record.role_swap,
                "action_0": record.action_0,
                "action_1": record.action_1,
                "reward": record.reward,
                "done": record.done,
                "sparse_r_by_agent": record.info.get("sparse_r_by_agent"),
                "shaped_r_by_agent": record.info.get("shaped_r_by_agent"),
            }
        )

    def log_episode(self, summary: dict[str, Any]):
        if self.save_episode_summary:
            self.episode_summaries.append(summary)

    def save_trajectory(self, episode_id: int, trajectory: list[Any]):
        if not self.save_trajectory_pickle:
            return
        traj_dir = ensure_dir(self.output_dir / "trajectories")
        with (traj_dir / f"episode_{episode_id:04d}.pkl").open("wb") as f:
            pickle.dump(trajectory, f)

    def flush(self):
        if self.save_step_log and self.step_records:
            self._write_csv(self.output_dir / "steps.csv", self.step_records)
        if self.save_episode_summary and self.episode_summaries:
            self._write_csv(self.output_dir / "episodes.csv", self.episode_summaries)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]):
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
