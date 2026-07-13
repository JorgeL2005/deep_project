"""Utilities for saving human demonstration datasets."""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ensure_dir
from src.dataset_metadata import to_jsonable


class DemonstrationRecorder:
    """Collect transitions suitable for imitation learning.

    The recorder stores one record per selected agent and timestep:

        obs_t, action_t, reward_t, done_t, optional obs_{t+1}

    For imitation learning, the most important fields are `obs` and `action`.
    Use observation.type=featurized or observation.type=lossless_grid if you want
    directly trainable tensors. Use observation.type=state only for debugging or
    if you plan to build your own encoder later.

    The pickle payload has the structure:

        {
            "metadata": {...},
            "records": [...],
            "episode_summaries": [...]
        }
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.record_agent_indices = [int(i) for i in self.config.get("record_agent_indices", [1])]
        self.include_next_obs = bool(self.config.get("include_next_obs", True))
        self.include_info = bool(self.config.get("include_info", False))
        self.auto_name = bool(self.config.get("auto_name", True))
        self.overwrite = bool(self.config.get("overwrite", False))
        self.output_dir = Path(self.config.get("output_dir", "data/demonstrations"))
        self.records: list[dict[str, Any]] = []
        self.episode_summaries: list[dict[str, Any]] = []
        self.output_path: Path | None = None
        self.npz_path: Path | None = None
        self.metadata_json_path: Path | None = None

    def record_step(
        self,
        *,
        obs_builder,
        episode_id: int,
        timestep: int,
        layout_name: str,
        role_swap: bool,
        episode_seed: int | None,
        state,
        next_state,
        joint_action_indices: list[int],
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ):
        if not self.enabled:
            return

        for agent_index in self.record_agent_indices:
            obs = obs_builder(state, agent_index)
            next_obs = obs_builder(next_state, agent_index) if self.include_next_obs else None
            record = {
                "episode_id": int(episode_id),
                "episode_seed": None if episode_seed is None else int(episode_seed),
                "timestep": int(timestep),
                "layout_name": layout_name,
                "role_swap": bool(role_swap),
                "agent_index": int(agent_index),
                "obs": obs,
                "action": int(joint_action_indices[agent_index]),
                "reward": float(reward),
                "done": bool(done),
            }
            if self.include_next_obs:
                record["next_obs"] = next_obs
            if self.include_info:
                record["info"] = info or {}
            self.records.append(record)

    def record_episode(self, summary: dict[str, Any]):
        if self.enabled:
            self.episode_summaries.append(dict(summary))

    def flush(self, metadata: dict[str, Any] | None = None):
        if not self.enabled:
            return

        metadata = to_jsonable(metadata or {})
        self._resolve_output_paths(metadata)
        assert self.output_path is not None
        assert self.metadata_json_path is not None

        metadata = dict(metadata)
        metadata["files"] = {
            "pickle_path": str(self.output_path),
            "npz_path": None if self.npz_path is None else str(self.npz_path),
            "metadata_json_path": str(self.metadata_json_path),
            "auto_name": self.auto_name,
            "overwrite": self.overwrite,
        }
        metadata = to_jsonable(metadata)

        ensure_dir(self.output_path.parent)
        payload = {
            "metadata": metadata,
            "records": self.records,
            "episode_summaries": self.episode_summaries,
        }
        with self.output_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved demonstrations to {self.output_path} ({len(self.records)} records)")

        self._save_metadata_json(self.metadata_json_path, metadata)

        if self.npz_path is not None:
            self._try_save_npz(self.npz_path, metadata)

    def _resolve_output_paths(self, metadata: dict[str, Any]):
        """Resolve pickle/NPZ/JSON output paths without accidental overwrites.

        By default, file names are generated automatically from the layout name,
        current timestamp and a numeric suffix when needed. This prevents students
        from losing previous recordings by re-running the same command.
        """
        if not self.auto_name:
            output_path = Path(self.config.get("output_path", "data/demonstrations/demo.pkl"))
            npz_cfg = self.config.get("npz_path")
            metadata_cfg = self.config.get("metadata_json_path")
            npz_path = Path(npz_cfg) if npz_cfg else None
            metadata_json_path = Path(metadata_cfg) if metadata_cfg else output_path.with_suffix(".metadata.json")
            if not self.overwrite:
                output_path = _next_available_path(output_path)
                if npz_path is not None:
                    npz_path = output_path.with_suffix(".npz")
                metadata_json_path = output_path.with_suffix(".metadata.json")
            self.output_path = output_path
            self.npz_path = npz_path
            self.metadata_json_path = metadata_json_path
            return

        output_dir = ensure_dir(self.output_dir)
        layout_label = _layout_label_from_metadata(metadata)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{layout_label}_{timestamp}"
        save_npz = bool(self.config.get("save_npz", True))
        output_path, npz_path, metadata_json_path = _next_available_recording_paths(
            output_dir=output_dir,
            base_name=base_name,
            save_npz=save_npz,
        )
        self.output_path = output_path
        self.npz_path = npz_path
        self.metadata_json_path = metadata_json_path

    def _save_metadata_json(self, path: Path, metadata: dict[str, Any]):
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {path}")

    def _try_save_npz(self, npz_path: Path, metadata: dict[str, Any]):
        """Save a tensor-only NPZ when observations are numeric arrays.

        This is convenient for quick behavioral cloning experiments. If the
        observations are not numeric arrays, the pickle file remains the source
        of truth and NPZ export is skipped.
        """
        obs_arrays = []
        next_obs_arrays = []
        actions = []
        rewards = []
        dones = []
        episode_ids = []
        episode_seeds = []
        timesteps = []
        agent_indices = []
        role_swaps = []

        include_next_obs = self.include_next_obs
        for record in self.records:
            arr = _extract_numeric_obs(record["obs"])
            if arr is None:
                print("Skipping NPZ export because at least one observation is not a numeric array")
                return
            obs_arrays.append(arr)

            if include_next_obs:
                next_arr = _extract_numeric_obs(record["next_obs"])
                if next_arr is None:
                    print("Skipping NPZ export because at least one next_obs is not a numeric array")
                    return
                next_obs_arrays.append(next_arr)

            actions.append(record["action"])
            rewards.append(record["reward"])
            dones.append(record["done"])
            episode_ids.append(record["episode_id"])
            episode_seeds.append(-1 if record.get("episode_seed") is None else record["episode_seed"])
            timesteps.append(record["timestep"])
            agent_indices.append(record["agent_index"])
            role_swaps.append(record["role_swap"])

        if not obs_arrays:
            return

        ensure_dir(npz_path.parent)
        arrays = {
            "obs": np.stack(obs_arrays).astype(np.float32, copy=False),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.bool_),
            "episode_ids": np.asarray(episode_ids, dtype=np.int64),
            "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
            "timesteps": np.asarray(timesteps, dtype=np.int64),
            "agent_indices": np.asarray(agent_indices, dtype=np.int64),
            "role_swaps": np.asarray(role_swaps, dtype=np.bool_),
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        }
        if include_next_obs:
            arrays["next_obs"] = np.stack(next_obs_arrays).astype(np.float32, copy=False)

        np.savez_compressed(npz_path, **arrays)
        print(f"Saved tensor dataset to {npz_path}")


def _layout_label_from_metadata(metadata: dict[str, Any]) -> str:
    layout = metadata.get("layout", {}) if isinstance(metadata, dict) else {}
    env = metadata.get("environment", {}) if isinstance(metadata, dict) else {}

    label = layout.get("layout_name") or env.get("layout_name")
    if not label:
        layout_file = layout.get("layout_file") or env.get("layout_file")
        if layout_file:
            label = Path(str(layout_file)).stem
    if not label:
        label = "custom_layout"
    return _slugify(str(label))


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "layout"


def _next_available_recording_paths(
    *,
    output_dir: Path,
    base_name: str,
    save_npz: bool,
) -> tuple[Path, Path | None, Path]:
    idx = 1
    while True:
        suffix = "" if idx == 1 else f"_{idx:02d}"
        stem = f"{base_name}{suffix}"
        pkl_path = output_dir / f"{stem}.pkl"
        npz_path = output_dir / f"{stem}.npz" if save_npz else None
        metadata_json_path = output_dir / f"{stem}.metadata.json"
        candidates = [pkl_path, metadata_json_path]
        if npz_path is not None:
            candidates.append(npz_path)
        if not any(path.exists() for path in candidates):
            return pkl_path, npz_path, metadata_json_path
        idx += 1


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}_{idx:02d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _extract_numeric_obs(obs) -> np.ndarray | None:
    if isinstance(obs, dict) and "obs" in obs:
        obs = obs["obs"]
    arr = np.asarray(obs)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(np.float32, copy=False)
