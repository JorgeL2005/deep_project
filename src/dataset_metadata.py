"""Dataset metadata utilities for demonstration collection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import INDEX_TO_OVERCOOKED_ACTION, OVERCOOKED_ACTION_TO_INDEX, ACTION_INDEX_TO_NAME
from src.environment import load_custom_layout_dict


def to_jsonable(value: Any) -> Any:
    """Convert common Python/NumPy objects to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    # Overcooked objects often have useful reprs but are not JSON-serializable.
    return repr(value)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _terrain_matrix_to_rows(terrain_mtx: Any) -> list[str] | None:
    """Convert mdp.terrain_mtx to rows when available.

    In Overcooked-AI, terrain_mtx is usually a list of rows where each element is
    a terrain character. Some versions store strings, others lists/tuples.
    """
    if terrain_mtx is None:
        return None

    rows: list[str] = []
    try:
        for row in terrain_mtx:
            if isinstance(row, str):
                rows.append(row)
            else:
                rows.append("".join(str(cell) for cell in row))
    except Exception:
        return None

    if not rows:
        return None
    return rows


def _positions_to_lists(obj: Any) -> Any:
    """Make position dictionaries/lists easier to serialize and read."""
    if isinstance(obj, dict):
        return {str(k): _positions_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_positions_to_lists(v) for v in obj]
    if isinstance(obj, list):
        return [_positions_to_lists(v) for v in obj]
    return obj


def extract_layout_metadata(env: Any, environment_config: dict[str, Any]) -> dict[str, Any]:
    """Extract as much scenario/layout information as possible.

    The function is intentionally defensive because different versions of
    Overcooked-AI expose slightly different attributes.
    """
    mdp = env.mdp
    layout_file = environment_config.get("layout_file")
    layout_name = _safe_getattr(mdp, "layout_name", environment_config.get("layout_name"))

    terrain_rows = _terrain_matrix_to_rows(_safe_getattr(mdp, "terrain_mtx"))
    height = len(terrain_rows) if terrain_rows is not None else _safe_getattr(mdp, "height")
    width = len(terrain_rows[0]) if terrain_rows else _safe_getattr(mdp, "width")

    layout_dict = None
    if layout_file:
        try:
            layout_dict = load_custom_layout_dict(layout_file)
        except Exception as exc:
            layout_dict = {"error_loading_layout_file": str(exc)}

    metadata = {
        "layout_name": layout_name,
        "layout_file": layout_file,
        "horizon": _safe_getattr(env, "horizon", environment_config.get("horizon")),
        "old_dynamics": environment_config.get("old_dynamics"),
        "width": width,
        "height": height,
        "grid_rows": terrain_rows,
        "grid_string": "\n".join(terrain_rows) if terrain_rows else None,
        "terrain_pos_dict": _positions_to_lists(_safe_getattr(mdp, "terrain_pos_dict")),
        "start_player_positions": _positions_to_lists(_safe_getattr(mdp, "start_player_positions")),
        "start_order_list": to_jsonable(_safe_getattr(mdp, "start_order_list")),
        "rew_shaping_params": to_jsonable(_safe_getattr(mdp, "rew_shaping_params")),
        "custom_layout_dict": to_jsonable(layout_dict),
    }
    return to_jsonable(metadata)


def compact_policy_metadata(policies_config: dict[str, Any]) -> dict[str, Any]:
    """Keep policy metadata useful without dumping huge objects."""
    result: dict[str, Any] = {}
    for agent_key, cfg in (policies_config or {}).items():
        keep = {}
        for key in [
            "type",
            "name",
            "path",
            "class_name",
            "model_path",
            "device",
            "ingredient",
            "avoid_teammate",
            "random_action_prob",
            "max_action_time_ms",
            "invalid_action",
            "timeout_action",
            "keymap",
            "priority",
        ]:
            if key in cfg:
                keep[key] = cfg[key]
        result[str(agent_key)] = keep
    return to_jsonable(result)


def build_dataset_metadata(
    *,
    config: dict[str, Any],
    env: Any,
    obs_builder: Any,
    num_episodes: int,
    episode_seeds: list[int],
    role_swaps: list[bool],
) -> dict[str, Any]:
    """Build a self-contained metadata block for imitation datasets."""
    environment_config = config.get("environment", {}) or {}
    observation_config = config.get("observation", {}) or {}
    execution_config = config.get("execution", {}) or {}
    data_config = config.get("data_collection", {}) or {}

    index_to_action = {int(k): ACTION_INDEX_TO_NAME.get(int(k), repr(v)) for k, v in INDEX_TO_OVERCOOKED_ACTION.items()}
    action_to_index = {repr(k): int(v) for k, v in OVERCOOKED_ACTION_TO_INDEX.items()}

    metadata = {
        "format_version": "overcooked_demonstrations_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "human_or_policy_demonstrations_for_imitation_learning",
        "mode": config.get("mode"),
        "seed": config.get("seed"),
        "environment": to_jsonable(environment_config),
        "layout": extract_layout_metadata(env, environment_config),
        "observation": {
            "config": to_jsonable(observation_config),
            "type": getattr(obs_builder, "obs_type", observation_config.get("type")),
            "include_agent_index": observation_config.get("include_agent_index"),
            "note": "Each record stores obs and, when enabled, next_obs as produced by ObservationBuilder.",
        },
        "actions": {
            "convention": "0=north/up, 1=south/down, 2=east/right, 3=west/left, 4=stay, 5=interact",
            "index_to_action_repr": index_to_action,
            "action_repr_to_index": action_to_index,
        },
        "policies": compact_policy_metadata(config.get("policies", {}) or {}),
        "execution": {
            "config": to_jsonable(execution_config),
            "num_episodes": int(num_episodes),
            "episode_seeds": [int(s) for s in episode_seeds[:num_episodes]],
            "role_swaps": [bool(x) for x in role_swaps],
        },
        "data_collection": {
            "enabled": bool(data_config.get("enabled", False)),
            "record_agent_indices": [int(i) for i in data_config.get("record_agent_indices", [])],
            "include_next_obs": bool(data_config.get("include_next_obs", True)),
            "include_info": bool(data_config.get("include_info", False)),
            "output_path": data_config.get("output_path"),
            "npz_path": data_config.get("npz_path"),
            "metadata_json_path": data_config.get("metadata_json_path"),
        },
    }
    return to_jsonable(metadata)
