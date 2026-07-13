"""Environment and layout creation utilities."""

from __future__ import annotations


import ast
from pathlib import Path
from typing import Any

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


class LayoutConfigError(ValueError):
    """Raised when the layout configuration cannot be loaded."""


def _clean_grid_string(grid: str) -> list[str]:
    rows = [row.rstrip("\n") for row in grid.split("\n")]
    rows = [row.strip() for row in rows if row.strip() != ""]
    if not rows:
        raise LayoutConfigError("Layout grid is empty")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise LayoutConfigError(f"Layout rows must have equal width, got widths={sorted(widths)}")
    return rows


def load_custom_layout_dict(layout_file: str | Path) -> dict[str, Any]:
    """Load an Overcooked-AI .layout file as a Python literal dictionary.

    Official Overcooked-AI layout files are Python-literal dictionaries, not YAML.
    This function uses ast.literal_eval instead of eval for safety.
    """
    layout_path = Path(layout_file)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")
    text = layout_path.read_text(encoding="utf-8")
    layout_dict = ast.literal_eval(text)
    if not isinstance(layout_dict, dict):
        raise LayoutConfigError(f"Layout file must contain a dict, got {type(layout_dict)}")
    if "grid" not in layout_dict:
        raise LayoutConfigError("Layout dictionary must contain a 'grid' field")
    return layout_dict


def build_mdp(environment_config: dict[str, Any]) -> OvercookedGridworld:
    """Build an OvercookedGridworld from a built-in layout name or a custom file."""
    layout_name = environment_config.get("layout_name")
    layout_file = environment_config.get("layout_file")
    old_dynamics = bool(environment_config.get("old_dynamics", True))

    mdp_overwrites = {
        "old_dynamics": old_dynamics,
    }

    # Optional recipe/dynamics overrides can be added directly under environment.mdp_overrides.
    mdp_overwrites.update(environment_config.get("mdp_overrides", {}) or {})

    if layout_file:
        layout_dict = load_custom_layout_dict(layout_file)
        grid = _clean_grid_string(layout_dict.pop("grid"))
        layout_dict.setdefault("layout_name", Path(layout_file).stem)
        return OvercookedGridworld.from_grid(
            layout_grid=grid,
            base_layout_params=layout_dict,
            params_to_overwrite=mdp_overwrites,
        )

    if not layout_name:
        raise LayoutConfigError("Set either environment.layout_name or environment.layout_file")

    return OvercookedGridworld.from_layout_name(str(layout_name), **mdp_overwrites)


def build_env(environment_config: dict[str, Any]) -> OvercookedEnv:
    """Build the OvercookedEnv wrapper."""
    mdp = build_mdp(environment_config)
    horizon = int(environment_config.get("horizon", 400))
    return OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
