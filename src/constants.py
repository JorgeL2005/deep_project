"""Shared constants for the Overcooked competition runner."""

from __future__ import annotations


from numbers import Integral
from overcooked_ai_py.mdp.actions import Action

# ---------------------------------------------------------------------------
# Robust action mapping
# ---------------------------------------------------------------------------
# Different Overcooked-AI versions expose Action.INDEX_TO_ACTION differently:
# some versions use a dict {0: action, ...}, while others use a list/tuple.
# The starter code normalizes both variants into one dictionary.
_raw_index_to_action = Action.INDEX_TO_ACTION
if hasattr(_raw_index_to_action, "items"):
    INDEX_TO_OVERCOOKED_ACTION = {int(idx): action for idx, action in _raw_index_to_action.items()}
else:
    INDEX_TO_OVERCOOKED_ACTION = {idx: action for idx, action in enumerate(_raw_index_to_action)}

NUM_ACTIONS = len(INDEX_TO_OVERCOOKED_ACTION)
OVERCOOKED_ACTION_TO_INDEX = {action: idx for idx, action in INDEX_TO_OVERCOOKED_ACTION.items()}

# Public integer action convention exposed to students.
# This follows Overcooked-AI's Action.INDEX_TO_ACTION order in the standard repo:
# 0 north, 1 south, 2 east, 3 west, 4 stay, 5 interact.
ACTION_NAME_TO_INDEX = {
    "north": 0,
    "up": 0,
    "south": 1,
    "down": 1,
    "east": 2,
    "right": 2,
    "west": 3,
    "left": 3,
    "stay": 4,
    "wait": 4,
    "noop": 4,
    "no_op": 4,
    "interact": 5,
    "space": 5,
}

ACTION_INDEX_TO_NAME = {
    0: "north",
    1: "south",
    2: "east",
    3: "west",
    4: "stay",
    5: "interact",
}


def action_index_to_overcooked_action(action_index: int):
    """Convert public integer action into an Overcooked-AI action object."""
    if isinstance(action_index, Integral):
        action_index = int(action_index)
    else:
        raise TypeError(f"Action must be an int in [0, {NUM_ACTIONS - 1}], got {type(action_index)}")

    if action_index not in INDEX_TO_OVERCOOKED_ACTION:
        raise ValueError(f"Action index must be one of {sorted(INDEX_TO_OVERCOOKED_ACTION)}, got {action_index}")
    return INDEX_TO_OVERCOOKED_ACTION[action_index]


def overcooked_action_to_index(action) -> int:
    """Convert an Overcooked-AI action object into the public integer convention."""
    if action not in OVERCOOKED_ACTION_TO_INDEX:
        raise ValueError(f"Unknown Overcooked action: {action}")
    return int(OVERCOOKED_ACTION_TO_INDEX[action])


def action_name_to_index(action_name: str) -> int:
    """Convert a human-readable action name into the public integer convention."""
    key = str(action_name).strip().lower()
    if key not in ACTION_NAME_TO_INDEX:
        raise ValueError(f"Unknown action name '{action_name}'. Valid names: {sorted(ACTION_NAME_TO_INDEX)}")
    idx = ACTION_NAME_TO_INDEX[key]
    if idx not in INDEX_TO_OVERCOOKED_ACTION:
        raise ValueError(
            f"Action name '{action_name}' maps to index {idx}, but this Overcooked-AI installation "
            f"only exposes indices {sorted(INDEX_TO_OVERCOOKED_ACTION)}"
        )
    return idx
