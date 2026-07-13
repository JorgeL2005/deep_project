"""Keyboard-controlled human policy for collecting demonstrations.

This policy is intentionally simple: each call reads the current keyboard state
and returns one action index using the public convention:

    0 north, 1 south, 2 east, 3 west, 4 stay, 5 interact.

It is meant to be used with rendering.mode: window.
"""

from __future__ import annotations


from typing import Any

from overcooked_ai_py.agents.agent import Agent

from src.constants import action_index_to_overcooked_action, action_name_to_index


DEFAULT_KEYMAP = {
    "up": ["up", "w"],
    "down": ["down", "s"],
    "left": ["left", "a"],
    "right": ["right", "d"],
    "interact": ["space", "e", "return"],
    "stay": [],
}

DEFAULT_PRIORITY = ["interact", "up", "down", "left", "right", "stay"]

KEY_ALIASES = {
    "esc": "escape",
    "enter": "return",
    "spacebar": "space",
    "arrow_up": "up",
    "arrow_down": "down",
    "arrow_left": "left",
    "arrow_right": "right",
}


class HumanKeyboardPolicy(Agent):
    """Agent controlled by the keyboard.

    YAML options:
        keymap:
          up: ["up", "w"]
          down: ["down", "s"]
          left: ["left", "a"]
          right: ["right", "d"]
          interact: ["space", "e"]
        priority: ["interact", "up", "down", "left", "right", "stay"]

    If multiple keys are pressed, the first action in `priority` wins.
    If no mapped key is pressed, the policy returns stay.
    """

    def __init__(self, keymap: dict[str, Any] | None = None, priority: list[str] | None = None):
        super().__init__()
        self.keymap = _merge_keymap(DEFAULT_KEYMAP, keymap or {})
        self.priority = [str(a).lower() for a in (priority or DEFAULT_PRIORITY)]
        self._pygame = None
        self._key_codes_by_action: dict[str, list[int]] | None = None

    def action(self, state):
        pygame = self._ensure_pygame()
        pygame.event.pump()
        pressed = pygame.key.get_pressed()

        for action_name in self.priority:
            if action_name == "stay":
                continue
            for key_code in self._key_codes_by_action.get(action_name, []):
                if pressed[key_code]:
                    action_idx = action_name_to_index(action_name)
                    return action_index_to_overcooked_action(action_idx), {
                        "policy_name": "human_keyboard",
                        "action_index": action_idx,
                        "source": "keyboard",
                    }

        stay_idx = action_name_to_index("stay")
        return action_index_to_overcooked_action(stay_idx), {
            "policy_name": "human_keyboard",
            "action_index": stay_idx,
            "source": "keyboard",
        }

    def _ensure_pygame(self):
        if self._pygame is not None:
            return self._pygame

        import pygame

        if not pygame.get_init():
            pygame.init()
        self._pygame = pygame
        self._key_codes_by_action = {
            action: [_key_name_to_code(pygame, key_name) for key_name in key_names]
            for action, key_names in self.keymap.items()
        }
        return pygame


def _merge_keymap(default: dict[str, list[str]], override: dict[str, Any]) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in default.items()}
    for action, value in override.items():
        action_key = str(action).lower()
        if value is None:
            merged[action_key] = []
        elif isinstance(value, str):
            merged[action_key] = [value]
        else:
            merged[action_key] = [str(v) for v in value]
    return merged


def _key_name_to_code(pygame, key_name: str) -> int:
    key = str(key_name).strip().lower().replace(" ", "_")
    key = KEY_ALIASES.get(key, key)

    # Common explicit aliases first, because pygame.key.key_code is strict about names.
    explicit = {
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "space": pygame.K_SPACE,
        "return": pygame.K_RETURN,
        "escape": pygame.K_ESCAPE,
        "tab": pygame.K_TAB,
        "shift": pygame.K_LSHIFT,
        "ctrl": pygame.K_LCTRL,
    }
    if key in explicit:
        return explicit[key]

    try:
        return pygame.key.key_code(key)
    except Exception as exc:
        raise ValueError(f"Unknown pygame key name: {key_name!r}") from exc
