"""Rendering utilities for debug, live visualization, GIF export and replay."""

from __future__ import annotations


import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


class Renderer:
    """Renderer for Overcooked episodes.

    Supported modes:
        none:      no visualization.
        terminal:  print textual state in the terminal.
        rgb_array: collect rendered frames as numpy arrays.
        gif:       collect rendered frames and save a GIF at the end.
        window:    show a live pygame window while the episode runs.

    Notes:
        - mode=window is intended for local debugging, not for official evaluation.
        - save_gif can be enabled together with window if you want live playback and GIF export.
    """

    VALID_MODES = {"none", "terminal", "rgb_array", "gif", "window"}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.mode = str(self.config.get("mode", "none"))
        self.fps = float(self.config.get("fps", 5))
        self.save_gif = bool(self.config.get("save_gif", False)) or self.mode == "gif"
        self.gif_path = Path(self.config.get("gif_path", "outputs/episode.gif"))
        self.window_caption = str(self.config.get("window_caption", "Overcooked-AI"))
        self.quit_keys = self.config.get("quit_keys", ["escape", "q"])
        if isinstance(self.quit_keys, str):
            self.quit_keys = [self.quit_keys]
        self._quit_key_codes = None

        self.frames: list[np.ndarray] = []
        self.visualizer = StateVisualizer()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._closed_by_user = False

        if self.mode not in self.VALID_MODES:
            raise ValueError(f"rendering.mode must be one of: {sorted(self.VALID_MODES)}")

    @property
    def closed_by_user(self) -> bool:
        """Whether the pygame window was closed manually."""
        return self._closed_by_user

    def reset(self):
        self.frames.clear()
        self._closed_by_user = False

    def maybe_render(self, env, timestep: int, joint_action=None, reward: float | None = None):
        if self.mode == "none" and not self.save_gif:
            return

        if self.mode == "terminal":
            print(f"\nTimestep {timestep}")
            if joint_action is not None:
                print(f"Joint action: {joint_action}; reward={reward}")
            print(env)
            self._sleep_if_needed()
            return

        if self.mode == "window":
            surface = self.render_surface(env)
            self._show_surface_in_window(surface)
            if self.save_gif:
                self.frames.append(self.surface_to_array(surface))
            self._sleep_if_needed()
            return

        if self.mode in {"rgb_array", "gif"} or self.save_gif:
            frame = self.render_frame(env)
            self.frames.append(frame)
            self._sleep_if_needed()

    def render_surface(self, env):
        """Render the current environment state as a pygame Surface."""
        rewards_dict = {}
        for key, value in env.game_stats.items():
            if key in ["cumulative_shaped_rewards_by_agent", "cumulative_sparse_rewards_by_agent"]:
                rewards_dict[key] = value

        return self.visualizer.render_state(
            state=env.state,
            grid=env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(env.state, **rewards_dict),
        )

    def render_frame(self, env) -> np.ndarray:
        """Render the current environment state as an H x W x C numpy array."""
        surface = self.render_surface(env)
        return self.surface_to_array(surface)

    def surface_to_array(self, surface) -> np.ndarray:
        """Convert a pygame Surface to an H x W x C numpy array."""
        import pygame

        buffer = pygame.surfarray.array3d(surface)
        image = np.flip(np.rot90(buffer, 3), 1)
        return image

    def _show_surface_in_window(self, surface):
        """Display a pygame Surface in a live window."""
        import pygame

        if self._pygame is None:
            pygame.init()
            self._pygame = pygame
            self._clock = pygame.time.Clock()
            self._quit_key_codes = {_key_name_to_code(pygame, key_name) for key_name in self.quit_keys}

        width, height = surface.get_size()
        if self._screen is None:
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(self.window_caption)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed_by_user = True
                return
            if event.type == pygame.KEYDOWN and event.key in self._quit_key_codes:
                self._closed_by_user = True
                return

        self._screen.blit(surface, (0, 0))
        pygame.display.flip()

        if self._clock is not None and self.fps > 0:
            self._clock.tick(self.fps)

    def close(self):
        if self.save_gif and self.frames:
            self.gif_path.parent.mkdir(parents=True, exist_ok=True)
            duration = 1.0 / max(self.fps, 1e-6)
            imageio.mimsave(self.gif_path, self.frames, duration=duration)
            print(f"Saved GIF to {self.gif_path}")

        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    def _sleep_if_needed(self):
        if self.mode == "window":
            return
        if self.fps > 0 and self.mode in {"terminal", "rgb_array", "gif"}:
            time.sleep(1.0 / self.fps)


def _key_name_to_code(pygame, key_name: str) -> int:
    """Translate a small set of human-readable key names to pygame key codes."""
    key = str(key_name).strip().lower().replace(" ", "_")
    aliases = {
        "esc": "escape",
        "enter": "return",
        "spacebar": "space",
        "arrow_up": "up",
        "arrow_down": "down",
        "arrow_left": "left",
        "arrow_right": "right",
    }
    key = aliases.get(key, key)
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
        raise ValueError(f"Unknown pygame key name in rendering.quit_keys: {key_name!r}") from exc
