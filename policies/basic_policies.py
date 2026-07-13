"""Simple non-learning policies for the Overcooked competition runner.

These policies are intended for debugging and as weak baselines. They do not
train, do not use neural networks, and only read the raw Overcooked state.
"""

from __future__ import annotations


from collections import deque
from typing import Iterable

import numpy as np

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import Recipe


class StayPolicy(Agent):
    """Agent that always stays still.

    Useful as a sanity check: the episode should run, render, and log correctly,
    but the return should normally be zero unless the other agent can solve the
    task alone.
    """

    def action(self, state):
        return Action.STAY, {"policy_name": "stay"}


class RandomMotionPolicy(Agent):
    """Random policy over movement actions only.

    This policy samples from north/south/east/west/stay. It deliberately never
    emits interact, so it is useful for testing movement, collisions, rendering,
    and the runner without accidentally completing subtasks.
    """

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.actions = list(Action.MOTION_ACTIONS)
        if Action.STAY not in self.actions:
            self.actions.append(Action.STAY)

    def action(self, state):
        idx = int(self.rng.integers(0, len(self.actions)))
        action = self.actions[idx]
        return action, {"policy_name": "random_motion", "sampled_idx": idx}


class GreedyFullTaskPolicy(Agent):
    """Simple hand-written policy that tries to complete the whole soup pipeline.

    The policy is intentionally simple and non-optimal. It is meant to be a
    readable baseline, not a strong Overcooked solver.

    Priority order:
    1. If holding soup, deliver it.
    2. If holding a dish, pick up ready soup from a pot.
    3. If holding an ingredient, place it in a non-full pot.
    4. If empty-handed and soup is ready, get a dish.
    5. If empty-handed and a pot needs ingredients, get an onion.
    6. Otherwise wait or move minimally.

    Assumption: default onion soup tasks. This is appropriate for the standard
    old_dynamics=True layouts with three-onion recipes.
    """

    def __init__(
        self,
        ingredient: str = "onion",
        avoid_teammate: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        if ingredient not in {"onion", "tomato"}:
            raise ValueError("ingredient must be 'onion' or 'tomato'")
        self.ingredient = ingredient
        self.avoid_teammate = bool(avoid_teammate)
        self.rng = np.random.default_rng(seed)

    def action(self, state):
        mdp = self.mdp
        player = state.players[self.agent_index]
        held = player.held_object

        try:
            target = self._choose_target(state)
            if target is None:
                return Action.STAY, {"policy_name": "greedy_full_task", "target": None}

            action = self._move_or_interact_towards(state, target)
            return action, {
                "policy_name": "greedy_full_task",
                "held_object": None if held is None else held.name,
                "target": target,
            }
        except Exception as exc:
            # This is a baseline policy. It should never crash the runner.
            return Action.STAY, {
                "policy_name": "greedy_full_task",
                "fallback": True,
                "error": repr(exc),
            }

    # ---------------------------------------------------------------------
    # High-level task logic
    # ---------------------------------------------------------------------

    def _choose_target(self, state) -> tuple[int, int] | None:
        mdp = self.mdp
        player = state.players[self.agent_index]
        held = player.held_object
        pot_states = mdp.get_pot_states(state)

        if held is not None:
            if held.name == "soup":
                return self._nearest(player.position, mdp.get_serving_locations())

            if held.name == "dish":
                ready_pots = list(pot_states.get("ready", []))
                if ready_pots:
                    return self._nearest(player.position, ready_pots)
                # If no soup is ready yet, wait near a cooking/full pot when possible.
                almost_ready = list(pot_states.get("cooking", [])) + list(
                    pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", [])
                )
                if almost_ready:
                    return self._nearest(player.position, almost_ready)
                return None

            if held.name in {"onion", "tomato"}:
                return self._nearest(player.position, self._pots_that_can_accept_ingredients(state, pot_states))

            return None

        # Empty-handed: first see whether a useful object is already on a counter.
        ready_pots = list(pot_states.get("ready", []))
        if ready_pots:
            counter_dishes = self._counter_objects_by_name(state, "dish")
            if counter_dishes:
                return self._nearest(player.position, counter_dishes)
            dish_disps = mdp.get_dish_dispenser_locations()
            if dish_disps:
                return self._nearest(player.position, dish_disps)

        # If someone dropped a useful ingredient on a counter, prefer using it.
        pots_needing_items = self._pots_that_can_accept_ingredients(state, pot_states)
        if pots_needing_items:
            counter_ingredients = self._counter_objects_by_name(state, self.ingredient)
            if counter_ingredients:
                return self._nearest(player.position, counter_ingredients)
            ingredient_disps = self._ingredient_dispenser_locations()
            if ingredient_disps:
                return self._nearest(player.position, ingredient_disps)

        # If a pot is full but not cooking, face/interact with it. This matters for
        # non-old dynamics; old_dynamics starts cooking automatically.
        full_not_cooking = list(pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", []))
        if full_not_cooking:
            return self._nearest(player.position, full_not_cooking)

        # If soup is cooking, get ready for the next pickup by waiting near dishes.
        if list(pot_states.get("cooking", [])):
            dish_disps = mdp.get_dish_dispenser_locations()
            if dish_disps:
                return self._nearest(player.position, dish_disps)

        return None

    def _ingredient_dispenser_locations(self) -> list[tuple[int, int]]:
        if self.ingredient == "onion":
            return list(self.mdp.get_onion_dispenser_locations())
        return list(self.mdp.get_tomato_dispenser_locations())

    def _pots_that_can_accept_ingredients(self, state, pot_states) -> list[tuple[int, int]]:
        """Return pots that are empty or partially filled but not cooking/ready."""
        candidate_positions: list[tuple[int, int]] = []
        candidate_positions.extend(list(pot_states.get("empty", [])))
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            candidate_positions.extend(list(pot_states.get(f"{k}_items", [])))
        return candidate_positions

    def _counter_objects_by_name(self, state, object_name: str) -> list[tuple[int, int]]:
        return [obj.position for obj in state.objects.values() if obj.name == object_name]

    # ---------------------------------------------------------------------
    # Navigation and interaction
    # ---------------------------------------------------------------------

    def _move_or_interact_towards(self, state, target: tuple[int, int]):
        """Return an Overcooked action that moves/faces/interacts with target.

        The target is normally a non-walkable feature tile: onion dispenser, dish
        dispenser, pot, serving location, or a counter with an object. To interact
        with it, the player must stand on an adjacent walkable tile and face it.
        """
        player = state.players[self.agent_index]
        pos = player.position
        orientation = player.orientation

        if self._is_adjacent(pos, target):
            desired_direction = self._direction_from_to(pos, target)
            if orientation == desired_direction:
                return Action.INTERACT
            return desired_direction

        next_pos = self._next_step_towards_interaction_tile(state, target)
        if next_pos is None:
            return Action.STAY
        return Action.determine_action_for_change_in_pos(pos, next_pos)

    def _next_step_towards_interaction_tile(self, state, target: tuple[int, int]) -> tuple[int, int] | None:
        player = state.players[self.agent_index]
        start = player.position

        valid_positions = set(self.mdp.get_valid_player_positions())
        blocked = set()
        if self.avoid_teammate:
            for idx, other_player in enumerate(state.players):
                if idx != self.agent_index:
                    blocked.add(other_player.position)

        goals = [
            p
            for p in self._adjacent_positions(target)
            if p in valid_positions and p not in blocked
        ]
        if not goals:
            goals = [p for p in self._adjacent_positions(target) if p in valid_positions]
        if not goals:
            return None

        path = self._bfs_shortest_path(start, set(goals), valid_positions, blocked)
        if path is None or len(path) < 2:
            return None
        return path[1]

    def _bfs_shortest_path(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
        valid_positions: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()
            if pos in goals:
                return path

            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(pos, direction)
                if nxt not in valid_positions:
                    continue
                if nxt in blocked and nxt not in goals:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

        return None

    # ---------------------------------------------------------------------
    # Small geometry utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _nearest(origin: tuple[int, int], positions: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
        positions = list(positions)
        if not positions:
            return None
        return min(positions, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))

    @staticmethod
    def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _adjacent_positions(pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [Action.move_in_direction(pos, d) for d in Direction.ALL_DIRECTIONS]

    @staticmethod
    def _direction_from_to(a: tuple[int, int], b: tuple[int, int]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        direction = (dx, dy)
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError(f"Positions are not adjacent: {a} -> {b}")
        return direction
