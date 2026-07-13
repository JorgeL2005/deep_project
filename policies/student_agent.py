"""Adaptive planning agent for the Overcooked-AI competition.

Design rationale (state of the art, mid-2026):

Learned zero-shot-coordination methods (PPO_BC / Human-Aware RL, FCP, MEP,
TrajeDi, COLE, E3T) train one policy *per layout* and recent benchmarks
(the Overcooked Generalisation Challenge, TMLR 2025; OvercookedV2, ICLR 2025)
show that they generalise poorly to layouts never seen during training.
LLM-based planners (ProAgent) adapt well but cannot meet a 100 ms/action
budget. For a competition on *unknown* layouts, with *unknown* partners,
role swapping and a hard per-action time limit, the strongest practical
approach is an ad-hoc-teamwork planning agent:

- Generalises by construction: it reads the layout from the raw state and
  plans with BFS, so any grid, recipe set (onion/tomato mixes), cook times
  and pot counts work out of the box.
- Partner-adaptive: it infers what the teammate is doing from its held
  object and position and picks complementary subtasks (no duplicated
  dishes, no double-fetching).
- Handles "forced coordination" layouts: it detects disconnected regions
  and passes ingredients/dishes/soups over counters reachable by both
  players.
- Deadlock-safe: detects being blocked and side-steps stochastically.
- Fast: a handful of BFS runs over <=200 tiles per step (well under 1 ms).

Runner interface (see policies/template.py):

    StudentAgent(config) ; reset() ; act(obs) -> int in {0..5}

IMPORTANT: this agent requires the raw-state observation. In the YAML config
use:

    observation:
      type: state
      include_agent_index: true

Action convention: 0=north/up, 1=south/down, 2=east/right, 3=west/left,
4=stay, 5=interact.
"""

from __future__ import annotations

import sys
from collections import Counter, deque

import numpy as np

# Public action indices (match src/constants.py and Overcooked's
# Action.INDEX_TO_ACTION order).
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
DIR_TO_ACTION = {(0, -1): NORTH, (0, 1): SOUTH, (1, 0): EAST, (-1, 0): WEST}
ALL_DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]

INF = float("inf")


def _add(pos, d):
    return (pos[0] + d[0], pos[1] + d[1])


class StudentAgent:
    """Hierarchical planning agent with partner-aware task allocation."""

    def __init__(self, config=None):
        self.config = config or {}
        self._debug = bool(self.config.get("debug", False))
        seed = self.config.get("seed", 0)
        self.rng = np.random.default_rng(None if seed is None else int(seed))
        self._warned_bad_obs = False
        self._static = None  # cached per-layout static info
        self.reset()

    # ------------------------------------------------------------------
    # Runner API
    # ------------------------------------------------------------------

    def reset(self):
        self._pos_history = deque(maxlen=8)
        self._move_intent_history = deque(maxlen=8)
        self._partner_history = deque(maxlen=8)
        self._static = None
        self._start_pos = None
        self._last_feature = None  # last feature we interacted with (retreat anchor)
        self._pocket_wait = 0  # steps spent yielding in a pocket
        self._partner_wall = False  # treat a permanently static partner as a wall
        self._retreat_target = None  # committed give-way pocket
        self._standoff_partner_pos = None  # partner position when standoff began
        self._standoff_wait = 0  # steps holding ground with right-of-way

    def act(self, obs) -> int:
        try:
            state, mdp, agent_index = self._parse_obs(obs)
            if state is None:
                return STAY
            return int(self._decide(state, mdp, agent_index))
        except Exception as exc:  # never crash the runner
            if not self._warned_bad_obs:
                import traceback

                print(f"[student_agent] internal error, defaulting to stay: {exc!r}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self._warned_bad_obs = True
            return STAY

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------

    def _parse_obs(self, obs):
        if isinstance(obs, dict) and "state" in obs and "mdp" in obs:
            agent_index = int(obs.get("agent_index", self.config.get("agent_index", 0)))
            return obs["state"], obs["mdp"], agent_index
        if not self._warned_bad_obs:
            print(
                "[student_agent] This agent needs observation.type: 'state' in the "
                "YAML config (got a featurized/grid observation). Falling back to 'stay'.",
                file=sys.stderr,
            )
            self._warned_bad_obs = True
        return None, None, None

    # ------------------------------------------------------------------
    # Static layout info (cached)
    # ------------------------------------------------------------------

    def _layout_static(self, mdp):
        if self._static is not None and self._static["mdp_id"] == id(mdp):
            return self._static
        self._static = {
            "mdp_id": id(mdp),
            "valid": set(mdp.get_valid_player_positions()),
            "pots": list(mdp.get_pot_locations()),
            "onions": list(mdp.get_onion_dispenser_locations()),
            "tomatoes": list(mdp.get_tomato_dispenser_locations()),
            "dishes": list(mdp.get_dish_dispenser_locations()),
            "serving": list(mdp.get_serving_locations()),
            "counters": list(mdp.get_counter_locations()),
            "max_ingredients": self._max_ingredients(mdp),
        }
        return self._static

    @staticmethod
    def _max_ingredients(mdp):
        try:
            from overcooked_ai_py.mdp.overcooked_mdp import Recipe

            return int(Recipe.MAX_NUM_INGREDIENTS)
        except Exception:
            return 3

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def _decide(self, state, mdp, idx) -> int:
        st = self._layout_static(mdp)
        me = state.players[idx]
        partner = state.players[1 - idx] if len(state.players) > 1 else None
        my_pos = me.position
        held = me.held_object.name if me.held_object is not None else None
        partner_pos = partner.position if partner is not None else None
        partner_held = (
            partner.held_object.name
            if partner is not None and partner.held_object is not None
            else None
        )

        # A partner that resumes moving stops being treated as a wall.
        if self._partner_history and self._partner_history[-1] != partner_pos:
            self._partner_wall = False
            self._pocket_wait = 0

        # Reachability maps.
        blocked_by_partner = {partner_pos} if partner_pos else set()
        dist_free = self._bfs(my_pos, st["valid"], set())
        dist_block = self._bfs(my_pos, st["valid"], blocked_by_partner)
        if self._partner_wall:
            # The partner has proven immobile: plan around it, not through it.
            dist_free = dist_block
        partner_dist = (
            self._bfs(partner_pos, st["valid"], set()) if partner_pos else {}
        )

        ctx = {
            "state": state,
            "mdp": mdp,
            "st": st,
            "idx": idx,
            "me": me,
            "partner": partner,
            "my_pos": my_pos,
            "held": held,
            "partner_pos": partner_pos,
            "partner_held": partner_held,
            "dist_free": dist_free,
            "dist_block": dist_block,
            "partner_dist": partner_dist,
        }

        if self._start_pos is None:
            self._start_pos = my_pos

        ctx["pots"] = self._analyze_pots(ctx)
        ctx["orders"] = self._order_info(state)
        ctx["counter_objs"] = mdp.get_counter_objects_dict(state)
        ctx["shared_counters"] = self._shared_counters(ctx)

        action = self._retreat_step(ctx)
        if action is None:
            action = self._choose_action(ctx)
            if ctx.get("head_on"):
                # Partner stands on every path to my goal: resolve the
                # standoff (hold ground with right-of-way, or give way).
                action = self._standoff_action(ctx, action)
            else:
                self._standoff_wait = 0
            action = self._anti_stuck(ctx, action)
            action = self._yield_if_blocking(ctx, action)

        self._pos_history.append(my_pos)
        self._partner_history.append(partner_pos)
        self._move_intent_history.append(action in (NORTH, SOUTH, EAST, WEST))
        if self._debug:
            print(
                f"[dbg] t={state.timestep:3d} idx={idx} pos={my_pos} held={held} "
                f"act={action} target={ctx.get('nav_target')} head_on={ctx.get('head_on')} "
                f"retreat={self._retreat_target} wall={self._partner_wall}",
                file=sys.stderr,
            )
        return action

    # ------------------------------------------------------------------
    # World analysis
    # ------------------------------------------------------------------

    def _analyze_pots(self, ctx):
        """Return list of pot dicts with contents and status."""
        state, st = ctx["state"], ctx["st"]
        pots = []
        for pos in st["pots"]:
            info = {
                "pos": pos,
                "contents": Counter(),
                "idle": True,
                "cooking": False,
                "ready": False,
                "remaining": 0,
            }
            if state.has_object(pos):
                soup = state.get_object(pos)
                info["contents"] = Counter(soup.ingredients)
                info["idle"] = bool(soup.is_idle)
                info["cooking"] = bool(soup.is_cooking)
                info["ready"] = bool(soup.is_ready)
                if info["cooking"]:
                    try:
                        info["remaining"] = int(soup.cook_time_remaining)
                    except Exception:
                        info["remaining"] = 5
            pots.append(info)
        return pots

    def _order_info(self, state):
        """Return list of (Counter(ingredients), time, value), fastest first."""
        orders = []
        try:
            for r in state.all_orders:
                try:
                    t = int(r.time)
                except Exception:
                    t = 20
                try:
                    v = int(r.value)
                except Exception:
                    v = 20
                orders.append((Counter(r.ingredients), t, v))
        except Exception:
            pass
        if not orders:
            orders = [(Counter({"onion": 3}), 20, 20)]
        # Prefer soups that are fast to make (score counts #soups); tie-break value.
        orders.sort(key=lambda o: (o[1] + 3 * sum(o[0].values()), -o[2]))
        return orders

    def _shared_counters(self, ctx):
        """Counters that both players can interact with (handoff points)."""
        st = ctx["st"]
        shared = []
        for c in st["counters"]:
            mine = any(t in ctx["dist_free"] for t in self._adj(c, st))
            theirs = any(t in ctx["partner_dist"] for t in self._adj(c, st))
            if mine and theirs:
                shared.append(c)
        return shared

    # ------------------------------------------------------------------
    # Feature interaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adj(pos, st):
        return [p for d in ALL_DIRS if (p := _add(pos, d)) in st["valid"]]

    def _feature_cost(self, ctx, pos, dist_map=None):
        """Steps to stand next to `pos` (INF if unreachable)."""
        dist_map = ctx["dist_free"] if dist_map is None else dist_map
        best = INF
        for t in self._adj(pos, ctx["st"]):
            d = dist_map.get(t, INF)
            if d < best:
                best = d
        return best

    def _reachable(self, ctx, pos, who="me"):
        dist_map = ctx["dist_free"] if who == "me" else ctx["partner_dist"]
        return self._feature_cost(ctx, pos, dist_map) < INF

    def _nearest_feature(self, ctx, positions):
        best, best_cost = None, INF
        for pos in positions:
            c = self._feature_cost(ctx, pos)
            if c < best_cost:
                best, best_cost = pos, c
        return best, best_cost

    # ------------------------------------------------------------------
    # Task selection
    # ------------------------------------------------------------------

    def _choose_action(self, ctx) -> int:
        held = ctx["held"]
        if held == "soup":
            return self._do_deliver_soup(ctx)
        if held == "dish":
            return self._do_use_dish(ctx)
        if held in ("onion", "tomato"):
            return self._do_place_ingredient(ctx)
        return self._do_empty_hand(ctx)

    # ---------------- holding soup ----------------

    def _do_deliver_soup(self, ctx):
        # A soup whose recipe matches no order is worth nothing: dump it on a
        # counter to free our hands (and the pot pipeline) quickly.
        try:
            held_ings = Counter(ctx["me"].held_object.ingredients)
        except Exception:
            held_ings = None
        if held_ings is not None and self._soup_worthless(ctx, held_ings):
            drop = self._nearest_empty_counter(ctx, avoid_shared=True)
            if drop is not None:
                return self._go_interact(ctx, drop)

        serving = [s for s in ctx["st"]["serving"] if self._reachable(ctx, s)]
        if serving:
            target, _ = self._nearest_feature(ctx, serving)
            return self._go_interact(ctx, target)
        # Cannot serve myself: hand the soup over a shared counter.
        drop = self._best_handoff_counter(ctx, dest_positions=ctx["st"]["serving"])
        if drop is not None:
            return self._go_interact(ctx, drop)
        return self._stage_near(ctx, ctx["st"]["serving"] or ctx["st"]["pots"])

    # ---------------- holding dish ----------------

    def _do_use_dish(self, ctx):
        pots = ctx["pots"]
        ready = [p for p in pots if p["ready"] and self._reachable(ctx, p["pos"])]
        if ready:
            target = min(ready, key=lambda p: self._feature_cost(ctx, p["pos"]))
            return self._go_interact(ctx, target["pos"])

        cooking = [p for p in pots if p["cooking"] and self._reachable(ctx, p["pos"])]
        if cooking:
            # Claim the pot the partner is NOT about to serve.
            if ctx["partner_held"] == "dish" and len(cooking) > 1:
                partner_claim = min(
                    cooking,
                    key=lambda p: self._feature_cost(ctx, p["pos"], ctx["partner_dist"]),
                )
                mine = [p for p in cooking if p is not partner_claim]
            else:
                mine = cooking
            target = min(mine, key=lambda p: self._feature_cost(ctx, p["pos"]))
            # Wait facing the pot, spamming interact (no-op until soup is ready,
            # then it grabs the soup on the exact tick it finishes).
            return self._go_interact(ctx, target["pos"])

        full_idle = [
            p
            for p in pots
            if p["idle"] and sum(p["contents"].values()) > 0
            and self._pot_cookable(ctx, p)
            and self._reachable(ctx, p["pos"])
        ]
        if full_idle:
            # Someone must press "cook" with an empty hand. If the partner is not
            # going to, drop the dish and do it ourselves.
            target = min(full_idle, key=lambda p: self._feature_cost(ctx, p["pos"]))
            partner_close = (
                ctx["partner_held"] is None
                and self._feature_cost(ctx, target["pos"], ctx["partner_dist"]) <= 2
            )
            if partner_close:
                return self._stage_near(ctx, [target["pos"]])
            drop = self._nearest_empty_counter(ctx, avoid_shared=True)
            if drop is not None:
                return self._go_interact(ctx, drop)
            return self._stage_near(ctx, [target["pos"]])

        # No soup in sight for this dish.
        if not any(self._reachable(ctx, p) for p in ctx["st"]["pots"]):
            # Supply mode: pass the dish to the partner's side.
            drop = self._best_handoff_counter(ctx, dest_positions=ctx["st"]["pots"])
            if drop is not None:
                return self._go_interact(ctx, drop)
        # If ingredients are still needed and I am holding a useless dish, park it.
        if self._pots_accepting(ctx):
            drop = self._nearest_empty_counter(ctx, avoid_shared=True)
            if drop is not None:
                return self._go_interact(ctx, drop)
        return self._stage_near(ctx, ctx["st"]["pots"])

    # ---------------- holding an ingredient ----------------

    def _do_place_ingredient(self, ctx):
        ing = ctx["held"]
        candidates = []
        for p in ctx["pots"]:
            if not p["idle"] or not self._reachable(ctx, p["pos"]):
                continue
            if self._fits_some_order(ctx, p["contents"] + Counter({ing: 1})):
                candidates.append(p)
        if candidates:
            # Prefer the pot closest to completing an order, then distance.
            def key(p):
                missing = self._missing_for_best_order(ctx, p["contents"])
                return (missing, self._feature_cost(ctx, p["pos"]))

            target = min(candidates, key=key)
            return self._go_interact(ctx, target["pos"])

        if not any(self._reachable(ctx, p) for p in ctx["st"]["pots"]):
            # Supply mode: hand the ingredient to the partner over a counter.
            drop = self._best_handoff_counter(ctx, dest_positions=ctx["st"]["pots"])
            if drop is not None:
                return self._go_interact(ctx, drop)
            return self._stage_near(ctx, ctx["shared_counters"])

        # Ingredient does not fit any pot right now.
        if self._ingredient_useful_later(ctx, ing):
            return self._stage_near(ctx, [p["pos"] for p in ctx["pots"]])
        drop = self._nearest_empty_counter(ctx, avoid_shared=True)
        if drop is not None:
            return self._go_interact(ctx, drop)
        return STAY

    # ---------------- empty hand ----------------

    def _do_empty_hand(self, ctx):
        pots = ctx["pots"]
        st = ctx["st"]

        # 1. Start cooking a pot whose contents match an order. Also flush
        # "poisoned" pots (contents that can never complete any order, e.g. a
        # partner dropped an onion into a tomato-only recipe): cooking them
        # immediately is the fastest way to free the pot.
        startable = [
            p
            for p in pots
            if p["idle"] and sum(p["contents"].values()) > 0
            and (self._pot_cookable(ctx, p) or not self._fits_some_order(ctx, p["contents"]))
            and self._reachable(ctx, p["pos"])
        ]
        if startable:
            target = min(startable, key=lambda p: self._feature_cost(ctx, p["pos"]))
            partner_closer = (
                ctx["partner_held"] is None
                and self._feature_cost(ctx, target["pos"], ctx["partner_dist"]) + 1
                < self._feature_cost(ctx, target["pos"])
            )
            if not partner_closer:
                return self._go_interact(ctx, target["pos"])

        # 2. A finished soup sitting on a counter: pick it up and deliver
        # (only if its recipe is actually worth something).
        soup_counters = []
        for c in ctx["counter_objs"].get("soup", []):
            if not self._reachable(ctx, c):
                continue
            try:
                soup = ctx["state"].get_object(c)
                if not soup.is_ready or self._soup_worthless(ctx, Counter(soup.ingredients)):
                    continue
            except Exception:
                pass
            soup_counters.append(c)
        if soup_counters:
            target, _ = self._nearest_feature(ctx, soup_counters)
            return self._go_interact(ctx, target)

        # 3. Dish logistics: soups ready/cooking/about-to-cook need dishes.
        # Also predictive: a pot missing exactly the ingredient the partner is
        # carrying will need a dish very soon (and fetching one keeps us out of
        # the partner's way instead of camping next to the pot).
        partner_ing = ctx["partner_held"] if ctx["partner_held"] in ("onion", "tomato") else None
        service_pots = [
            p
            for p in pots
            if p["ready"]
            or p["cooking"]
            or (p["idle"] and self._pot_cookable(ctx, p))
            or (
                p["idle"]
                and partner_ing is not None
                and self._missing_for_best_order(ctx, p["contents"]) == 1
                and self._fits_some_order(ctx, p["contents"] + Counter({partner_ing: 1}))
            )
        ]
        my_service = [p for p in service_pots if self._reachable(ctx, p["pos"])]
        dishes_in_flight = int(ctx["partner_held"] == "dish")
        if len(my_service) > dishes_in_flight:
            src = self._nearest_item_source(ctx, "dish")
            if src is not None:
                # Just-in-time: skip only if the soup is far from done AND the
                # pot still needs me for ingredients elsewhere.
                return self._go_interact(ctx, src)

        # 4. Fetch an ingredient for a pot that accepts one.
        accepting = self._pots_accepting(ctx)
        if accepting:
            needed = self._needed_ingredients(ctx, accepting)
            # Discount what the partner is already carrying.
            if ctx["partner_held"] in needed:
                needed[ctx["partner_held"]] -= 1
            for ing, cnt in needed.most_common():
                if cnt <= 0:
                    continue
                src = self._nearest_item_source(ctx, ing)
                if src is not None:
                    return self._go_interact(ctx, src)

        # 5. Supply mode: I cannot reach any pot -> feed the partner's side.
        if not any(self._reachable(ctx, p) for p in st["pots"]):
            return self._do_supply_mode(ctx)

        # 6. Receive mode / idle: stage near the most relevant feature.
        if any(p["cooking"] or p["ready"] for p in pots):
            src = self._nearest_item_source(ctx, "dish")
            if src is not None:
                return self._go_interact(ctx, src)
        stage_targets = [p["pos"] for p in pots] or st["counters"]
        return self._stage_near(ctx, stage_targets)

    # ---------------- supply mode (disconnected layouts) ----------------

    def _do_supply_mode(self, ctx):
        """I cannot reach any pot: ferry items the partner's side lacks."""
        st = ctx["st"]
        shared = ctx["shared_counters"]
        if not shared:
            return self._stage_near(ctx, st["counters"])

        on_shared = Counter()
        for name, positions in ctx["counter_objs"].items():
            on_shared[name] += sum(1 for p in positions if p in set(shared))

        partner_pots = [p for p in ctx["pots"] if self._reachable(ctx, p["pos"], "partner")]

        # What do the partner's pots need?
        needed = self._needed_ingredients(ctx, [p for p in partner_pots if p["idle"]])
        for ing in list(needed):
            needed[ing] -= on_shared.get(ing, 0)
            if ctx["partner_held"] == ing:
                needed[ing] -= 1

        # Dishes for soups in progress.
        n_service = sum(
            1 for p in partner_pots if p["ready"] or p["cooking"] or (p["idle"] and self._pot_cookable(ctx, p))
        )
        partner_can_get_dish = any(
            self._reachable(ctx, d, "partner") for d in st["dishes"]
        )
        dish_need = 0
        if not partner_can_get_dish:
            dish_need = n_service - on_shared.get("dish", 0) - int(ctx["partner_held"] == "dish")

        # Fetch the most urgent item I can obtain.
        wishlist = []
        if dish_need > 0:
            wishlist.append(("dish", 2 * dish_need + 1))
        for ing, cnt in needed.items():
            if cnt > 0:
                wishlist.append((ing, cnt))
        wishlist.sort(key=lambda kv: -kv[1])
        for item, _ in wishlist:
            src = self._nearest_item_source(ctx, item, exclude_counters=set(shared))
            if src is not None:
                return self._go_interact(ctx, src)

        # Nothing to supply right now; also relay finished soups toward serving
        # if the partner cannot serve.
        return self._stage_near(ctx, shared)

    # ------------------------------------------------------------------
    # Order / recipe reasoning
    # ------------------------------------------------------------------

    def _soup_worthless(self, ctx, contents: Counter) -> bool:
        return not any(contents == order for order, _, _ in ctx["orders"])

    def _fits_some_order(self, ctx, contents: Counter) -> bool:
        if sum(contents.values()) > ctx["st"]["max_ingredients"]:
            return False
        for order, _, _ in ctx["orders"]:
            if all(contents[k] <= order[k] for k in contents):
                return True
        return False

    def _pot_cookable(self, ctx, pot) -> bool:
        """Contents exactly match an order, or pot is full (must be flushed)."""
        contents = pot["contents"]
        n = sum(contents.values())
        if n == 0:
            return False
        for order, _, _ in ctx["orders"]:
            if contents == order:
                return True
        return n >= ctx["st"]["max_ingredients"]

    def _missing_for_best_order(self, ctx, contents: Counter) -> int:
        best = INF
        for order, _, _ in ctx["orders"]:
            if all(contents[k] <= order[k] for k in contents):
                best = min(best, sum(order.values()) - sum(contents.values()))
        return best if best < INF else 99

    def _pots_accepting(self, ctx):
        return [
            p
            for p in ctx["pots"]
            if p["idle"]
            and self._reachable(ctx, p["pos"])
            and not self._pot_cookable(ctx, p)
            and self._fits_some_order(ctx, p["contents"])
        ]

    def _needed_ingredients(self, ctx, pots) -> Counter:
        """Aggregate missing ingredients (vs the fastest feasible order) per pot."""
        needed = Counter()
        for p in pots:
            contents = p["contents"]
            for order, _, _ in ctx["orders"]:
                if all(contents[k] <= order[k] for k in contents):
                    missing = order - contents
                    # Only count orders whose missing ingredients are obtainable.
                    if all(self._ingredient_obtainable(ctx, ing) for ing in missing):
                        needed += missing
                        break
        return needed

    def _ingredient_obtainable(self, ctx, ing) -> bool:
        st = ctx["st"]
        disp = st["onions"] if ing == "onion" else st["tomatoes"] if ing == "tomato" else []
        if any(self._reachable(ctx, d) or self._reachable(ctx, d, "partner") for d in disp):
            return True
        return bool(ctx["counter_objs"].get(ing))

    def _ingredient_useful_later(self, ctx, ing) -> bool:
        return any(order[ing] > 0 for order, _, _ in ctx["orders"])

    # ------------------------------------------------------------------
    # Sources and handoff counters
    # ------------------------------------------------------------------

    def _nearest_item_source(self, ctx, item, exclude_counters=None):
        """Nearest reachable source of `item`: counter object first, then dispenser."""
        st = ctx["st"]
        exclude = exclude_counters or set()
        options = [
            c for c in ctx["counter_objs"].get(item, []) if c not in exclude and self._reachable(ctx, c)
        ]
        if item == "onion":
            options += [d for d in st["onions"] if self._reachable(ctx, d)]
        elif item == "tomato":
            options += [d for d in st["tomatoes"] if self._reachable(ctx, d)]
        elif item == "dish":
            options += [d for d in st["dishes"] if self._reachable(ctx, d)]
        if not options:
            return None
        target, _ = self._nearest_feature(ctx, options)
        return target

    def _nearest_empty_counter(self, ctx, avoid_shared=False):
        state = ctx["state"]
        shared = set(ctx["shared_counters"]) if avoid_shared else set()
        options = [
            c
            for c in ctx["st"]["counters"]
            if not state.has_object(c) and self._reachable(ctx, c) and c not in shared
        ]
        if not options and avoid_shared:
            return self._nearest_empty_counter(ctx, avoid_shared=False)
        if not options:
            return None
        target, _ = self._nearest_feature(ctx, options)
        return target

    def _best_handoff_counter(self, ctx, dest_positions):
        """Empty shared counter minimising my cost + partner's onward cost."""
        state = ctx["state"]
        options = [c for c in ctx["shared_counters"] if not state.has_object(c) and self._reachable(ctx, c)]
        if not options:
            return None

        def key(c):
            mine = self._feature_cost(ctx, c)
            onward = INF
            for dest in dest_positions:
                # partner's distance from counter to destination ~ partner_dist proxy
                d = self._feature_cost(ctx, dest, ctx["partner_dist"])
                onward = min(onward, d)
            partner_pickup = self._feature_cost(ctx, c, ctx["partner_dist"])
            if onward is INF:
                onward = 0
            if partner_pickup is INF:
                partner_pickup = 50
            return mine + 0.5 * partner_pickup + 0.25 * onward

        return min(options, key=key)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _bfs(self, start, valid, blocked):
        """Distance map over walkable tiles; `blocked` tiles are impassable."""
        dist = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            d = dist[pos]
            for dr in ALL_DIRS:
                nxt = _add(pos, dr)
                if nxt not in valid or nxt in blocked or nxt in dist:
                    continue
                dist[nxt] = d + 1
                q.append(nxt)
        return dist

    def _path_next_step(self, ctx, goals, blocked):
        """First step of a shortest path from my position to any goal tile."""
        st = ctx["st"]
        start = ctx["my_pos"]
        if start in goals:
            return start
        prev = {start: None}
        q = deque([start])
        found = None
        while q:
            pos = q.popleft()
            if pos in goals:
                found = pos
                break
            for dr in ALL_DIRS:
                nxt = _add(pos, dr)
                if nxt not in st["valid"] or nxt in blocked or nxt in prev:
                    continue
                prev[nxt] = pos
                q.append(nxt)
        if found is None:
            return None
        node = found
        while prev[node] is not None and prev[node] != start:
            node = prev[node]
        return node if prev[node] == start else start

    def _go_interact(self, ctx, feature_pos) -> int:
        """Move to, face, and interact with the feature at `feature_pos`."""
        my_pos = ctx["my_pos"]
        st = ctx["st"]
        ctx["nav_target"] = feature_pos
        delta = (feature_pos[0] - my_pos[0], feature_pos[1] - my_pos[1])
        if delta in DIR_TO_ACTION:  # adjacent
            if ctx["me"].orientation == delta:
                self._last_feature = feature_pos
                return INTERACT
            return DIR_TO_ACTION[delta]

        goals = set(self._adj(feature_pos, st))
        blocked = {ctx["partner_pos"]} if ctx["partner_pos"] else set()
        nxt = self._path_next_step(ctx, goals, blocked)
        if nxt is None:
            if self._partner_wall:
                return STAY  # feature only reachable through an immobile partner
            # Partner blocks the only path: try ignoring it and wait if needed.
            nxt = self._path_next_step(ctx, goals, set())
            if nxt is None:
                return STAY
            # Head-on standoff in a corridor: flag it so _decide can retreat.
            ctx["head_on"] = True
            if nxt == ctx["partner_pos"]:
                return self._sidestep_or_wait(ctx)
        if nxt == my_pos:
            return STAY
        step = (nxt[0] - my_pos[0], nxt[1] - my_pos[1])
        return DIR_TO_ACTION.get(step, STAY)

    def _stage_near(self, ctx, positions) -> int:
        """Idle behaviour: park adjacent to the nearest of `positions`."""
        reachable = [p for p in positions if self._reachable(ctx, p)]
        if not reachable:
            return STAY
        target, cost = self._nearest_feature(ctx, reachable)
        if cost == 0:
            # Already adjacent: face it so an interact is instant later.
            delta = (target[0] - ctx["my_pos"][0], target[1] - ctx["my_pos"][1])
            if delta in DIR_TO_ACTION and ctx["me"].orientation != delta:
                return DIR_TO_ACTION[delta]
            return STAY
        # Move toward it but stop one tile early to avoid crowding.
        action = self._go_interact(ctx, target)
        return action if action != INTERACT else STAY

    def _standoff_action(self, ctx, fallback) -> int:
        """Resolve a head-on corridor standoff.

        Deterministic right-of-way avoids two adaptive agents mirroring each
        other forever: a loaded agent has priority over an empty-handed one;
        on ties, player 0 has priority. The yielding agent commits to a
        retreat pocket; the priority agent holds ground for a few steps (and
        yields anyway if the partner turns out not to move at all).
        """
        my_loaded = ctx["held"] is not None
        p_loaded = ctx["partner_held"] is not None
        i_have_priority = (my_loaded and not p_loaded) or (
            my_loaded == p_loaded and ctx["idx"] == 0
        )
        if i_have_priority:
            self._standoff_wait += 1
            if self._standoff_wait < 5:
                return fallback  # hold ground; a well-behaved partner yields
        self._standoff_wait = 0
        return self._begin_retreat(ctx, fallback)

    def _begin_retreat(self, ctx, fallback) -> int:
        """Commit to the nearest give-way pocket.

        A pocket is a tile from which the partner could still travel back to
        the area I came from (`anchor`): standing there, I no longer plug the
        corridor, the partner can pass, and my own path reopens afterwards.
        """
        st = ctx["st"]
        partner_pos = ctx["partner_pos"]
        if partner_pos is None:
            return fallback

        anchor = self._last_feature or self._start_pos or ctx["my_pos"]
        anchor_tiles = {anchor} if anchor in st["valid"] else set(self._adj(anchor, st))
        if not anchor_tiles:
            return fallback

        # Candidate pockets, nearest first (partner treated as a wall).
        candidates = sorted(ctx["dist_block"], key=ctx["dist_block"].get)[:60]
        for t in candidates:
            if t == partner_pos:
                continue
            passable = self._bfs(partner_pos, st["valid"], {t})
            if any(a in passable for a in anchor_tiles if a != t):
                self._retreat_target = t
                self._standoff_partner_pos = partner_pos
                self._pocket_wait = 0
                step_action = self._retreat_step(ctx)
                return STAY if step_action is None else step_action
        return fallback

    def _retreat_step(self, ctx):
        """Advance a committed retreat; return None when no retreat is active."""
        if self._retreat_target is None:
            return None
        partner_pos = ctx["partner_pos"]
        # Partner clearly moved away from the standoff: resume normal play.
        if (
            partner_pos is not None
            and self._standoff_partner_pos is not None
            and partner_pos != self._standoff_partner_pos
            and abs(partner_pos[0] - ctx["my_pos"][0]) + abs(partner_pos[1] - ctx["my_pos"][1]) > 1
        ):
            self._retreat_target = None
            return None

        if ctx["my_pos"] == self._retreat_target:
            # Sit in the pocket and let the partner pass. If it has a clear
            # way through and still refuses to move, it is not waiting for
            # us: replan treating it as a wall.
            self._pocket_wait += 1
            partner_static = bool(
                self._partner_history and self._partner_history[-1] == partner_pos
            )
            if self._pocket_wait >= 8:
                self._retreat_target = None
                if partner_static:
                    self._partner_wall = True
                return None
            return STAY

        blocked = {partner_pos} if partner_pos else set()
        nxt = self._path_next_step(ctx, {self._retreat_target}, blocked)
        if nxt is None or nxt == ctx["my_pos"]:
            self._retreat_target = None
            return None
        step = (nxt[0] - ctx["my_pos"][0], nxt[1] - ctx["my_pos"][1])
        return DIR_TO_ACTION.get(step, STAY)

    def _sidestep_or_wait(self, ctx) -> int:
        if self.rng.random() < 0.5:
            return STAY
        options = []
        for dr in ALL_DIRS:
            nxt = _add(ctx["my_pos"], dr)
            if nxt in ctx["st"]["valid"] and nxt != ctx["partner_pos"]:
                options.append(DIR_TO_ACTION[dr])
        if not options:
            return STAY
        return int(self.rng.choice(options))

    def _yield_if_blocking(self, ctx, action) -> int:
        """Step aside when idle while an adjacent partner has been stuck.

        Handles the classic livelock where we camp on the only interaction
        tile of a pot while the partner needs to stand exactly there.
        """
        if action != STAY or ctx["partner_pos"] is None:
            return action
        p = ctx["partner_pos"]
        adjacent = abs(p[0] - ctx["my_pos"][0]) + abs(p[1] - ctx["my_pos"][1]) == 1
        if not adjacent or len(self._partner_history) < 2:
            return action
        partner_stuck = (
            self._partner_history[-1] == self._partner_history[-2] == p
        )
        if not partner_stuck:
            return action
        options = []
        for dr in ALL_DIRS:
            nxt = _add(ctx["my_pos"], dr)
            if nxt in ctx["st"]["valid"] and nxt != p:
                options.append(DIR_TO_ACTION[dr])
        if not options:
            return action
        return int(self.rng.choice(options))

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    def _anti_stuck(self, ctx, action) -> int:
        """If we tried to move for 3+ steps without changing position, side-step."""
        h = self._pos_history
        if action in (NORTH, SOUTH, EAST, WEST) and len(h) >= 3:
            recent_static = h[-1] == h[-2] == h[-3] == ctx["my_pos"]
            tried_moving = sum(list(self._move_intent_history)[-3:]) >= 2
            if recent_static and tried_moving:
                options = []
                intended = _add(ctx["my_pos"], next(d for d, a in DIR_TO_ACTION.items() if a == action))
                for dr in ALL_DIRS:
                    nxt = _add(ctx["my_pos"], dr)
                    if (
                        nxt in ctx["st"]["valid"]
                        and nxt != ctx["partner_pos"]
                        and nxt != intended
                    ):
                        options.append(DIR_TO_ACTION[dr])
                if options and self.rng.random() < 0.8:
                    return int(self.rng.choice(options))
        return action
