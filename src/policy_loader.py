"""Policy loading utilities."""

from __future__ import annotations


import importlib.util
import sys
from pathlib import Path
from typing import Any

from overcooked_ai_py.agents.agent import GreedyHumanModel, RandomAgent, StayAgent

from policies.basic_policies import GreedyFullTaskPolicy, RandomMotionPolicy, StayPolicy
from policies.human_keyboard_policy import HumanKeyboardPolicy

from src.observations import ObservationBuilder
from src.policy_wrappers import StudentAgentAdapter, wrap_agent


class PolicyLoadError(ValueError):
    """Raised when a policy cannot be loaded."""


def import_class_from_file(path: str | Path, class_name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    module_name = f"student_policy_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PolicyLoadError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise PolicyLoadError(f"Class '{class_name}' not found in {path}") from exc


def build_builtin_agent(name: str, env, policy_config: dict[str, Any] | None = None):
    policy_config = policy_config or {}
    key = str(name).strip().lower()

    # Local readable baselines provided with this starter code.
    if key == "stay":
        return StayPolicy()
    if key == "random_motion":
        return RandomMotionPolicy(seed=policy_config.get("seed"))
    if key == "greedy_full_task":
        return GreedyFullTaskPolicy(
            ingredient=policy_config.get("ingredient", "onion"),
            avoid_teammate=policy_config.get("avoid_teammate", True),
            seed=policy_config.get("seed"),
        )
    if key == "human_keyboard":
        return HumanKeyboardPolicy(
            keymap=policy_config.get("keymap"),
            priority=policy_config.get("priority"),
        )

    # Official Overcooked-AI baselines kept for convenience.
    if key == "random":
        return RandomAgent(all_actions=True)
    if key == "greedy_human_model":
        return GreedyHumanModel(env.mlam)

    raise PolicyLoadError(
        f"Unknown builtin policy '{name}'. Valid builtins: "
        "stay, random_motion, random, greedy_full_task, human_keyboard, greedy_human_model"
    )


def build_policy(policy_config: dict[str, Any], env, obs_builder: ObservationBuilder, seed: int | None = None):
    """Build and wrap one policy from YAML configuration."""
    policy_type = str(policy_config.get("type", "builtin"))
    name = str(policy_config.get("name", policy_type))

    if policy_type == "builtin":
        if "name" not in policy_config:
            raise PolicyLoadError("Builtin policy requires field 'name'")
        base_agent = build_builtin_agent(policy_config["name"], env, policy_config=policy_config)
    elif policy_type == "python_class":
        if "path" not in policy_config:
            raise PolicyLoadError("python_class policy requires field 'path'")
        class_name = policy_config.get("class_name", "StudentAgent")
        cls = import_class_from_file(policy_config["path"], class_name)
        student_config = policy_config.get("config", {}) or {}
        student_agent = cls(student_config)
        base_agent = StudentAgentAdapter(student_agent, obs_builder, name=name)
    else:
        raise PolicyLoadError(f"Unknown policy type '{policy_type}'. Use builtin or python_class")

    return wrap_agent(base_agent, policy_config, seed=seed)


def build_two_policies(config: dict[str, Any], env, obs_builder: ObservationBuilder, seed: int | None = None):
    policies_config = config.get("policies")
    if not isinstance(policies_config, dict):
        raise PolicyLoadError("Config must contain a 'policies' mapping")
    if "agent_0" not in policies_config or "agent_1" not in policies_config:
        raise PolicyLoadError("Config policies must contain 'agent_0' and 'agent_1'")

    agent0 = build_policy(policies_config["agent_0"], env, obs_builder, seed=None if seed is None else seed + 1000)
    agent1 = build_policy(policies_config["agent_1"], env, obs_builder, seed=None if seed is None else seed + 2000)
    return agent0, agent1
