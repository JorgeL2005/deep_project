"""Main execution loop for the Overcooked competition base."""

from __future__ import annotations


import random
from typing import Any

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.actions import Action

from src.constants import overcooked_action_to_index
from src.demonstrations import DemonstrationRecorder
from src.environment import build_env
from src.dataset_metadata import build_dataset_metadata
from src.logging_utils import CompetitionLogger, StepRecord
from src.observations import ObservationBuilder
from src.policy_loader import build_two_policies
from src.rendering import Renderer


def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _action_to_name(action) -> str:
    return Action.ACTION_TO_CHAR.get(action, str(action))


def _uses_human_keyboard(config: dict[str, Any]) -> bool:
    policies = config.get("policies", {}) or {}
    for policy_cfg in policies.values():
        if str(policy_cfg.get("type", "builtin")).lower() == "builtin" and str(policy_cfg.get("name", "")).lower() == "human_keyboard":
            return True
    return False


def _prepare_rendering_config(config: dict[str, Any]) -> dict[str, Any]:
    rendering_config = dict(config.get("rendering", {}) or {})
    if _uses_human_keyboard(config) and rendering_config.get("mode", "none") != "window":
        rendering_config["mode"] = "window"
        rendering_config.setdefault("fps", 5)
        print("Human keyboard policy detected: forcing rendering.mode='window'.")
    return rendering_config


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run one configured evaluation/debug session and return aggregate results."""
    seed = config.get("seed")
    set_global_seed(None if seed is None else int(seed))

    env = build_env(config["environment"])
    obs_builder = ObservationBuilder(env, config.get("observation", {}))
    logger = CompetitionLogger(config.get("logging", {}))
    renderer = Renderer(_prepare_rendering_config(config))
    demo_recorder = DemonstrationRecorder(config.get("data_collection", {}))

    num_episodes = int(config.get("execution", {}).get("num_episodes", 1))
    episode_seeds = config.get("execution", {}).get("episode_seeds")
    if episode_seeds is None:
        episode_seeds = list(range(num_episodes))
    if len(episode_seeds) < num_episodes:
        raise ValueError("execution.episode_seeds must have at least num_episodes elements")

    swap_agent_positions = bool(config.get("execution", {}).get("swap_agent_positions", False))
    role_swaps = [False, True] if swap_agent_positions else [False]

    all_episode_returns: list[float] = []
    episode_id = 0
    stop_requested = False

    for base_episode_idx in range(num_episodes):
        if stop_requested:
            break
        for role_swap in role_swaps:
            if stop_requested:
                break
            ep_seed = int(episode_seeds[base_episode_idx])
            set_global_seed(ep_seed)

            # Rebuild policies per rollout to avoid hidden state leakage across role swaps.
            agent0, agent1 = build_two_policies(config, env, obs_builder, seed=ep_seed)
            if role_swap:
                agent0, agent1 = agent1, agent0

            agent_pair = AgentPair(agent0, agent1)
            env.reset(regen_mdp=False)
            agent_pair.reset()
            agent_pair.set_mdp(env.mdp)
            renderer.reset()
            renderer.maybe_render(env, timestep=0)
            if renderer.closed_by_user:
                stop_requested = True
                break

            done = False
            episode_return = 0.0
            trajectory = []
            info: dict[str, Any] = {}

            while not done:
                state = env.state
                joint_action_and_infos = agent_pair.joint_action(state)
                joint_action, joint_infos = zip(*joint_action_and_infos)
                joint_action_indices = [overcooked_action_to_index(action) for action in joint_action]

                next_state, reward, done, info = env.step(joint_action, joint_infos)
                episode_return += float(reward)

                demo_recorder.record_step(
                    obs_builder=obs_builder,
                    episode_id=episode_id,
                    timestep=state.timestep,
                    layout_name=env.mdp.layout_name,
                    role_swap=role_swap,
                    episode_seed=ep_seed,
                    state=state,
                    next_state=next_state,
                    joint_action_indices=joint_action_indices,
                    reward=float(reward),
                    done=bool(done),
                    info=info,
                )

                step_record = StepRecord(
                    episode_id=episode_id,
                    timestep=state.timestep,
                    layout_name=env.mdp.layout_name,
                    role_swap=role_swap,
                    action_0=_action_to_name(joint_action[0]),
                    action_1=_action_to_name(joint_action[1]),
                    reward=float(reward),
                    done=bool(done),
                    info=info,
                )
                logger.log_step(step_record)

                trajectory.append((state, joint_action, reward, done, info))
                renderer.maybe_render(
                    env,
                    timestep=next_state.timestep,
                    joint_action=tuple(_action_to_name(a) for a in joint_action),
                    reward=float(reward),
                )
                if renderer.closed_by_user:
                    done = True
                    stop_requested = True

            episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
            summary = {
                "episode_id": episode_id,
                "layout_name": env.mdp.layout_name,
                "seed": ep_seed,
                "role_swap": role_swap,
                "return_sparse": episode_return,
                "ep_sparse_r": episode_info.get("ep_sparse_r", episode_return),
                "ep_shaped_r": episode_info.get("ep_shaped_r"),
                "ep_length": episode_info.get("ep_length", env.state.timestep),
            }
            logger.log_episode(summary)
            demo_recorder.record_episode(summary)
            logger.save_trajectory(episode_id, trajectory)
            all_episode_returns.append(episode_return)
            episode_id += 1
            if stop_requested:
                break

    logger.flush()
    demo_metadata = build_dataset_metadata(
        config=config,
        env=env,
        obs_builder=obs_builder,
        num_episodes=num_episodes,
        episode_seeds=[int(s) for s in episode_seeds],
        role_swaps=role_swaps,
    )
    demo_recorder.flush(metadata=demo_metadata)
    renderer.close()

    aggregate = {
        "num_rollouts": len(all_episode_returns),
        "mean_return_sparse": float(np.mean(all_episode_returns)) if all_episode_returns else 0.0,
        "std_return_sparse": float(np.std(all_episode_returns)) if all_episode_returns else 0.0,
        "returns_sparse": all_episode_returns,
        "output_dir": str(logger.output_dir),
    }
    return aggregate
