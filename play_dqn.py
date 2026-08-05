"""Play snake with a trained DQN agent."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    from .dqn_agent import DQNAgent, flatten_observation
    from .env import (
        Action,
        GameConfig,
        RelativeAction,
        SnakeGameEnv,
        relative_to_absolute,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dqn_agent import DQNAgent, flatten_observation
    from env import (
        Action,
        GameConfig,
        RelativeAction,
        SnakeGameEnv,
        relative_to_absolute,
    )


DEFAULT_MODEL_PATH = Path("models/dqn_snake_v3_best.pt")
LEGACY_MODEL_PATH = Path("models/dqn_snake.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for a trained DQN snake agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the trained model checkpoint (.pt). Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    parser.add_argument(
        "--delay", type=float, default=0.05, help="Delay between steps (seconds)"
    )
    parser.add_argument(
        "--cell-size", type=int, default=25, help="GUI cell size in pixels"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Run in console (ASCII) mode instead of GUI",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render ASCII board to the console (console mode)",
    )
    parser.add_argument(
        "--disable-safety-check",
        action="store_true",
        help="Disable the safety fallback that avoids immediate collisions during inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for the environment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for inference (cpu/cuda)",
    )
    return parser.parse_args()


def build_env_from_metadata(agent: DQNAgent, seed: int | None) -> SnakeGameEnv:
    if agent.game_config is not None:
        base_config = GameConfig(**asdict(agent.game_config))
    else:
        base_config = GameConfig()
    base_config.seed = seed if seed is not None else None
    return SnakeGameEnv(base_config)


def action_mask(agent: DQNAgent, env: SnakeGameEnv) -> list[bool]:
    if agent.action_dim == len(RelativeAction):
        return [True] * agent.action_dim
    if agent.action_dim == len(Action):
        # Legacy four-action checkpoints learned with the old reverse-action alias
        # (the environment executes it as straight). Preserve that behavior during
        # inference; corrected training masks it in train_dqn.action_mask.
        return [True] * agent.action_dim
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def absolute_action(agent: DQNAgent, env: SnakeGameEnv, action_index: int) -> Action:
    if agent.action_dim == len(RelativeAction):
        return relative_to_absolute(env.direction, RelativeAction(action_index))
    if agent.action_dim == len(Action):
        return Action(action_index)
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def select_safe_action(
    agent: DQNAgent, env: SnakeGameEnv, state: torch.Tensor, safety_enabled: bool
) -> int:
    legal_mask = action_mask(agent, env)
    action = agent.select_action(state, epsilon_override=0.0, action_mask=legal_mask)
    if not safety_enabled:
        return action
    if env.is_safe_action(absolute_action(agent, env, action)):
        return action
    with torch.no_grad():
        q_values = agent.policy_net(state.unsqueeze(0))
    for candidate in torch.argsort(q_values[0], descending=True).tolist():
        if legal_mask[candidate] and env.is_safe_action(
            absolute_action(agent, env, candidate)
        ):
            return candidate
    return action


def step_agent_action(
    agent: DQNAgent, env: SnakeGameEnv, action_index: int
) -> tuple[dict[str, object], float, bool, dict[str, object]]:
    if agent.action_dim == len(RelativeAction):
        return env.step_relative(RelativeAction(action_index))
    return env.step(Action(action_index))


def run_episode(
    agent: DQNAgent, env: SnakeGameEnv, delay: float, render: bool, safety_enabled: bool
) -> dict:
    env.reset()
    state = flatten_observation(env, agent.device, expected_channels=agent.obs_shape[0])
    total_reward = 0.0
    while True:
        action_idx = select_safe_action(agent, env, state, safety_enabled)
        obs, reward, done, info = step_agent_action(agent, env, action_idx)
        total_reward += reward
        state = flatten_observation(
            env, agent.device, expected_channels=agent.obs_shape[0]
        )
        if render:
            board = env.render(to_string=True)
            print(board)
            print(
                f"score={obs['score']} steps={obs['steps']} reward={reward:+.3f} event={info.get('event', 'continue')}"
            )
            print("-" * (2 * env.config.width))
            time.sleep(delay)
        if done:
            break
    return {"reward": total_reward, "score": env.score, "steps": env.steps}


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if (
        model_path == DEFAULT_MODEL_PATH
        and not model_path.exists()
        and LEGACY_MODEL_PATH.exists()
    ):
        model_path = LEGACY_MODEL_PATH
        print(f"v3 model not found; falling back to legacy checkpoint: {model_path}")
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    agent = DQNAgent.load(str(model_path), device=args.device)
    agent.policy_net.eval()
    agent.target_net.eval()
    env = build_env_from_metadata(agent, args.seed)

    print(f"Loaded model from {model_path.resolve()}")

    if args.console:
        print(
            f"Playing {args.episodes} episode(s) with grid {env.config.width}x{env.config.height} (console mode)"
        )
        stats: List[dict] = []
        for idx in range(1, args.episodes + 1):
            print(f"Episode {idx}")
            result = run_episode(
                agent, env, args.delay, args.render, not args.disable_safety_check
            )
            stats.append(result)
            print(
                f" -> reward={result['reward']:.3f} score={result['score']} steps={result['steps']}"
            )
        if stats:
            avg_reward = np.mean([s["reward"] for s in stats])
            avg_score = np.mean([s["score"] for s in stats])
            avg_steps = np.mean([s["steps"] for s in stats])
            print(
                f"Averages: reward={avg_reward:.3f} score={avg_score:.2f} steps={avg_steps:.1f}"
            )
        return

    try:
        from .gui import SnakeGameGUI
    except ImportError:
        from gui import SnakeGameGUI

    agent.epsilon = 0.0
    results: List[dict] = []
    gui: Optional[SnakeGameGUI] = None
    safety_enabled = not args.disable_safety_check

    def controller(current_env: SnakeGameEnv) -> Action:
        state_tensor = flatten_observation(
            current_env, agent.device, expected_channels=agent.obs_shape[0]
        )
        action_idx = select_safe_action(
            agent, current_env, state_tensor, safety_enabled
        )
        return absolute_action(agent, current_env, action_idx)

    def on_episode_end(summary: dict) -> None:
        results.append(summary)
        idx = len(results)
        print(
            f"Episode {idx} -> reward={summary['reward']:.3f} "
            f"score={summary['score']} steps={summary['steps']}"
        )
        if gui is not None:
            if idx >= args.episodes:
                gui.root.after(500, gui.root.quit)
            else:
                gui.root.after(500, gui.reset)

    speed_ms = max(20, int(args.delay * 1000))
    gui = SnakeGameGUI(
        env.config,
        cell_size=args.cell_size,
        speed_ms=speed_ms,
        title="Snake DQN Inference",
        controller=controller,
        on_episode_end=on_episode_end,
    )

    print(
        f"Playing {args.episodes} episode(s) with grid {gui.env.config.width}x{gui.env.config.height} (GUI mode)"
    )
    gui.start()

    if results:
        avg_reward = np.mean([s["reward"] for s in results])
        avg_score = np.mean([s["score"] for s in results])
        avg_steps = np.mean([s["steps"] for s in results])
        print(
            f"Averages: reward={avg_reward:.3f} score={avg_score:.2f} steps={avg_steps:.1f}"
        )
    else:
        print("No episodes completed.")


if __name__ == "__main__":
    main()
