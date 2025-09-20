"""Play snake with a trained DQN agent."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

try:
    from .dqn_agent import DQNAgent, flatten_observation
    from .env import Action, GameConfig, SnakeGameEnv
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dqn_agent import DQNAgent, flatten_observation
    from env import Action, GameConfig, SnakeGameEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a trained DQN snake agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.05, help="Sleep time between steps (seconds)")
    parser.add_argument("--render", action="store_true", help="Render ASCII board to the console")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for the environment")
    return parser.parse_args()


def build_env_from_metadata(agent: DQNAgent, seed: int | None) -> SnakeGameEnv:
    if agent.game_config is not None:
        base_config = GameConfig(**asdict(agent.game_config))
    else:
        base_config = GameConfig()
    if seed is not None:
        base_config.seed = seed
    return SnakeGameEnv(base_config)


def run_episode(agent: DQNAgent, env: SnakeGameEnv, delay: float, render: bool) -> dict:
    env.reset()
    state = flatten_observation(env.as_numpy())
    total_reward = 0.0
    while True:
        action = agent.select_action(state, epsilon_override=0.0)
        obs, reward, done, info = env.step(Action(action))
        total_reward += reward
        state = flatten_observation(env.as_numpy())
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
    agent = DQNAgent.load(args.model)
    env = build_env_from_metadata(agent, args.seed)

    print(f"Loaded model from {Path(args.model).resolve()}")
    print(f"Playing {args.episodes} episode(s) with grid {env.config.width}x{env.config.height}")

    stats: List[dict] = []
    for idx in range(1, args.episodes + 1):
        print(f"Episode {idx}")
        result = run_episode(agent, env, args.delay, args.render)
        stats.append(result)
        print(
            f" -> reward={result['reward']:.3f} score={result['score']} steps={result['steps']}"
        )

    avg_reward = np.mean([s["reward"] for s in stats]) if stats else 0.0
    avg_score = np.mean([s["score"] for s in stats]) if stats else 0.0
    avg_steps = np.mean([s["steps"] for s in stats]) if stats else 0.0
    print(f"Averages: reward={avg_reward:.3f} score={avg_score:.2f} steps={avg_steps:.1f}")


if __name__ == "__main__":
    main()

