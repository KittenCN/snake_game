"""Training entry-point for a DQN agent that plays snake."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="Train a DQN agent to play snake using PyTorch")
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--width", type=int, default=12, help="Grid width")
    parser.add_argument("--height", type=int, default=12, help="Grid height")
    parser.add_argument("--initial-length", type=int, default=3, help="Initial snake length")
    parser.add_argument("--allow-wrap", action="store_true", help="Enable wrap-around movement")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--eval-interval", type=int, default=50, help="Episodes between evaluation runs")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes per checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for gradient updates")
    parser.add_argument("--replay-capacity", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--min-replay", type=int, default=2_000, help="Minimum replay buffer size before learning")
    parser.add_argument("--target-update", type=int, default=1_000, help="Target network update interval (steps)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay per step")
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256], help="Hidden layer sizes for the Q-network")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/dqn_snake.pt", help="Where to store the trained model")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for training logs")
    parser.add_argument("--render-frequency", type=int, default=0, help="Render ASCII board every N episodes (0 to disable)")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_seed = seed % (2**32)
    import random

    random.seed(random_seed)


def evaluate_agent(agent: DQNAgent, env: SnakeGameEnv, episodes: int) -> Dict[str, float]:
    rewards: List[float] = []
    scores: List[int] = []
    lengths: List[int] = []
    original_epsilon = agent.epsilon
    try:
        for _ in range(episodes):
            env.reset()
            state = flatten_observation(env.as_numpy(), agent.device)
            total_reward = 0.0
            for _ in range(10_000):
                action = agent.select_action(state, epsilon_override=0.0)
                _, reward, done, _ = env.step(Action(action))
                total_reward += reward
                state = flatten_observation(env.as_numpy(), agent.device)
                if done:
                    break
            rewards.append(total_reward)
            scores.append(env.score)
            lengths.append(env.steps)
    finally:
        agent.epsilon = original_epsilon
    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "avg_steps": float(np.mean(lengths)) if lengths else 0.0,
    }


def train() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    game_config = GameConfig(
        width=args.width,
        height=args.height,
        initial_length=args.initial_length,
        allow_wrap=args.allow_wrap,
        seed=args.seed,
    )
    env = SnakeGameEnv(game_config)
    observation_shape = env.observation_shape()
    state_dim = int(np.prod(observation_shape))
    action_dim = len(Action)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(args.hidden),
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay,
        target_update_interval=args.target_update,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay=args.epsilon_decay,
        device=args.device,
        game_config=game_config,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_log_{int(time.time())}.jsonl"

    best_reward = -math.inf
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        env.reset(seed=args.seed + episode)
        state = flatten_observation(env.as_numpy(), agent.device)
        episode_reward = 0.0
        losses: List[float] = []

        for _ in range(args.max_steps):
            action = agent.select_action(state)
            _, reward, done, info = env.step(Action(action))
            next_state = flatten_observation(env.as_numpy(), agent.device)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_reward += reward
            total_steps += 1
            if done:
                break

        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "score": env.score,
            "steps": env.steps,
            "epsilon": agent.epsilon,
            "avg_loss": float(np.mean(losses)) if losses else None,
        }

        if args.render_frequency and episode % args.render_frequency == 0:
            print("Episode", episode)
            print(env.render(to_string=True))

        if episode % args.eval_interval == 0:
            eval_stats = evaluate_agent(agent, env, args.eval_episodes)
            metrics.update({f"eval_{k}": v for k, v in eval_stats.items()})
            avg_reward = eval_stats["avg_reward"]
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(str(output_path))
                with open(output_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "game_config": asdict(game_config),
                            "train_args": vars(args),
                            "best_avg_reward": best_reward,
                        },
                        f,
                        indent=2,
                    )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:5d} | reward={episode_reward:7.3f} | score={env.score:3d} | steps={env.steps:4d} | "
                f"epsilon={agent.epsilon:.3f} | avg_loss={metrics['avg_loss']}"
            )

    if best_reward == -math.inf:
        agent.save(str(output_path))

    print(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    train()
