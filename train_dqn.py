"""Training entry-point for a DQN agent that plays snake."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes in this run")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--width", type=int, default=12, help="Grid width")
    parser.add_argument("--height", type=int, default=12, help="Grid height")
    parser.add_argument("--initial-length", type=int, default=3, help="Initial snake length")
    parser.add_argument("--allow-wrap", action="store_true", help="Enable wrap-around movement")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--eval-interval", type=int, default=50, help="Episodes between evaluation runs")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes per checkpoint")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for gradient updates")
    parser.add_argument("--replay-capacity", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--min-replay", type=int, default=5_000, help="Minimum replay buffer size before learning")
    parser.add_argument("--target-update", type=int, default=5_000, help="Fallback hard target update interval (steps)")
    parser.add_argument("--target-update-tau", type=float, default=0.005, help="Soft target update coefficient (0 disables)")
    parser.add_argument("--hard-update-interval", type=int, default=0, help="Explicit hard update interval when tau is 0")
    parser.add_argument("--disable-double-dqn", action="store_true", help="Disable Double DQN updates")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.99, help="Multiplicative epsilon decay per step")
    parser.add_argument("--reward-shaping-scale", type=float, default=0.1, help="Scaling factor for distance-based reward shaping")
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256], help="Hidden layer sizes for the Q-network")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/dqn_snake.pt", help="Where to store the trained model")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for training logs")
    parser.add_argument("--render-frequency", type=int, default=0, help="Render ASCII board every N episodes (0 to disable)")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random

    random.seed(seed % (2**32))


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")

    best_reward = -math.inf
    start_episode = 1

    if output_path.exists():
        print(f"Resuming training from {output_path}")
        agent = DQNAgent.load(
            str(output_path),
            device=args.device,
        )
        if agent.game_config is not None and asdict(agent.game_config) != asdict(game_config):
            print("Loaded agent configuration differs from CLI arguments; using configuration from checkpoint.")
            game_config = agent.game_config
            env = SnakeGameEnv(game_config)
            observation_shape = env.observation_shape()
            state_dim = int(np.prod(observation_shape))
            action_dim = len(Action)
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as meta_fp:
                    previous_meta = json.load(meta_fp)
                best_reward = previous_meta.get("best_avg_reward", best_reward)
                start_episode = max(1, previous_meta.get("episodes_completed", 0) + 1)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse metadata file {meta_path}; continuing from defaults.")
    else:
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
            target_update_tau=args.target_update_tau,
            hard_update_interval=args.hard_update_interval,
            use_double_dqn=not args.disable_double_dqn,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
            device=args.device,
            game_config=game_config,
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_log_{int(time.time())}.jsonl"

    def write_metadata(best_reward_value: Optional[float], episodes_completed: int) -> None:
        metadata = {
            "game_config": asdict(agent.game_config or game_config),
            "train_args": vars(args),
            "best_avg_reward": best_reward_value,
            "episodes_completed": episodes_completed,
            "epsilon": agent.epsilon,
            "learn_step_counter": agent.learn_step_counter,
        }
        with meta_path.open("w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2)

    for episode in range(start_episode, start_episode + args.episodes):
        env.reset(seed=args.seed + episode)
        state = flatten_observation(env.as_numpy(), agent.device)
        episode_env_reward = 0.0
        episode_shaped_reward = 0.0
        losses: List[float] = []

        for _ in range(args.max_steps):
            prev_head = env.snake[0]
            prev_food = env.food
            prev_distance = None
            if args.reward_shaping_scale > 0 and prev_food is not None:
                prev_distance = manhattan_distance(prev_head, prev_food)

            action = agent.select_action(state)
            obs, reward, done, info = env.step(Action(action))
            next_state = flatten_observation(env.as_numpy(), agent.device)

            shaped_reward = reward
            if (
                args.reward_shaping_scale > 0
                and prev_distance is not None
                and info.get("event") != "ate_food"
                and prev_food is not None
            ):
                new_distance = manhattan_distance(env.snake[0], prev_food)
                shaped_reward += args.reward_shaping_scale * (prev_distance - new_distance)

            agent.remember(state, action, shaped_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_env_reward += reward
            episode_shaped_reward += shaped_reward
            if done:
                break

        metrics = {
            "episode": episode,
            "reward": episode_env_reward,
            "shaped_reward": episode_shaped_reward,
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
            write_metadata(best_reward if best_reward != -math.inf else None, episode)

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if episode % 10 == 0 or episode == start_episode:
            print(
                f"Episode {episode:5d} | reward={episode_env_reward:7.3f} | score={env.score:3d} | steps={env.steps:4d} | "
                f"epsilon={agent.epsilon:.3f} | avg_loss={metrics['avg_loss']} | shaped={episode_shaped_reward:7.3f}"
            )

    last_episode = start_episode + args.episodes - 1
    if best_reward == -math.inf:
        agent.save(str(output_path))
        write_metadata(None, last_episode)
    else:
        write_metadata(best_reward, last_episode)

    print(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    train()
