"""Training entry-point for a DQN agent that plays snake."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

GridSeed = Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play snake using PyTorch")
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes in this run")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--width", type=int, default=12, help="Grid width")
    parser.add_argument("--height", type=int, default=12, help="Grid height")
    parser.add_argument("--initial-length", type=int, default=3, help="Initial snake length")
    parser.add_argument("--allow-wrap", action="store_true", help="Enable wrap-around movement")
    parser.add_argument("--seed", type=int, default=None, help="Global seed (omit for fully random training)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Episodes between evaluation runs")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes per checkpoint")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for gradient updates")
    parser.add_argument("--replay-capacity", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--min-replay", type=int, default=5_000, help="Minimum replay buffer size before learning")
    parser.add_argument("--target-update", type=int, default=5_000, help="Fallback hard target update interval (steps)")
    parser.add_argument("--target-update-tau", type=float, default=0.0075, help="Soft target update coefficient (0 disables)")
    parser.add_argument("--hard-update-interval", type=int, default=0, help="Explicit hard update interval when tau is 0")
    parser.add_argument("--disable-double-dqn", action="store_true", help="Disable Double DQN updates")
    parser.add_argument("--disable-dueling", action="store_true", help="Disable dueling network architecture")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-final", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay per step")
    parser.add_argument("--reward-step", type=float, default=-0.002, help="Base reward per step (typically negative)")
    parser.add_argument("--reward-food", type=float, default=5.0, help="Reward granted for eating food")
    parser.add_argument("--reward-death", type=float, default=-2.0, help="Penalty for dying")
    parser.add_argument("--reward-shaping-scale", type=float, default=0.15, help="Scaling factor for distance-based reward shaping")
    parser.add_argument("--max-idle-steps", type=int, default=120, help="Terminate episode after this many steps without eating (0 disables)")
    parser.add_argument("--idle-penalty", type=float, default=-4.0, help="Additional penalty applied on idle timeout")
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256], help="Hidden layer sizes for the Q-network")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/dqn_snake.pt", help="Where to store the trained model")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for training logs")
    parser.add_argument("--render-frequency", type=int, default=0, help="Render ASCII board every N episodes (0 to disable)")
    return parser.parse_args()


def set_global_seed(seed: Optional[int]) -> random.Random:
    rng = random.Random()
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
        print(f"Using generated seed {seed}")
    seed32 = seed % (2**32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)
    random.seed(seed32)
    rng.seed(seed32)
    return rng


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def evaluate_agent(
    agent: DQNAgent,
    env: SnakeGameEnv,
    episodes: int,
    rng: Optional[random.Random],
) -> Dict[str, float]:
    rewards: List[float] = []
    scores: List[int] = []
    lengths: List[int] = []
    original_epsilon = agent.epsilon
    try:
        for _ in range(episodes):
            eval_seed: GridSeed = rng.randint(0, 2**32 - 1) if rng is not None else None
            env.reset(seed=eval_seed)
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
    rng = set_global_seed(args.seed)

    game_config = GameConfig(
        width=args.width,
        height=args.height,
        initial_length=args.initial_length,
        reward_step=args.reward_step,
        reward_food=args.reward_food,
        reward_death=args.reward_death,
        allow_wrap=args.allow_wrap,
        seed=None,
        max_idle_steps=args.max_idle_steps,
        idle_penalty=args.idle_penalty,
    )

    train_env = SnakeGameEnv(game_config)
    eval_env = SnakeGameEnv(game_config)
    observation_shape = train_env.observation_shape()
    state_dim = int(np.prod(observation_shape))
    action_dim = len(Action)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")

    best_reward = -math.inf
    start_episode = 1

    if output_path.exists():
        print(f"Resuming training from {output_path}")
        agent = DQNAgent.load(str(output_path), device=args.device)
        agent.game_config = game_config
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as meta_fp:
                    previous_meta = json.load(meta_fp)
                best_reward = previous_meta.get("best_avg_reward", best_reward)
                start_episode = max(1, previous_meta.get("episodes_completed", 0) + 1)
                print(f"Resuming from episode {start_episode} with best avg reward {best_reward}")
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
            use_dueling=not args.disable_dueling,
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
        episode_seed: GridSeed = rng.randint(0, 2**32 - 1) if rng is not None else None
        train_env.reset(seed=episode_seed)
        state = flatten_observation(train_env.as_numpy(), agent.device)
        episode_env_reward = 0.0
        episode_shaped_reward = 0.0
        losses: List[float] = []

        for _ in range(args.max_steps):
            previous_food = train_env.food
            previous_distance = (
                manhattan_distance(train_env.snake[0], previous_food)
                if args.reward_shaping_scale > 0 and previous_food is not None
                else None
            )

            action = agent.select_action(state)
            obs, reward, done, info = train_env.step(Action(action))
            next_state = flatten_observation(train_env.as_numpy(), agent.device)

            shaped_reward = reward
            if (
                args.reward_shaping_scale > 0
                and previous_distance is not None
                and previous_food is not None
                and info.get("event") != "ate_food"
            ):
                new_distance = manhattan_distance(train_env.snake[0], previous_food)
                shaped_reward += args.reward_shaping_scale * (previous_distance - new_distance)

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
            "score": train_env.score,
            "steps": train_env.steps,
            "epsilon": agent.epsilon,
            "avg_loss": float(np.mean(losses)) if losses else None,
        }

        if args.render_frequency and episode % args.render_frequency == 0:
            print("Episode", episode)
            print(train_env.render(to_string=True))

        if episode % args.eval_interval == 0:
            eval_stats = evaluate_agent(agent, eval_env, args.eval_episodes, rng)
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
                f"Episode {episode:5d} | reward={episode_env_reward:7.3f} | score={train_env.score:3d} | steps={train_env.steps:4d} | "
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
