"""Training entry-point for a DQN agent that plays snake."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import shutil
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
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for gradient updates")
    parser.add_argument("--replay-capacity", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--min-replay", type=int, default=5_000, help="Minimum replay buffer size before learning")
    parser.add_argument("--target-update", type=int, default=5_000, help="Fallback hard target update interval (steps)")
    parser.add_argument("--target-update-tau", type=float, default=0.006, help="Soft target update coefficient (0 disables)")
    parser.add_argument("--hard-update-interval", type=int, default=0, help="Explicit hard update interval when tau is 0")
    parser.add_argument("--disable-double-dqn", action="store_true", help="Disable Double DQN updates")
    parser.add_argument("--disable-dueling", action="store_true", help="Disable dueling network architecture")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-final", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Multiplicative epsilon decay per step")
    parser.add_argument("--reward-step", type=float, default=-0.003, help="Base reward per step (typically negative)")
    parser.add_argument("--reward-food", type=float, default=5.0, help="Reward granted for eating food")
    parser.add_argument("--reward-death", type=float, default=-2.0, help="Penalty for dying")
    parser.add_argument("--reward-shaping-scale", type=float, default=0.18, help="Scaling factor for distance-based reward shaping")
    parser.add_argument("--max-idle-steps", type=int, default=90, help="Terminate episode after this many steps without eating (0 disables)")
    parser.add_argument("--idle-penalty", type=float, default=-5.0, help="Additional penalty applied on idle timeout")
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256], help="Hidden layer sizes for the Q-network")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/dqn_snake.pt", help="Where to store the trained model")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for training logs")
    parser.add_argument("--render-frequency", type=int, default=0, help="Render ASCII board every N episodes (0 to disable)")
    parser.add_argument("--segment-length", type=int, default=0, help="Episodes per training segment for checkpointing (0 disables)")
    parser.add_argument("--save-segment-checkpoints", action="store_true", help="If set, keep a checkpoint at the end of each completed segment")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Number of evaluation windows without sufficient improvement before early stop (0 disables)")
    parser.add_argument("--early-stop-delta", type=float, default=0.0, help="Minimum eval reward improvement required to reset the early-stop counter")
    parser.add_argument("--resume-best-on-decline", action="store_true", help="Reload the best checkpoint whenever an evaluation does not improve")
    parser.add_argument("--resume-decline-threshold", type=float, default=0.0, help="Minimum drop in eval reward before reloading the best checkpoint (0 disables)")
    parser.add_argument("--resume-decline-cooldown", type=int, default=0, help="Number of evaluation windows to wait before reloading the best checkpoint again (0 disables)")
    parser.add_argument("--early-stop-min-evals", type=int, default=0, help="Minimum number of evaluation windows before early stopping can trigger")
    parser.add_argument("--best-history-limit", type=int, default=0, help="Maximum number of best checkpoints to keep (0 disables history)")
    parser.add_argument("--best-history-dir", type=str, default=None, help="Optional directory to store best checkpoint snapshots")
    parser.add_argument("--disable-early-stop", action="store_true", help="Disable evaluation-based early stopping")
    parser.add_argument("--disable-train-best", action="store_true", help="Disable saving checkpoints based on training performance")
    parser.add_argument("--train-best-metric", type=str, default="reward", choices=["reward", "score"], help="Metric used when saving training-best checkpoints")
    parser.add_argument("--train-best-delta", type=float, default=5.0, help="Minimum improvement required to refresh the training-best checkpoint")
    parser.add_argument("--network-version", type=int, choices=[1, 2], default=2, help="Network architecture version (1 legacy CNN, 2 enhanced residual CNN)")
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
            state = flatten_observation(env, agent.device, expected_channels=agent.obs_shape[0])
            total_reward = 0.0
            for _ in range(10_000):
                action = agent.select_action(state, epsilon_override=0.0)
                _, reward, done, _ = env.step(Action(action))
                total_reward += reward
                state = flatten_observation(env, agent.device, expected_channels=agent.obs_shape[0])
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
    train_env.reset()
    eval_env.reset()
    init_expected_channels = 3 if args.network_version == 1 else None
    initial_state = flatten_observation(train_env, device="cpu", expected_channels=init_expected_channels)
    obs_shape = tuple(int(dim) for dim in initial_state.shape)
    state_dim = int(np.prod(obs_shape))
    action_dim = len(Action)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")

    best_history_limit = max(0, args.best_history_limit)
    best_history_dir = Path(args.best_history_dir) if args.best_history_dir else output_path.parent / f"{output_path.stem}_history"
    best_history: list[dict[str, object]] = []

    train_best_enabled = not args.disable_train_best
    train_best_metric = args.train_best_metric
    train_best_delta = args.train_best_delta
    train_best_value = -math.inf
    train_best_episode: Optional[int] = None
    train_best_path = output_path.with_name(f"{output_path.stem}_best_{train_best_metric}.pt")

    best_reward = -math.inf
    best_eval_episode: Optional[int] = None
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
                best_eval_episode = previous_meta.get("best_eval_episode", best_eval_episode)
                history_meta = previous_meta.get("best_history")
                if isinstance(history_meta, list):
                    for item in history_meta:
                        if not isinstance(item, dict):
                            continue
                        path_value = item.get("path")
                        reward_value = item.get("reward")
                        episode_value = item.get("episode")
                        if path_value is None or reward_value is None:
                            continue
                        snapshot_path = Path(path_value)
                        if snapshot_path.exists():
                            best_history.append({
                                "path": str(snapshot_path),
                                "reward": float(reward_value),
                                "episode": int(episode_value) if episode_value is not None else None,
                            })
                stored_metric = previous_meta.get("train_best_metric")
                if stored_metric and stored_metric not in {"reward", "score"}:
                    stored_metric = None
                if stored_metric and stored_metric != train_best_metric:
                    print("Train-best metric changed from metadata; resetting train-best tracker.")
                elif stored_metric:
                    train_best_metric = stored_metric
                    train_best_value = previous_meta.get("train_best_value", train_best_value)
                    train_best_episode = previous_meta.get("train_best_episode", train_best_episode)
                    train_best_path = output_path.with_name(f"{output_path.stem}_best_{train_best_metric}.pt")
                stored_path = previous_meta.get("train_best_path")
                if stored_path and (stored_metric is None or stored_metric == train_best_metric):
                    train_best_path = Path(stored_path)
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
            obs_shape=obs_shape,
            network_version=args.network_version,
        )

    if best_history_limit <= 0:
        if best_history:
            for entry in best_history:
                path_str = entry.get("path") if isinstance(entry, dict) else None
                if path_str:
                    snapshot_path = Path(path_str)
                    if snapshot_path.exists():
                        try:
                            snapshot_path.unlink()
                        except OSError:
                            pass
        best_history = []
    elif best_history:
        best_history.sort(key=lambda entry: entry.get("reward", float("-inf")), reverse=True)
        if best_history_limit > 0 and len(best_history) > best_history_limit:
            for entry in best_history[best_history_limit:]:
                path_str = entry.get("path")
                if path_str:
                    snapshot_path = Path(path_str)
                    if snapshot_path.exists():
                        try:
                            snapshot_path.unlink()
                        except OSError:
                            pass
            best_history = best_history[:best_history_limit]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_log_{int(time.time())}.jsonl"


    segment_length = max(0, args.segment_length)
    patience_counter = 0
    stop_training = False
    final_episode = start_episode - 1
    eval_counter = 0
    last_reload_eval = -1
    early_stop_enabled = not args.disable_early_stop

    def write_metadata(best_reward_value: Optional[float], episodes_completed: int) -> None:
        metadata = {
            "game_config": asdict(agent.game_config or game_config),
            "train_args": vars(args),
            "best_avg_reward": best_reward_value,
            "episodes_completed": episodes_completed,
            "epsilon": agent.epsilon,
            "learn_step_counter": agent.learn_step_counter,
            "best_eval_episode": best_eval_episode,
            "best_history": best_history,
            "train_best_metric": train_best_metric if train_best_enabled else None,
            "train_best_value": float(train_best_value) if (train_best_enabled and train_best_value != -math.inf) else None,
            "train_best_episode": train_best_episode if train_best_enabled else None,
            "train_best_path": str(train_best_path) if (train_best_enabled and train_best_value != -math.inf) else None,
        }
        with meta_path.open("w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2)

    for episode in range(start_episode, start_episode + args.episodes):
        episode_seed: GridSeed = rng.randint(0, 2**32 - 1) if rng is not None else None
        train_env.reset(seed=episode_seed)
        state = flatten_observation(train_env, agent.device, expected_channels=agent.obs_shape[0])
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
            next_state = flatten_observation(train_env, agent.device, expected_channels=agent.obs_shape[0])

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

        if train_best_enabled:
            metric_value = episode_env_reward if train_best_metric == "reward" else float(train_env.score)
            if train_best_value == -math.inf or metric_value >= train_best_value + train_best_delta:
                train_best_value = metric_value
                train_best_episode = episode
                try:
                    train_best_path.parent.mkdir(parents=True, exist_ok=True)
                    agent.save(str(train_best_path))
                    metric_display = f"{metric_value:.3f}" if train_best_metric == "reward" else f"{int(metric_value)}"
                    print(f"New training-best {train_best_metric} {metric_display} at episode {episode}. Saved to {train_best_path}")
                except Exception as exc:
                    print(f"Warning: Could not save training-best checkpoint: {exc}")
            metrics[f"train_best_{train_best_metric}"] = train_best_value if train_best_value != -math.inf else None

        if args.render_frequency and episode % args.render_frequency == 0:
            print("Episode", episode)
            print(train_env.render(to_string=True))

        if episode % args.eval_interval == 0:
            eval_counter += 1
            eval_stats = evaluate_agent(agent, eval_env, args.eval_episodes, rng)
            metrics.update({f"eval_{k}": v for k, v in eval_stats.items()})
            avg_reward = eval_stats["avg_reward"]
            improvement_threshold = best_reward + args.early_stop_delta if best_reward != -math.inf else -math.inf
            improved = avg_reward > improvement_threshold
            if improved:
                best_reward = avg_reward
                best_eval_episode = episode
                patience_counter = 0
                last_reload_eval = eval_counter
                agent.save(str(output_path))
                print(f"New best avg reward {best_reward:.3f} at episode {episode}. Saved to {output_path}")
                if best_history_limit != 0:
                    try:
                        best_history_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                    snapshot_path = best_history_dir / f"{output_path.stem}_ep{episode}_reward{best_reward:.2f}.pt"
                    try:
                        shutil.copy2(output_path, snapshot_path)
                        best_history.append({
                            "path": str(snapshot_path),
                            "reward": float(best_reward),
                            "episode": episode,
                        })
                    except OSError as exc:
                        print(f"Warning: Could not copy best checkpoint to {snapshot_path}: {exc}")
                    if best_history_limit > 0:
                        best_history.sort(key=lambda entry: entry.get("reward", float("-inf")), reverse=True)
                        while len(best_history) > best_history_limit:
                            entry = best_history.pop()
                            path_str = entry.get("path")
                            if path_str:
                                old_path = Path(path_str)
                                if old_path.exists():
                                    try:
                                        old_path.unlink()
                                    except OSError:
                                        pass
            else:
                decline = best_reward - avg_reward if best_reward != -math.inf else 0.0
                reload_allowed = (
                    args.resume_best_on_decline
                    and best_reward != -math.inf
                    and output_path.exists()
                )
                cooldown_ok = args.resume_decline_cooldown <= 0 or last_reload_eval < 0 or (eval_counter - last_reload_eval) >= args.resume_decline_cooldown
                threshold_ok = args.resume_decline_threshold <= 0.0 or decline >= args.resume_decline_threshold
                if reload_allowed and cooldown_ok and threshold_ok:
                    try:
                        checkpoint = torch.load(output_path, map_location=agent.device)
                        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
                        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
                        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        restored_epsilon = checkpoint.get("metadata", {}).get("epsilon")
                        if restored_epsilon is not None:
                            agent.epsilon = min(agent.epsilon, float(restored_epsilon))
                        restored_steps = checkpoint.get("metadata", {}).get("learn_step_counter")
                        if restored_steps is not None:
                            agent.learn_step_counter = int(restored_steps)
                        last_reload_eval = eval_counter
                        print("Reloaded best checkpoint after evaluation decline.")
                    except Exception as exc:
                        print(f"Warning: Could not reload best checkpoint: {exc}")
                if early_stop_enabled and args.early_stop_patience > 0 and best_reward != -math.inf:
                    if args.early_stop_min_evals <= 0 or eval_counter >= args.early_stop_min_evals:
                        patience_counter += 1
                        if patience_counter >= args.early_stop_patience:
                            stop_training = True
                            print(f"Early stopping triggered at episode {episode} (best eval reward {best_reward:.3f})")
            write_metadata(best_reward if best_reward != -math.inf else None, episode)

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        final_episode = episode

        if segment_length > 0:
            episodes_into_run = episode - start_episode + 1
            if episodes_into_run % segment_length == 0:
                if args.save_segment_checkpoints:
                    segment_path = output_path.with_name(f"{output_path.stem}_ep{episode}.pt")
                    agent.save(str(segment_path))
                    print(f"Saved segment checkpoint to {segment_path}, best eval reward {best_reward if best_reward != -math.inf else 'N/A'}")
                write_metadata(best_reward if best_reward != -math.inf else None, episode)
                best_display = f"{best_reward:.3f}" if best_reward != -math.inf else "N/A"
                print(f"Segment complete at episode {episode}. Best eval reward: {best_display}")

        if stop_training:
            break

        if episode % 10 == 0 or episode == start_episode:
            print(
                f"Episode {episode:5d} | reward={episode_env_reward:7.3f} | score={train_env.score:3d} | steps={train_env.steps:4d} | "
                f"epsilon={agent.epsilon:.3f} | avg_loss={metrics['avg_loss']} | shaped={episode_shaped_reward:7.3f}"
            )

    last_episode = final_episode if final_episode >= start_episode else start_episode + args.episodes - 1
    if best_reward == -math.inf:
        agent.save(str(output_path))
        write_metadata(None, last_episode)
    else:
        write_metadata(best_reward, last_episode)

    if stop_training:
        print(f"Early stopping triggered; halted at episode {last_episode}.")

    print(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    train()
