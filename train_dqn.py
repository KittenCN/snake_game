"""Train a reproducible DQN agent to play Snake.

Version 3 deliberately separates resumable ``latest`` checkpoints from evaluated
``best`` checkpoints.  Legacy v1/v2 checkpoints remain loadable when supplied via
``--resume-from``, but new training uses a three-action relative control space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

try:
    from .dqn_agent import DQNAgent, flatten_observation
    from .env import Action, GameConfig, RelativeAction, SnakeGameEnv
except ImportError:
    from dqn_agent import DQNAgent, flatten_observation
    from env import Action, GameConfig, RelativeAction, SnakeGameEnv


CHECKPOINT_FORMAT = 3
V3_OBSERVATION_CHANNELS = 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train a DQN Snake agent using PyTorch"
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Episode cap; 0 uses 20x board area and relies primarily on the idle limit",
    )
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--initial-length", type=int, default=3)
    parser.add_argument("--allow-wrap", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--min-replay", type=int, default=2_000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=500_000)
    parser.add_argument("--target-update", type=int, default=5_000)
    parser.add_argument("--target-update-tau", type=float, default=0.005)
    parser.add_argument("--hard-update-interval", type=int, default=0)
    parser.add_argument("--disable-double-dqn", action="store_true")
    parser.add_argument("--disable-dueling", action="store_true")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000)
    parser.add_argument(
        "--resume-epsilon",
        type=float,
        default=0.25,
        help="Exploration floor after resume because replay is intentionally not checkpointed",
    )
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--reward-step", type=float, default=-0.003)
    parser.add_argument("--reward-food", type=float, default=5.0)
    parser.add_argument("--reward-death", type=float, default=-5.0)
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=1.0,
        help="Scale for potential-based food/topology shaping; 0 disables",
    )
    parser.add_argument("--max-idle-steps", type=int, default=90)
    parser.add_argument("--idle-growth-per-food", type=int, default=2)
    parser.add_argument("--idle-penalty", type=float, default=-5.0)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Allow faster but potentially non-reproducible backend algorithms",
    )
    parser.add_argument("--network-version", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--output", default="models/dqn_snake_v3_best.pt")
    parser.add_argument("--latest-output", default="models/dqn_snake_v3_latest.pt")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore an existing latest checkpoint"
    )
    parser.add_argument(
        "--overwrite-fresh-output",
        action="store_true",
        help="Delete exact existing latest/best outputs before an intentional fresh run",
    )
    parser.add_argument(
        "--reset-best-evaluation",
        action="store_true",
        help="Discard the resumed best threshold when changing evaluation identity/output",
    )
    parser.add_argument(
        "--ignore-resume-metadata",
        action="store_true",
        help="Explicitly allow a missing or mismatched resume sidecar",
    )
    parser.add_argument(
        "--allow-environment-change",
        action="store_true",
        help="Allow an intentional MDP/config change while warm-starting weights",
    )
    parser.add_argument(
        "--allow-seed-change",
        action="store_true",
        help="Allow an intentional training episode-seed stream change on resume",
    )
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--render-frequency", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-delta", type=float, default=0.25)
    # Accepted only to make old commands fail safe instead of re-enabling the rollback loop.
    parser.add_argument(
        "--resume-best-on-decline", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume-decline-threshold", type=float, default=0.0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume-decline-cooldown", type=int, default=0, help=argparse.SUPPRESS
    )
    args = parser.parse_args(raw_argv)
    args._provided_options = sorted(
        {token.split("=", 1)[0] for token in raw_argv if token.startswith("--")}
    )
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error("episodes and eval-episodes must be positive")
    if args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("eval-interval and checkpoint-interval must be positive")
    if args.train_frequency <= 0 or args.gradient_steps <= 0:
        parser.error("train-frequency and gradient-steps must be positive")
    if Path(args.output).resolve() == Path(args.latest_output).resolve():
        parser.error("--output and --latest-output must be different paths")
    if args.resume_best_on_decline:
        print(
            "Warning: --resume-best-on-decline is retired; no model rollback will occur."
        )
    return args


def set_global_seed(seed: int) -> None:
    seed32 = int(seed) % (2**32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)


def deterministic_episode_seed(base_seed: int, episode: int, stream: int = 0) -> int:
    """Derive an episode seed without coupling training seeds to evaluation calls."""
    sequence = np.random.SeedSequence([int(base_seed), int(episode), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def episode_step_limit(config: GameConfig, configured_limit: int) -> int:
    if configured_limit > 0:
        return configured_limit
    if config.max_episode_steps > 0:
        return config.max_episode_steps
    return config.width * config.height * 20


def _neighbours(
    env: SnakeGameEnv, position: tuple[int, int]
) -> Iterable[tuple[int, int]]:
    for action in Action:
        dx, dy = action.vector
        x, y = position[0] + dx, position[1] + dy
        if env.config.allow_wrap:
            yield x % env.config.width, y % env.config.height
        elif 0 <= x < env.config.width and 0 <= y < env.config.height:
            yield x, y


def _food_distance(env: SnakeGameEnv) -> int:
    if env.food is None:
        return 0
    hx, hy = env.snake[0]
    fx, fy = env.food
    dx, dy = abs(hx - fx), abs(hy - fy)
    if env.config.allow_wrap:
        dx = min(dx, env.config.width - dx)
        dy = min(dy, env.config.height - dy)
    return dx + dy


def state_potential(env: SnakeGameEnv) -> float:
    """Bounded potential combining food progress and late-game topology.

    The tail is considered vacating for reachability.  This is deliberately a
    potential, not an immediate 'move toward food' reward, so safe detours do not
    accumulate a permanent penalty.
    """
    if env.done or env.food is None:
        return 0.0
    snake = env.snake
    head, tail = snake[0], snake[-1]
    blocked = set(snake[1:-1])
    queue = deque([head])
    reachable = {head}
    while queue:
        current = queue.popleft()
        for candidate in _neighbours(env, current):
            if candidate in blocked or candidate in reachable:
                continue
            reachable.add(candidate)
            queue.append(candidate)
    max_distance = max(1, env.config.width + env.config.height - 2)
    food_closeness = 1.0 - min(1.0, _food_distance(env) / max_distance)
    traversable = max(1, env.config.width * env.config.height - len(blocked))
    space_ratio = min(1.0, len(reachable) / traversable)
    tail_reachable = 1.0 if tail in reachable else 0.0
    return 0.55 * food_closeness + 0.30 * space_ratio + 0.15 * tail_reachable


def potential_shaping(
    previous_potential: float,
    env: SnakeGameEnv,
    *,
    gamma: float,
    scale: float,
    terminal: bool | None = None,
) -> float:
    if scale <= 0:
        return 0.0
    next_potential = (
        0.0 if (env.done if terminal is None else terminal) else state_potential(env)
    )
    return float(scale * (gamma * next_potential - previous_potential))


def action_mask(agent: DQNAgent, env: SnakeGameEnv) -> list[bool]:
    if agent.action_dim == len(RelativeAction):
        return [True] * agent.action_dim
    if agent.action_dim == len(Action):
        legal = set(env.legal_actions())
        return [action in legal for action in Action]
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def step_agent_action(
    agent: DQNAgent, env: SnakeGameEnv, action_index: int
) -> tuple[dict[str, object], float, bool, dict[str, object]]:
    if agent.action_dim == len(RelativeAction):
        return env.step_relative(RelativeAction(action_index))
    if agent.action_dim == len(Action):
        return env.step(Action(action_index))
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def evaluate_agent(
    agent: DQNAgent,
    game_config: GameConfig,
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, Any]:
    rewards: list[float] = []
    scores: list[int] = []
    steps_taken: list[int] = []
    events: Counter[str] = Counter()
    truncated_count = 0
    for seed in seeds:
        env = SnakeGameEnv(game_config)
        env.reset(seed=int(seed))
        state = flatten_observation(
            env, agent.device, expected_channels=agent.obs_shape[0]
        )
        total_reward = 0.0
        terminal_event = "truncated"
        for _ in range(max_steps):
            chosen = agent.select_action(
                state, epsilon_override=0.0, action_mask=action_mask(agent, env)
            )
            _, reward, done, info = step_agent_action(agent, env, chosen)
            total_reward += reward
            state = flatten_observation(
                env, agent.device, expected_channels=agent.obs_shape[0]
            )
            if done:
                terminal_event = str(info.get("event", "terminated"))
                if bool(info.get("truncated")):
                    truncated_count += 1
                break
        else:
            truncated_count += 1
        rewards.append(total_reward)
        scores.append(env.score)
        steps_taken.append(env.steps)
        events[terminal_event] += 1

    def distribution(values: Sequence[float | int]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        margin = 1.96 * std / math.sqrt(len(array))
        return {
            "mean": mean,
            "std": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "reward": distribution(rewards),
        "score": distribution(scores),
        "steps": distribution(steps_taken),
        "terminal_events": dict(sorted(events.items())),
        "truncated_count": truncated_count,
        "episodes": len(seeds),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".meta.json")


def evaluation_identity(
    train_args: argparse.Namespace, game_config: GameConfig
) -> dict[str, Any]:
    return {
        "selection_metric": "raw_score_mean",
        "safety_fallback": False,
        "run_seed": train_args.seed,
        "eval_seed_base": train_args.eval_seed_base,
        "eval_episodes": train_args.eval_episodes,
        "game_config": asdict(game_config),
    }


def effective_agent_config(agent: DQNAgent) -> dict[str, Any]:
    learning_rates = [float(group["lr"]) for group in agent.optimizer.param_groups]
    return {
        "network_version": agent.network_version,
        "hidden_sizes": list(agent.hidden_sizes),
        "lr": learning_rates,
        "gamma": agent.gamma,
        "batch_size": agent.batch_size,
        "replay_capacity": agent.replay_capacity,
        "min_replay_size": agent.min_replay_size,
        "n_step": agent.n_step,
        "per_alpha": agent.per_alpha,
        "per_beta_start": agent.per_beta_start,
        "per_beta_frames": agent.per_beta_frames,
        "target_update_interval": agent.target_update_interval,
        "target_update_tau": agent.target_update_tau,
        "hard_update_interval": agent.hard_update_interval,
        "use_double_dqn": agent.use_double_dqn,
        "use_dueling": agent.use_dueling,
        "epsilon_start": agent.epsilon_start,
        "epsilon_final": agent.epsilon_final,
        "epsilon_decay_steps": agent.epsilon_decay_steps,
        "amp_enabled": agent.amp_enabled,
    }


def save_checkpoint(
    agent: DQNAgent,
    path: Path,
    *,
    episode: int,
    run_seed: int,
    best_eval_score: float | None,
    best_eval_episode: int | None,
    train_args: argparse.Namespace,
    checkpoint_role: str,
    best_checkpoint_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    checkpoint_sha256 = _sha256(path)
    referenced_best_path: str | None = None
    referenced_best_sha256: str | None = None
    if best_eval_score is not None:
        candidate = path if checkpoint_role == "best_eval" else best_checkpoint_path
        if candidate is not None and candidate.exists():
            referenced_best_path = str(candidate.resolve())
            referenced_best_sha256 = (
                checkpoint_sha256
                if candidate.resolve() == path.resolve()
                else _sha256(candidate)
            )
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "episodes_completed": episode,
        "run_seed": run_seed,
        "best_eval_score": best_eval_score,
        "best_eval_episode": best_eval_episode,
        "best_checkpoint_path": referenced_best_path,
        "best_checkpoint_sha256": referenced_best_sha256,
        "evaluation_identity": evaluation_identity(
            train_args, agent.game_config or GameConfig()
        ),
        "network_version": agent.network_version,
        "action_dim": agent.action_dim,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
        "epsilon": agent.epsilon,
        "replay_size_at_save": len(agent.replay_buffer),
        "replay_restored": False,
        "game_config": asdict(agent.game_config) if agent.game_config else None,
        "effective_agent_config": effective_agent_config(agent),
        "train_args": vars(train_args),
    }
    _atomic_json(sidecar_path(path), payload)


def load_resume_metadata(path: Path, *, ignore_mismatch: bool) -> dict[str, Any]:
    meta_path = sidecar_path(path)
    if not meta_path.exists():
        if ignore_mismatch:
            return {}
        raise RuntimeError(
            f"Resume metadata is missing: {meta_path}. Use --ignore-resume-metadata "
            "only for an intentional legacy warm start."
        )
    with meta_path.open("r", encoding="utf-8-sig") as stream:
        metadata = json.load(stream)
    expected = metadata.get("checkpoint_sha256")
    actual = _sha256(path)
    if expected != actual:
        message = f"Checkpoint/metadata mismatch for {path}: expected SHA-256 {expected}, got {actual}."
        if not ignore_mismatch:
            raise RuntimeError(message + " Refusing a silent stale-best resume.")
        print("Warning:", message, "Treating it as a legacy warm start.")
        return {}
    return metadata


def validate_resume_identity(metadata: dict[str, Any], agent: DQNAgent) -> None:
    """Ensure sidecar architecture claims describe the checkpoint actually loaded."""
    if not metadata:
        return
    expected = {
        "network_version": agent.network_version,
        "action_dim": agent.action_dim,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Resume sidecar fields conflict with the checkpoint payload: "
            + ", ".join(
                f"{key}=sidecar:{pair[0]!r}/checkpoint:{pair[1]!r}"
                for key, pair in mismatches.items()
            )
        )


def validate_v3_contract(agent: DQNAgent) -> None:
    if agent.network_version < 3:
        return
    if agent.action_dim != len(RelativeAction):
        raise RuntimeError(
            "A v3 Snake checkpoint must use exactly three relative actions."
        )
    if agent.obs_shape[0] != V3_OBSERVATION_CHANNELS:
        raise RuntimeError(
            f"A v3 Snake checkpoint must use {V3_OBSERVATION_CHANNELS} observation channels; "
            f"got {agent.obs_shape[0]}. Start a fresh v3 run instead of silently dropping state."
        )


def validate_resume_agent_options(args: argparse.Namespace, agent: DQNAgent) -> None:
    """Reject explicitly requested hyperparameters that resume cannot silently apply."""
    provided = set(getattr(args, "_provided_options", ()))
    checks: dict[str, tuple[Any, Any]] = {
        "--network-version": (args.network_version, agent.network_version),
        "--hidden": (tuple(args.hidden), tuple(agent.hidden_sizes)),
        "--lr": (float(args.lr), float(agent.optimizer.param_groups[0]["lr"])),
        "--gamma": (args.gamma, agent.gamma),
        "--batch-size": (args.batch_size, agent.batch_size),
        "--replay-capacity": (args.replay_capacity, agent.replay_capacity),
        "--min-replay": (args.min_replay, agent.min_replay_size),
        "--n-step": (args.n_step, agent.n_step),
        "--per-alpha": (args.per_alpha, agent.per_alpha),
        "--per-beta-start": (args.per_beta_start, agent.per_beta_start),
        "--per-beta-frames": (args.per_beta_frames, agent.per_beta_frames),
        "--target-update": (args.target_update, agent.target_update_interval),
        "--target-update-tau": (args.target_update_tau, agent.target_update_tau),
        "--hard-update-interval": (
            args.hard_update_interval,
            agent.hard_update_interval,
        ),
        "--disable-double-dqn": (not args.disable_double_dqn, agent.use_double_dqn),
        "--disable-dueling": (not args.disable_dueling, agent.use_dueling),
        "--epsilon-start": (args.epsilon_start, agent.epsilon_start),
        "--epsilon-final": (args.epsilon_final, agent.epsilon_final),
        "--epsilon-decay-steps": (args.epsilon_decay_steps, agent.epsilon_decay_steps),
    }
    conflicts = {
        option: values
        for option, values in checks.items()
        if option in provided and values[0] != values[1]
    }
    if conflicts:
        detail = ", ".join(
            f"{option}=requested:{values[0]!r}/checkpoint:{values[1]!r}"
            for option, values in conflicts.items()
        )
        raise RuntimeError(
            "Explicit agent hyperparameters conflict with the resumed checkpoint: "
            + detail
            + ". Start a fresh run for a new optimizer/agent configuration."
        )


def validate_resume_seed(metadata: dict[str, Any], args: argparse.Namespace) -> None:
    if not metadata or metadata.get("run_seed") is None:
        return
    stored_seed = int(metadata["run_seed"])
    if stored_seed != args.seed and not args.allow_seed_change:
        raise RuntimeError(
            f"Resume run seed changed from {stored_seed} to {args.seed}. "
            "Use --allow-seed-change for an intentional new episode stream."
        )


def validate_resume_environment(
    agent: DQNAgent,
    requested: GameConfig,
    *,
    allow_change: bool,
) -> None:
    if agent.game_config is None:
        return
    stored = asdict(agent.game_config)
    current = asdict(requested)
    mismatches = {
        key: (stored.get(key), current.get(key))
        for key in current
        if stored.get(key) != current.get(key)
    }
    if not mismatches:
        return
    detail = ", ".join(
        f"{key}=checkpoint:{pair[0]!r}/requested:{pair[1]!r}"
        for key, pair in mismatches.items()
    )
    if not allow_change:
        raise RuntimeError(
            "Resume environment/MDP differs from the checkpoint: "
            + detail
            + ". Use --allow-environment-change only for an intentional warm start or curriculum."
        )
    print("Warning: intentionally changing resume environment:", detail)


def validate_resume_best(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    game_config: GameConfig,
    output_path: Path,
) -> tuple[float, int | None]:
    """Validate that a resumed best threshold names a real, comparable artifact."""
    if args.reset_best_evaluation:
        for artifact in (output_path, sidecar_path(output_path)):
            if artifact.exists():
                artifact.unlink()
        return -math.inf, None
    if not metadata or metadata.get("best_eval_score") is None:
        return -math.inf, None

    stored_identity = metadata.get("evaluation_identity")
    current_identity = evaluation_identity(args, game_config)
    if stored_identity != current_identity:
        raise RuntimeError(
            "Resume best evaluation is not comparable with the requested fixed suite/MDP. "
            "Use --reset-best-evaluation for an intentional new selection baseline."
        )

    referenced_path = metadata.get("best_checkpoint_path")
    referenced_sha = metadata.get("best_checkpoint_sha256")
    if not referenced_path or Path(referenced_path).resolve() != output_path.resolve():
        raise RuntimeError(
            "Resume metadata does not link its best score to the requested --output artifact. "
            "Use the original best path or --reset-best-evaluation."
        )
    if (
        not output_path.exists()
        or not referenced_sha
        or _sha256(output_path) != referenced_sha
    ):
        raise RuntimeError(
            "The best checkpoint linked by the resume metadata is missing or stale."
        )

    best_meta_path = sidecar_path(output_path)
    if not best_meta_path.exists():
        raise RuntimeError(f"Best checkpoint metadata is missing: {best_meta_path}")
    with best_meta_path.open("r", encoding="utf-8-sig") as stream:
        best_metadata = json.load(stream)
    score = float(metadata["best_eval_score"])
    episode_value = metadata.get("best_eval_episode")
    episode = int(episode_value) if episode_value is not None else None
    if (
        best_metadata.get("checkpoint_role") != "best_eval"
        or best_metadata.get("checkpoint_sha256") != referenced_sha
        or best_metadata.get("evaluation_identity") != current_identity
        or best_metadata.get("best_eval_score") != score
        or best_metadata.get("best_eval_episode") != episode
    ):
        raise RuntimeError(
            "Best checkpoint sidecar conflicts with the resumed best identity."
        )
    return score, episode


def _resume_path(args: argparse.Namespace) -> Path | None:
    if args.fresh:
        return None
    if args.resume_from:
        path = Path(args.resume_from)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    latest = Path(args.latest_output)
    return latest if latest.exists() else None


def _prepare_fresh_outputs(args: argparse.Namespace, resume_path: Path | None) -> None:
    if resume_path is not None:
        return
    outputs = (Path(args.output), Path(args.latest_output))
    artifacts = [
        item for path in outputs for item in (path, sidecar_path(path)) if item.exists()
    ]
    if not artifacts:
        return
    if not args.overwrite_fresh_output:
        rendered = ", ".join(str(path) for path in artifacts)
        raise RuntimeError(
            "Fresh training would mix with existing output artifacts: "
            f"{rendered}. Use distinct paths or --overwrite-fresh-output."
        )
    for artifact in artifacts:
        artifact.unlink()
    print(
        "Removed exact stale outputs for intentional fresh training:",
        ", ".join(map(str, artifacts)),
    )


def _reheat_exploration(agent: DQNAgent, minimum: float) -> None:
    target = min(agent.epsilon_start, max(agent.epsilon_final, float(minimum)))
    if agent.epsilon >= target:
        return
    span = agent.epsilon_start - agent.epsilon_final
    progress = 0.0 if span <= 0 else (agent.epsilon_start - target) / span
    agent.behavior_steps = int(max(0.0, min(1.0, progress)) * agent.epsilon_decay_steps)
    agent.epsilon = target


def train(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    set_global_seed(args.seed)
    torch.use_deterministic_algorithms(not args.allow_nondeterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.allow_nondeterministic
        torch.backends.cudnn.deterministic = not args.allow_nondeterministic

    configured_step_limit = (
        args.max_steps if args.max_steps > 0 else args.width * args.height * 20
    )
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
        idle_growth_per_food=args.idle_growth_per_food,
        max_episode_steps=configured_step_limit,
    )
    train_env = SnakeGameEnv(game_config)
    train_env.reset(seed=deterministic_episode_seed(args.seed, 0))
    step_limit = episode_step_limit(game_config, args.max_steps)

    resume_path = _resume_path(args)
    _prepare_fresh_outputs(args, resume_path)
    output_path = Path(args.output)
    latest_path = Path(args.latest_output)
    start_episode = 1
    best_eval_score = -math.inf
    best_eval_episode: int | None = None
    if resume_path is not None:
        metadata = load_resume_metadata(
            resume_path, ignore_mismatch=args.ignore_resume_metadata
        )
        agent = DQNAgent.load(str(resume_path), device=args.device)
        validate_resume_identity(metadata, agent)
        validate_resume_seed(metadata, args)
        validate_resume_agent_options(args, agent)
        validate_resume_environment(
            agent, game_config, allow_change=args.allow_environment_change
        )
        validate_v3_contract(agent)
        if agent.network_version < 3 and (
            args.latest_output == "models/dqn_snake_v3_latest.pt"
            or args.output == "models/dqn_snake_v3_best.pt"
        ):
            raise RuntimeError(
                "A legacy v1/v2 warm start must use explicit non-v3 --latest-output and "
                "--output paths so it cannot masquerade as a v3 checkpoint."
            )
        agent.game_config = game_config
        agent.configure_amp(False if args.disable_amp else None)
        resume_state = flatten_observation(
            train_env, agent.device, expected_channels=agent.obs_shape[0]
        )
        if tuple(resume_state.shape) != tuple(agent.obs_shape):
            raise RuntimeError(
                f"Resume board/observation shape {tuple(resume_state.shape)} does not match "
                f"checkpoint shape {tuple(agent.obs_shape)}. Start a fresh v3 run for a new board size."
            )
        if metadata:
            start_episode = int(metadata.get("episodes_completed", 0)) + 1
            best_eval_score, best_eval_episode = validate_resume_best(
                metadata,
                args,
                game_config,
                output_path,
            )
        _reheat_exploration(agent, args.resume_epsilon)
        print(
            f"Resuming {resume_path} at episode {start_episode}; replay starts empty and "
            f"epsilon is {agent.epsilon:.3f}."
        )
    else:
        initial_channels = 3 if args.network_version == 1 else None
        initial_state = flatten_observation(
            train_env, device="cpu", expected_channels=initial_channels
        )
        obs_shape = tuple(int(value) for value in initial_state.shape)
        agent = DQNAgent(
            state_dim=int(np.prod(obs_shape)),
            action_dim=len(RelativeAction)
            if args.network_version >= 3
            else len(Action),
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
            epsilon_decay_steps=args.epsilon_decay_steps,
            n_step=args.n_step,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.per_beta_frames,
            device=args.device,
            game_config=game_config,
            obs_shape=obs_shape,
            network_version=args.network_version,
            amp_enabled=False if args.disable_amp else None,
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_log_{int(time.time())}.jsonl"
    eval_seeds = [args.eval_seed_base + index for index in range(args.eval_episodes)]
    rolling_scores: deque[int] = deque(maxlen=100)
    no_improvement_evals = 0
    early_stop_reference_score = best_eval_score
    final_episode = start_episode - 1

    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "record_type": "run_start",
                    "seed": args.seed,
                    "start_episode": start_episode,
                    "resume_path": str(resume_path) if resume_path else None,
                    "network_version": agent.network_version,
                    "action_dim": agent.action_dim,
                    "obs_shape": list(agent.obs_shape),
                    "step_limit": step_limit,
                    "eval_seeds": eval_seeds,
                    "args": vars(args),
                }
            )
            + "\n"
        )

    for episode in range(start_episode, start_episode + args.episodes):
        train_env.reset(seed=deterministic_episode_seed(args.seed, episode))
        state = flatten_observation(
            train_env, agent.device, expected_channels=agent.obs_shape[0]
        )
        total_env_reward = 0.0
        total_shaped_reward = 0.0
        learn_metrics: list[dict[str, float]] = []
        terminal_event = "truncated"
        terminated = False
        truncated = False
        episode_started = time.perf_counter()

        for step_index in range(step_limit):
            previous_potential = (
                state_potential(train_env) if args.reward_shaping_scale > 0 else 0.0
            )
            chosen = agent.select_action(
                state, action_mask=action_mask(agent, train_env)
            )
            _, env_reward, env_done, info = step_agent_action(agent, train_env, chosen)
            next_state = flatten_observation(
                train_env, agent.device, expected_channels=agent.obs_shape[0]
            )
            truncated = bool(info.get("truncated")) or (
                not env_done and step_index + 1 >= step_limit
            )
            replay_done = env_done or truncated
            shaped_reward = env_reward + potential_shaping(
                previous_potential,
                train_env,
                gamma=agent.gamma,
                scale=args.reward_shaping_scale,
                terminal=replay_done,
            )
            agent.remember(
                state,
                chosen,
                shaped_reward,
                next_state,
                replay_done,
                next_action_mask=action_mask(agent, train_env),
            )
            if agent.behavior_steps % args.train_frequency == 0:
                for _ in range(args.gradient_steps):
                    result = agent.learn()
                    if result is not None:
                        learn_metrics.append(result)
            state = next_state
            total_env_reward += env_reward
            total_shaped_reward += shaped_reward
            if env_done:
                terminated = not truncated
                terminal_event = str(info.get("event", "terminated"))
                break

        rolling_scores.append(train_env.score)

        def metric_mean(name: str) -> float | None:
            values = [item[name] for item in learn_metrics if name in item]
            return float(statistics.mean(values)) if values else None

        metrics: dict[str, Any] = {
            "record_type": "episode",
            "episode": episode,
            "reward": total_env_reward,
            "shaped_reward": total_shaped_reward,
            "score": train_env.score,
            "snake_length": len(train_env.snake),
            "steps": train_env.steps,
            "epsilon": agent.epsilon,
            "behavior_steps": agent.behavior_steps,
            "learn_step_counter": agent.learn_step_counter,
            "replay_size": len(agent.replay_buffer),
            "avg_loss": metric_mean("loss"),
            "avg_td_error": metric_mean("td_error"),
            "avg_grad_norm": metric_mean("grad_norm"),
            "avg_q_mean": metric_mean("q_mean"),
            "rolling_score_100": float(statistics.mean(rolling_scores)),
            "terminal_event": terminal_event,
            "terminated": terminated,
            "truncated": truncated,
            "duration_seconds": time.perf_counter() - episode_started,
        }

        should_evaluate = episode % args.eval_interval == 0
        if should_evaluate:
            evaluation = evaluate_agent(agent, game_config, eval_seeds, step_limit)
            for group in ("reward", "score", "steps"):
                for name, value in evaluation[group].items():
                    metrics[f"eval_{group}_{name}"] = value
            metrics["eval_terminal_events"] = evaluation["terminal_events"]
            metrics["eval_truncated_count"] = evaluation["truncated_count"]
            average_score = float(evaluation["score"]["mean"])
            previous_best = best_eval_score
            improved = average_score > previous_best
            significant_improvement = (
                early_stop_reference_score == -math.inf
                or average_score >= early_stop_reference_score + args.early_stop_delta
            )
            if improved:
                best_eval_score = average_score
                best_eval_episode = episode
                save_checkpoint(
                    agent,
                    output_path,
                    episode=episode,
                    run_seed=args.seed,
                    best_eval_score=best_eval_score,
                    best_eval_episode=best_eval_episode,
                    train_args=args,
                    checkpoint_role="best_eval",
                    best_checkpoint_path=output_path,
                )
                print(
                    f"New best fixed-suite score {best_eval_score:.3f} at episode {episode}; "
                    f"saved {output_path}."
                )
            if significant_improvement:
                early_stop_reference_score = max(
                    early_stop_reference_score, average_score
                )
                no_improvement_evals = 0
            else:
                no_improvement_evals += 1
            save_checkpoint(
                agent,
                latest_path,
                episode=episode,
                run_seed=args.seed,
                best_eval_score=None
                if best_eval_score == -math.inf
                else best_eval_score,
                best_eval_episode=best_eval_episode,
                train_args=args,
                checkpoint_role="latest",
                best_checkpoint_path=output_path,
            )

        if episode % args.checkpoint_interval == 0 and not should_evaluate:
            save_checkpoint(
                agent,
                latest_path,
                episode=episode,
                run_seed=args.seed,
                best_eval_score=None
                if best_eval_score == -math.inf
                else best_eval_score,
                best_eval_episode=best_eval_episode,
                train_args=args,
                checkpoint_role="latest",
                best_checkpoint_path=output_path,
            )

        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        final_episode = episode

        if args.render_frequency and episode % args.render_frequency == 0:
            print(train_env.render(to_string=True))
        if episode % 10 == 0 or episode == start_episode:
            print(
                f"Episode {episode:6d} | score={train_env.score:3d} | "
                f"rolling100={metrics['rolling_score_100']:.2f} | steps={train_env.steps:4d} | "
                f"epsilon={agent.epsilon:.3f} | loss={metrics['avg_loss']} | {terminal_event}"
            )
        if (
            args.early_stop_patience > 0
            and no_improvement_evals >= args.early_stop_patience
        ):
            print(
                f"Early stopping at episode {episode}; best fixed-suite score "
                f"{best_eval_score:.3f} at {best_eval_episode}."
            )
            break

    save_checkpoint(
        agent,
        latest_path,
        episode=final_episode,
        run_seed=args.seed,
        best_eval_score=None if best_eval_score == -math.inf else best_eval_score,
        best_eval_episode=best_eval_episode,
        train_args=args,
        checkpoint_role="latest",
        best_checkpoint_path=output_path,
    )
    print(
        f"Training complete. Latest: {latest_path}; "
        f"best: {output_path if best_eval_episode is not None else 'not evaluated in this run'}; "
        f"log: {log_path}."
    )


if __name__ == "__main__":
    train()
