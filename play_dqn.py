"""Play snake with a trained DQN agent."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target map width for policy-only spatial transfer (requires --height)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target map height for policy-only spatial transfer (requires --width)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Target episode step limit; 0 uses width*height*20 for a map override",
    )
    parser.add_argument(
        "--model-sha256",
        type=str,
        default=None,
        help="Optional expected checkpoint SHA-256 (sidecar SHA is used automatically)",
    )
    parser.add_argument(
        "--allow-non-best-transfer",
        action="store_true",
        help="Allow cross-map play from a sidecar role other than immutable best_eval",
    )
    args = parser.parse_args(argv)
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("--width and --height must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.model_sha256 is not None and (
        len(args.model_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF" for character in args.model_sha256
        )
    ):
        parser.error("--model-sha256 must be a 64-character hexadecimal digest")
    return args


def _sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def load_inference_agent(
    model_path: Path, args: argparse.Namespace
) -> tuple[DQNAgent, dict[str, object]]:
    """Authenticate metadata when available and construct a policy-only agent."""
    expected_sha256 = args.model_sha256.lower() if args.model_sha256 else None
    sidecar = _sidecar_path(model_path)
    sidecar_metadata: dict[str, object] = {}
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Checkpoint sidecar is unreadable: {sidecar}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Checkpoint sidecar must be a JSON object: {sidecar}")
        sidecar_metadata = payload
        sidecar_sha = payload.get("checkpoint_sha256")
        if not isinstance(sidecar_sha, str) or len(sidecar_sha) != 64:
            raise RuntimeError(f"Checkpoint sidecar has no valid SHA-256: {sidecar}")
        if expected_sha256 is not None and sidecar_sha.lower() != expected_sha256:
            raise RuntimeError(
                "--model-sha256 conflicts with the checkpoint sidecar SHA-256"
            )
        expected_sha256 = sidecar_sha.lower()

    agent = DQNAgent.from_policy_checkpoint(
        str(model_path),
        target_width=args.width,
        target_height=args.height,
        target_max_episode_steps=args.max_steps if args.max_steps > 0 else None,
        device=args.device,
        expected_sha256=expected_sha256,
    )
    provenance = dict(agent.policy_transfer_provenance or {})
    DQNAgent.validate_policy_sidecar_identity(sidecar_metadata, provenance)
    if provenance.get("cross_map"):
        if expected_sha256 is None:
            raise RuntimeError(
                "Cross-map inference requires a sidecar SHA-256 or --model-sha256"
            )
        sidecar_role = sidecar_metadata.get("checkpoint_role")
        if sidecar_role != "best_eval" and not args.allow_non_best_transfer:
            raise RuntimeError(
                "Cross-map inference requires an authenticated best_eval sidecar by default; "
                "use --allow-non-best-transfer only for an intentional diagnostic"
            )
    provenance.update(
        {
            "source_sidecar_path": str(sidecar.resolve()) if sidecar.exists() else None,
            "source_sidecar_role": sidecar_metadata.get("checkpoint_role"),
            "source_sidecar_checkpoint_sha256": sidecar_metadata.get(
                "checkpoint_sha256"
            ),
            "source_sidecar_verified": sidecar.exists(),
        }
    )
    agent.policy_transfer_provenance = provenance
    return agent, provenance


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

    agent, transfer_provenance = load_inference_agent(model_path, args)
    agent.policy_net.eval()
    agent.target_net.eval()
    env = build_env_from_metadata(agent, args.seed)

    print(f"Loaded policy from {model_path.resolve()}")
    print(
        "Inference target: "
        f"{env.config.width}x{env.config.height}, "
        f"step_limit={env.config.max_episode_steps}, "
        f"cross_map={transfer_provenance['cross_map']}"
    )

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
