from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dqn_agent import DQNAgent, flatten_observation
from env import Action, GameConfig, RelativeAction, SnakeGameEnv
from play_dqn import absolute_action, run_episode
from train_dqn import (
    _prepare_fresh_outputs,
    deterministic_episode_seed,
    evaluate_agent,
    load_resume_metadata,
    parse_args,
    potential_shaping,
    state_potential,
    train,
    validate_v3_contract,
    validate_resume_seed,
)


def make_agent(config: GameConfig, action_dim: int = 3) -> DQNAgent:
    env = SnakeGameEnv(config)
    env.reset(seed=7)
    state = flatten_observation(env, "cpu")
    return DQNAgent(
        state_dim=state.numel(),
        action_dim=action_dim,
        hidden_sizes=(16,),
        batch_size=4,
        replay_capacity=32,
        min_replay_size=4,
        obs_shape=tuple(state.shape),
        network_version=3,
        device="cpu",
        game_config=config,
    )


def test_episode_seed_is_stable_and_stream_separated() -> None:
    assert deterministic_episode_seed(42, 5) == deterministic_episode_seed(42, 5)
    assert deterministic_episode_seed(42, 5) != deterministic_episode_seed(42, 6)
    assert deterministic_episode_seed(42, 5, 0) != deterministic_episode_seed(42, 5, 1)


def test_state_potential_is_bounded() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6))
    env.reset(seed=2)
    assert 0.0 <= state_potential(env) <= 1.0


def test_terminal_potential_shaping_has_no_bootstrap_value() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6, max_episode_steps=1))
    env.reset(seed=2)
    previous = state_potential(env)
    assert potential_shaping(
        previous, env, gamma=0.99, scale=1.5, terminal=True
    ) == pytest.approx(-1.5 * previous)


def test_output_paths_must_be_distinct() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output", "same.pt", "--latest-output", "same.pt"])


def test_resume_seed_change_requires_explicit_permission() -> None:
    args = parse_args(["--seed", "9"])
    with pytest.raises(RuntimeError, match="seed changed"):
        validate_resume_seed({"run_seed": 8}, args)
    args.allow_seed_change = True
    validate_resume_seed({"run_seed": 8}, args)


def test_fresh_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    best.write_bytes(b"best")
    latest.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    args = parse_args(["--output", str(best), "--latest-output", str(latest)])
    with pytest.raises(RuntimeError, match="existing output artifacts"):
        _prepare_fresh_outputs(args, None)
    args.overwrite_fresh_output = True
    _prepare_fresh_outputs(args, None)
    assert not best.exists()
    assert not latest.with_suffix(".meta.json").exists()


def test_relative_checkpoint_action_decodes_against_heading() -> None:
    config = GameConfig(width=6, height=6)
    env = SnakeGameEnv(config)
    env.reset(seed=1)
    agent = make_agent(config)
    assert env.direction is Action.RIGHT
    assert absolute_action(agent, env, int(RelativeAction.STRAIGHT)) is Action.RIGHT
    assert absolute_action(agent, env, int(RelativeAction.LEFT)) is Action.UP
    assert absolute_action(agent, env, int(RelativeAction.RIGHT)) is Action.DOWN


def test_fixed_seed_evaluation_is_repeatable() -> None:
    config = GameConfig(width=5, height=5, max_idle_steps=20)
    torch.manual_seed(3)
    agent = make_agent(config)
    first = evaluate_agent(agent, config, [10, 11, 12], max_steps=50)
    second = evaluate_agent(agent, config, [10, 11, 12], max_steps=50)
    assert first == second


def test_fixed_evaluation_counts_environment_time_limit() -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=1)
    agent = make_agent(config)
    result = evaluate_agent(agent, config, [1, 2, 3], max_steps=10)
    assert result["truncated_count"] == 3
    assert result["terminal_events"] == {"time_limit": 3}


def test_console_episode_obeys_checkpoint_horizon() -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=1)
    agent = make_agent(config)
    result = run_episode(agent, SnakeGameEnv(config), 0.0, False, False)
    assert result["steps"] == 1


def test_v3_contract_rejects_observation_without_horizon_channel() -> None:
    config = GameConfig(width=5, height=5)
    agent = DQNAgent(
        state_dim=19 * 5 * 5,
        action_dim=3,
        hidden_sizes=(16,),
        obs_shape=(19, 5, 5),
        network_version=3,
        device="cpu",
        game_config=config,
    )
    with pytest.raises(RuntimeError, match="20 observation channels"):
        validate_v3_contract(agent)


def test_resume_metadata_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    config = GameConfig(width=5, height=5)
    agent = make_agent(config)
    checkpoint = tmp_path / "latest.pt"
    agent.save(str(checkpoint))
    sidecar = checkpoint.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"checkpoint_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        load_resume_metadata(checkpoint, ignore_mismatch=False)
    assert load_resume_metadata(checkpoint, ignore_mismatch=True) == {}


def test_short_training_creates_distinct_latest_and_best(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    logs = tmp_path / "logs"
    args = parse_args(
        [
            "--episodes",
            "4",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-interval",
            "2",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "2",
            "--batch-size",
            "4",
            "--lr",
            "0.001",
            "--min-replay",
            "4",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(args)
    assert latest.exists()
    assert best.exists()
    assert latest != best
    latest_meta = json.loads(
        latest.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    best_meta = json.loads(best.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert latest_meta["checkpoint_role"] == "latest"
    assert best_meta["checkpoint_role"] == "best_eval"
    assert latest_meta["best_checkpoint_path"] == str(best.resolve())
    assert latest_meta["best_checkpoint_sha256"] == best_meta["checkpoint_sha256"]
    assert latest_meta["effective_agent_config"]["lr"] == [0.001]
    assert latest_meta["episodes_completed"] == 4
    assert list(logs.glob("train_log_*.jsonl"))

    resume_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "1",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(resume_args)
    resumed_meta = json.loads(
        latest.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert resumed_meta["episodes_completed"] == 5
    assert resumed_meta["epsilon"] >= 0.05
    assert resumed_meta["effective_agent_config"]["lr"] == [0.001]

    conflicting_lr_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-episodes",
            "2",
            "--lr",
            "0.002",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(RuntimeError, match="hyperparameters conflict"):
        train(conflicting_lr_args)

    drift_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "10",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(RuntimeError, match="environment/MDP"):
        train(drift_args)
