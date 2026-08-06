from __future__ import annotations

import hashlib
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
    load_warm_start_metadata,
    parse_args,
    potential_shaping,
    save_checkpoint,
    state_potential,
    train,
    validate_resume_seed,
    validate_v3_contract,
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_source_checkpoint(path: Path, *, episode: int = 17) -> DQNAgent:
    config = GameConfig(width=5, height=5, max_episode_steps=2)
    agent = make_agent(config)
    agent.behavior_steps = 321
    agent.learn_step_counter = 45
    agent.epsilon = 0.07
    args = parse_args(
        [
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "2",
            "--hidden",
            "16",
            "--output",
            str(path.parent / "unused-best.pt"),
            "--latest-output",
            str(path.parent / "unused-latest.pt"),
        ]
    )
    save_checkpoint(
        agent,
        path,
        episode=episode,
        run_seed=123,
        best_eval_score=None,
        best_eval_episode=None,
        train_args=args,
        checkpoint_role="latest",
        episodes_started=episode,
    )
    return agent


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


def test_warm_start_arguments_require_one_distinct_existing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    common = [
        "--output",
        str(tmp_path / "best.pt"),
        "--latest-output",
        str(tmp_path / "latest.pt"),
    ]
    with pytest.raises(SystemExit):
        parse_args(
            ["--warm-start-from", str(source), "--resume-from", str(source), *common]
        )
    with pytest.raises(SystemExit):
        parse_args(["--warm-start-from", str(source), "--fresh", *common])
    with pytest.raises(SystemExit):
        parse_args(["--warm-start-from", str(tmp_path / "missing.pt"), *common])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--resume-epsilon",
                "0.4",
                *common,
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--output",
                str(source),
                "--latest-output",
                str(tmp_path / "latest.pt"),
            ]
        )
    # source.meta.json must not be usable as an output checkpoint either.
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--output",
                str(source.with_suffix(".meta.json")),
                "--latest-output",
                str(tmp_path / "latest.pt"),
            ]
        )


def test_output_checkpoint_and_sidecar_artifacts_cannot_cross(tmp_path: Path) -> None:
    latest = tmp_path / "collision.pt"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output",
                str(latest.with_suffix(".meta.json")),
                "--latest-output",
                str(latest),
            ]
        )


def test_parallel_collection_defaults_and_validation() -> None:
    args = parse_args([])
    assert args.num_envs == 1
    assert args.rollout_steps == 1
    assert args.updates_per_collection == 0
    with pytest.raises(SystemExit):
        parse_args(["--num-envs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--rollout-steps", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--updates-per-collection", "-1"])


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


def test_warm_start_metadata_is_required_and_authenticated(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=False
    )
    assert verified is True
    assert metadata["checkpoint_role"] == "latest"
    assert actual_sha256 == file_sha256(source)

    source.with_suffix(".meta.json").unlink()
    with pytest.raises(RuntimeError, match="metadata is missing"):
        load_warm_start_metadata(source, ignore_mismatch=False)
    assert load_warm_start_metadata(source, ignore_mismatch=True) == (
        {},
        False,
        file_sha256(source),
    )

    source.with_suffix(".meta.json").write_text(
        json.dumps({"checkpoint_sha256": "0" * 64, "checkpoint_role": "latest"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        load_warm_start_metadata(source, ignore_mismatch=False)
    ignored_metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=True
    )
    assert verified is False
    assert ignored_metadata["checkpoint_role"] == "latest"
    assert actual_sha256 == file_sha256(source)


@pytest.mark.parametrize("invalid_json", ["[]", "null", '"metadata"'])
def test_warm_start_non_object_metadata_obeys_ignore_flag(
    tmp_path: Path, invalid_json: str
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    source.with_suffix(".meta.json").write_text(invalid_json, encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        load_warm_start_metadata(source, ignore_mismatch=False)

    metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=True
    )
    assert metadata == {}
    assert verified is False
    assert actual_sha256 == file_sha256(source)


@pytest.mark.parametrize("initialization_mode", ["fresh", "warm"])
def test_legacy_network_cannot_use_v3_output_identity(
    tmp_path: Path, initialization_mode: str
) -> None:
    options = [
        "--episodes",
        "1",
        "--network-version",
        "1",
        "--device",
        "cpu",
        "--disable-amp",
    ]
    if initialization_mode == "fresh":
        options.append("--fresh")
    else:
        source = tmp_path / "legacy-source.pt"
        source.write_bytes(b"source is not read before the output identity gate")
        options.extend(["--warm-start-from", str(source)])

    with pytest.raises(RuntimeError, match="legacy v1/v2 run"):
        train(parse_args(options))


def test_invalid_warm_start_never_deletes_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    best.write_bytes(b"preserve-best")
    latest.write_bytes(b"preserve-latest")
    best_hash = file_sha256(best)
    latest_hash = file_sha256(latest)

    # The source uses hidden size 16, so this intentional size mismatch must fail
    # before --overwrite-fresh-output is allowed to remove either destination.
    args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "6",
            "--height",
            "7",
            "--max-steps",
            "1",
            "--hidden",
            "32",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--overwrite-fresh-output",
            "--log-dir",
            str(tmp_path / "logs"),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(RuntimeError, match="incompatible policy checkpoint"):
        train(args)
    assert file_sha256(best) == best_hash
    assert file_sha256(latest) == latest_hash


def test_warm_start_is_fresh_cross_map_training_and_resume_keeps_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    source_sidecar = source.with_suffix(".meta.json")
    source_hash = file_sha256(source)
    source_sidecar_hash = file_sha256(source_sidecar)

    best = tmp_path / "new-best.pt"
    latest = tmp_path / "new-latest.pt"
    logs = tmp_path / "logs"
    # An existing latest must not trigger automatic resume when warm start is explicit.
    latest.write_bytes(b"stale destination, not a resumable checkpoint")
    args = parse_args(
        [
            "--episodes",
            "2",
            "--width",
            "6",
            "--height",
            "7",
            "--max-steps",
            "2",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "1",
            "--checkpoint-interval",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "0.003",
            "--min-replay",
            "32",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--overwrite-fresh-output",
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(args)

    assert file_sha256(source) == source_hash
    assert file_sha256(source_sidecar) == source_sidecar_hash
    metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    provenance = metadata["warm_start_provenance"]
    assert metadata["episodes_completed"] == 2
    assert metadata["episodes_started"] == 2
    assert metadata["behavior_steps"] <= 4
    assert metadata["learn_step_counter"] == 0
    assert metadata["effective_agent_config"]["batch_size"] == 2
    assert metadata["effective_agent_config"]["lr"] == [0.003]
    assert metadata["obs_shape"] == [20, 7, 6]
    assert provenance == {
        "source_path": str(source.resolve()),
        "checkpoint_sha256": source_hash,
        "sidecar_role": "latest",
        "sidecar_episode": 17,
        "embedded_network_version": 3,
        "embedded_obs_shape": [20, 5, 5],
        "source_sidecar_verified": True,
    }
    log_path = next(logs.glob("train_log_*.jsonl"))
    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    run_start = next(
        record for record in records if record["record_type"] == "run_start"
    )
    assert run_start["start_episode"] == 1
    assert run_start["resume_path"] is None
    assert run_start["warm_start_provenance"] == provenance

    resume_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "6",
            "--height",
            "7",
            "--max-steps",
            "2",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "1",
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
    resumed = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert resumed["episodes_completed"] == 3
    assert resumed["warm_start_provenance"] == provenance
    assert file_sha256(source) == source_hash
    assert file_sha256(source_sidecar) == source_sidecar_hash


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
    assert latest_meta["episodes_started"] == 4
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


def test_parallel_training_batches_environments_and_logs_throughput(
    tmp_path: Path,
) -> None:
    best = tmp_path / "parallel_best.pt"
    latest = tmp_path / "parallel_latest.pt"
    logs = tmp_path / "parallel_logs"
    args = parse_args(
        [
            "--episodes",
            "5",
            "--num-envs",
            "3",
            "--rollout-steps",
            "2",
            "--updates-per-collection",
            "2",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "3",
            "--eval-interval",
            "5",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "5",
            "--batch-size",
            "2",
            "--min-replay",
            "2",
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

    metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert metadata["episodes_completed"] == 5
    assert metadata["episodes_started"] == 5
    assert metadata["train_args"]["num_envs"] == 3

    log_path = next(logs.glob("train_log_*.jsonl"))
    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    episodes = [record for record in records if record["record_type"] == "episode"]
    collections = [
        record for record in records if record["record_type"] == "collection"
    ]
    assert len(episodes) == 5
    assert {record["seed_index"] for record in episodes} == {1, 2, 3, 4, 5}
    assert any(record["collection_transitions"] > 1 for record in collections)
    assert any(record["collection_updates"] > 0 for record in collections)
    for key in (
        "env_steps_per_second",
        "updates_per_second",
        "sampling_seconds",
        "gpu_wait_seconds",
    ):
        assert all(key in record for record in collections)

    resume_args = parse_args(
        [
            "--episodes",
            "2",
            "--num-envs",
            "2",
            "--rollout-steps",
            "2",
            "--updates-per-collection",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "3",
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
    resumed = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert resumed["episodes_completed"] == 7
    assert resumed["episodes_started"] == 7
