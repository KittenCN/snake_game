from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from dqn_agent import DQNAgent, flatten_observation
from env import GameConfig
from play_dqn import (
    action_mask,
    build_env_from_metadata,
    load_inference_agent,
    parse_args,
    select_safe_action,
    step_agent_action,
)


def _save_source(path: Path) -> tuple[DQNAgent, str]:
    source = DQNAgent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        hidden_sizes=(16,),
        batch_size=2,
        min_replay_size=2,
        replay_capacity=16,
        obs_shape=(20, 8, 8),
        network_version=3,
        action_mask_mode="one_step_survival_v1",
        game_config=GameConfig(width=8, height=8, max_episode_steps=1280),
        device="cpu",
        amp_enabled=False,
    )
    source.behavior_steps = 99
    source.learn_step_counter = 12
    source.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "checkpoint_role": "best_eval",
                "checkpoint_sha256": digest,
                "episodes_completed": 500,
                "network_version": source.network_version,
                "action_dim": source.action_dim,
                "action_mask_mode": source.action_mask_mode,
                "obs_shape": list(source.obs_shape),
            }
        ),
        encoding="utf-8",
    )
    return source, digest


def test_play_policy_only_transfer_8x8_to_10x10_and_infers(tmp_path: Path) -> None:
    model = tmp_path / "v5-best.pt"
    source, digest = _save_source(model)
    args = parse_args(
        [
            "--model",
            str(model),
            "--width",
            "10",
            "--height",
            "10",
            "--console",
            "--device",
            "cpu",
        ]
    )

    agent, provenance = load_inference_agent(model, args)
    env = build_env_from_metadata(agent, seed=123)
    env.reset(seed=123)
    state = flatten_observation(env, agent.device, expected_channels=agent.obs_shape[0])
    action = select_safe_action(agent, env, state, safety_enabled=True)
    observation, _, _, _ = step_agent_action(agent, env, action)

    assert agent.obs_shape == (20, 10, 10)
    assert agent.action_mask_mode == "one_step_survival_v1"
    assert env.config.width == 10
    assert env.config.height == 10
    assert env.config.max_episode_steps == 2000
    assert observation["steps"] == 1
    assert agent.optimizer.state_dict()["state"] == {}
    assert agent.behavior_steps == 0
    assert agent.learn_step_counter == 0
    assert provenance["checkpoint_sha256"] == digest
    assert provenance["source_sidecar_role"] == "best_eval"
    assert provenance["source_sidecar_verified"] is True
    assert provenance["cross_map"] is True
    for key, value in source.policy_net.state_dict().items():
        assert torch.equal(agent.policy_net.state_dict()[key], value)


def test_play_same_map_still_uses_fresh_policy_only_agent(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    _save_source(model)

    agent, provenance = load_inference_agent(
        model, parse_args(["--model", str(model), "--device", "cpu"])
    )

    assert agent.obs_shape == (20, 8, 8)
    assert agent.game_config is not None
    assert agent.game_config.max_episode_steps == 1280
    assert agent.behavior_steps == 0
    assert agent.learn_step_counter == 0
    assert agent.optimizer.state_dict()["state"] == {}
    assert provenance["cross_map"] is False


def test_play_uses_checkpointed_survival_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "best.pt"
    _save_source(model)
    agent, _ = load_inference_agent(
        model, parse_args(["--model", str(model), "--device", "cpu"])
    )
    env = build_env_from_metadata(agent, seed=123)
    monkeypatch.setattr(env, "relative_survival_mask", lambda: (True, False, False))

    assert action_mask(agent, env) == [True, False, False]


def test_play_dispatches_checkpointed_topology_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "best.pt"
    _save_source(model)
    agent, _ = load_inference_agent(
        model, parse_args(["--model", str(model), "--device", "cpu"])
    )
    agent.action_mask_mode = "topology_survival_v1"
    env = build_env_from_metadata(agent, seed=123)
    monkeypatch.setattr(
        env, "relative_topology_survival_mask", lambda: (False, True, False)
    )

    assert action_mask(agent, env) == [False, True, False]


def test_play_cli_rejects_partial_map_override_and_sidecar_hash_conflict(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--width", "10"])

    model = tmp_path / "best.pt"
    _save_source(model)
    args = parse_args(
        ["--model", str(model), "--model-sha256", "0" * 64, "--device", "cpu"]
    )
    with pytest.raises(RuntimeError, match="conflicts with the checkpoint sidecar"):
        load_inference_agent(model, args)


def test_play_cross_map_requires_best_sidecar_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "latest.pt"
    _save_source(model)
    sidecar = model.with_suffix(".meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["checkpoint_role"] = "latest"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    common = [
        "--model",
        str(model),
        "--width",
        "10",
        "--height",
        "10",
        "--device",
        "cpu",
    ]

    with pytest.raises(
        RuntimeError, match="requires an authenticated best_eval sidecar"
    ):
        load_inference_agent(model, parse_args(common))

    agent, provenance = load_inference_agent(
        model, parse_args([*common, "--allow-non-best-transfer"])
    )
    assert agent.obs_shape == (20, 10, 10)
    assert provenance["cross_map"] is True


def test_play_rejects_sidecar_architecture_claim_conflict(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    _save_source(model)
    sidecar = model.with_suffix(".meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["network_version"] = 2
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sidecar conflicts"):
        load_inference_agent(
            model, parse_args(["--model", str(model), "--device", "cpu"])
        )


def test_play_cross_map_digest_without_best_sidecar_needs_explicit_override(
    tmp_path: Path,
) -> None:
    model = tmp_path / "legacy.pt"
    _, digest = _save_source(model)
    model.with_suffix(".meta.json").unlink()
    common = [
        "--model",
        str(model),
        "--model-sha256",
        digest,
        "--width",
        "10",
        "--height",
        "10",
        "--device",
        "cpu",
    ]

    with pytest.raises(RuntimeError, match="authenticated best_eval sidecar"):
        load_inference_agent(model, parse_args(common))

    agent, provenance = load_inference_agent(
        model, parse_args([*common, "--allow-non-best-transfer"])
    )
    assert agent.obs_shape == (20, 10, 10)
    assert provenance["source_sidecar_verified"] is False
