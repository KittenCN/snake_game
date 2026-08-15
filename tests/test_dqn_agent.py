from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dqn_agent import (
    BaselineConvDuelingQNetwork,
    DQNAgent,
    ReplayBuffer,
    SpatialGroupNormDuelingQNetwork,
    flatten_observation,
)
from env import Action, GameConfig, SnakeGameEnv


def make_agent(**overrides: object) -> DQNAgent:
    options: dict[str, object] = {
        "state_dim": 20 * 12 * 12,
        "action_dim": 4,
        "hidden_sizes": (32,),
        "batch_size": 2,
        "min_replay_size": 2,
        "replay_capacity": 32,
        "obs_shape": (20, 12, 12),
        "network_version": 3,
        "device": "cpu",
        "amp_enabled": False,
    }
    options.update(overrides)
    return DQNAgent(**options)


def test_explicit_unavailable_accelerator_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert DQNAgent._resolve_device(None).type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA or ROCm"):
        DQNAgent._resolve_device("cuda")


def test_checkpoint_ignores_stale_saved_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stale-device.pt"
    make_agent().save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["device"] = "cuda"
    torch.save(checkpoint, path)
    resolved_inputs: list[object] = []

    def resolve_to_cpu(device: object) -> torch.device:
        resolved_inputs.append(device)
        return torch.device("cpu")

    monkeypatch.setattr(DQNAgent, "_resolve_device", staticmethod(resolve_to_cpu))
    loaded = DQNAgent.load(str(path))

    assert loaded.device.type == "cpu"
    assert resolved_inputs[0] is None
    assert "cuda" not in resolved_inputs


def test_policy_checkpoint_constructs_fresh_cross_map_agent(tmp_path: Path) -> None:
    source_config = GameConfig(
        width=8,
        height=8,
        max_idle_steps=90,
        idle_limit_floor_steps=64,
        max_episode_steps=40,
    )
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=source_config,
        lr=0.003,
        action_mask_mode="topology_survival_v1",
    )
    source.behavior_steps = 123
    source.learn_step_counter = 17
    source.replay_buffer.push(
        torch.zeros(20, 8, 8),
        0,
        1.0,
        torch.zeros(20, 8, 8),
        False,
        discount=0.99,
    )
    path = tmp_path / "source-8x8.pt"
    source.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target_config = GameConfig(
        width=10,
        height=10,
        max_idle_steps=90,
        idle_limit_floor_steps=100,
        max_episode_steps=60,
    )

    transferred = DQNAgent.from_policy_checkpoint(
        str(path),
        target_game_config=target_config,
        expected_sha256=digest,
        device="cpu",
        agent_options={"lr": 0.0004, "batch_size": 4, "replay_capacity": 64},
    )

    assert transferred.obs_shape == (20, 10, 10)
    assert transferred.state_dim == 20 * 10 * 10
    assert transferred.game_config == target_config
    assert transferred.hidden_sizes == source.hidden_sizes
    assert transferred.network_version == source.network_version
    assert transferred.action_mask_mode == "topology_survival_v1"
    assert transferred.lr == pytest.approx(0.0004)
    assert transferred.optimizer.state_dict()["state"] == {}
    assert len(transferred.replay_buffer) == 0
    assert transferred.behavior_steps == 0
    assert transferred.learn_step_counter == 0
    assert transferred.replay_restored is False
    for key, value in source.policy_net.state_dict().items():
        assert torch.equal(transferred.policy_net.state_dict()[key], value)
        assert torch.equal(transferred.target_net.state_dict()[key], value)
    provenance = transferred.policy_transfer_provenance
    assert provenance is not None
    assert provenance["source_map"] == {"width": 8, "height": 8}
    assert provenance["target_map"] == {"width": 10, "height": 10}
    assert provenance["cross_map"] is True
    assert provenance["sha256_verified"] is True
    assert provenance["optimizer_restored"] is False
    assert provenance["rng_restored"] is False

    transferred_path = tmp_path / "transferred-10x10.pt"
    transferred.save(str(transferred_path))
    resumed = DQNAgent.load(str(transferred_path), device="cpu")
    assert resumed.obs_shape == (20, 10, 10)
    assert resumed.game_config is not None
    assert resumed.game_config.idle_limit_floor_steps == 100
    assert resumed.policy_transfer_provenance == provenance


def test_policy_transfer_changes_finite_horizon_feature_to_timeless_target(
    tmp_path: Path,
) -> None:
    source_config = GameConfig(width=8, height=8, max_episode_steps=40)
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=source_config,
        time_feature_mode="finite_horizon_progress_v1",
    )
    path = tmp_path / "finite-source.pt"
    source.save(str(path))

    transferred = DQNAgent.from_policy_checkpoint(
        str(path),
        target_game_config=GameConfig(width=8, height=8, max_episode_steps=0),
        device="cpu",
        agent_options={
            "truncation_bootstrap_mode": "bootstrap_v1",
            "time_feature_mode": "constant_zero_v1",
        },
    )

    assert transferred.time_feature_mode == "constant_zero_v1"
    assert transferred.truncation_bootstrap_mode == "bootstrap_v1"
    assert transferred.policy_transfer_provenance is not None
    assert (
        transferred.policy_transfer_provenance["source_time_feature_mode"]
        == "finite_horizon_progress_v1"
    )
    assert (
        transferred.policy_transfer_provenance["target_time_feature_mode"]
        == "constant_zero_v1"
    )


def test_legacy_checkpoint_restores_terminal_truncation_contract(tmp_path: Path) -> None:
    config = GameConfig(width=12, height=12, max_episode_steps=100)
    source = make_agent(game_config=config)
    path = tmp_path / "legacy-boundary.pt"
    source.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["checkpoint_schema_version"] = 4
    checkpoint["metadata"]["checkpoint_schema_version"] = 4
    checkpoint["metadata"].pop("truncation_bootstrap_mode")
    checkpoint["metadata"].pop("time_feature_mode")
    torch.save(checkpoint, path)

    restored = DQNAgent.restore_training_checkpoint(str(path), device="cpu")

    assert restored.truncation_bootstrap_mode == "terminal_v1"
    assert restored.time_feature_mode == "finite_horizon_progress_v1"


def test_v3_horizon_channel_matches_versioned_time_feature_identity() -> None:
    unlimited = SnakeGameEnv(GameConfig(width=5, height=5, max_episode_steps=0))
    unlimited.reset(seed=1)
    unlimited.step(Action.RIGHT)
    unlimited_state = flatten_observation(unlimited, "cpu", expected_channels=20)
    assert torch.count_nonzero(unlimited_state[19]).item() == 0

    finite = SnakeGameEnv(GameConfig(width=5, height=5, max_episode_steps=10))
    finite.reset(seed=1)
    finite.step(Action.RIGHT)
    finite_state = flatten_observation(finite, "cpu", expected_channels=20)
    assert torch.allclose(finite_state[19], torch.full((5, 5), 0.1))


def test_policy_checkpoint_sha_and_structure_fail_before_transfer(
    tmp_path: Path,
) -> None:
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=GameConfig(width=8, height=8),
    )
    path = tmp_path / "source.pt"
    source.save(str(path))

    with pytest.raises(RuntimeError, match="expected SHA-256"):
        DQNAgent.from_policy_checkpoint(
            str(path),
            target_game_config=GameConfig(width=10, height=10),
            expected_sha256="0" * 64,
            device="cpu",
        )

    transferred = DQNAgent.from_policy_checkpoint(
        str(path),
        target_game_config=GameConfig(width=10, height=10),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        device="cpu",
    )
    assert transferred.obs_shape == (20, 10, 10)

    with pytest.raises(ValueError, match="64-character hexadecimal"):
        DQNAgent.from_policy_checkpoint(
            str(path),
            target_game_config=GameConfig(width=10, height=10),
            expected_sha256="not-a-digest",
            device="cpu",
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["hidden_sizes"] = (99,)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="incompatible policy checkpoint"):
        DQNAgent.from_policy_checkpoint(
            str(path),
            target_game_config=GameConfig(width=10, height=10),
            device="cpu",
        )


def test_policy_checkpoint_rejects_game_config_obs_shape_conflict(
    tmp_path: Path,
) -> None:
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=GameConfig(width=8, height=8),
    )
    path = tmp_path / "source.pt"
    source.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["game_config"]["width"] = 9
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="game_config dimensions conflict"):
        DQNAgent.from_policy_checkpoint(str(path), device="cpu")


@pytest.mark.parametrize(
    "target",
    [GameConfig(width=2, height=10), GameConfig(width=3, height=10, initial_length=3)],
)
def test_policy_checkpoint_rejects_invalid_target_game_config(
    tmp_path: Path, target: GameConfig
) -> None:
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=GameConfig(width=8, height=8),
    )
    path = tmp_path / "source.pt"
    source.save(str(path))

    with pytest.raises(ValueError):
        DQNAgent.from_policy_checkpoint(
            str(path), target_game_config=target, device="cpu"
        )


def test_policy_checkpoint_rejects_unknown_version_and_cross_map_schema(
    tmp_path: Path,
) -> None:
    source = make_agent(
        state_dim=20 * 8 * 8,
        action_dim=3,
        obs_shape=(20, 8, 8),
        game_config=GameConfig(width=8, height=8),
    )
    path = tmp_path / "source.pt"
    source.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    checkpoint["metadata"]["network_version"] = 99
    torch.save(checkpoint, path)
    with pytest.raises(
        RuntimeError, match="unsupported policy checkpoint network_version"
    ):
        DQNAgent.from_policy_checkpoint(
            str(path), target_game_config=GameConfig(width=10, height=10), device="cpu"
        )

    source.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["metadata"]["action_schema"] = "absolute_actions_v1"
    torch.save(checkpoint, path)
    with pytest.raises(
        RuntimeError, match="spatial-transfer action/observation schema"
    ):
        DQNAgent.from_policy_checkpoint(
            str(path), target_game_config=GameConfig(width=10, height=10), device="cpu"
        )


def test_legacy_policy_factory_supports_same_map_but_rejects_cross_map(
    tmp_path: Path,
) -> None:
    source = make_agent(
        state_dim=17 * 8 * 8,
        action_dim=4,
        obs_shape=(17, 8, 8),
        network_version=2,
        game_config=GameConfig(width=8, height=8),
    )
    path = tmp_path / "legacy-v2.pt"
    source.save(str(path))

    same_map = DQNAgent.from_policy_checkpoint(str(path), device="cpu")
    assert same_map.obs_shape == (17, 8, 8)
    assert same_map.action_dim == 4
    assert same_map.policy_transfer_provenance is not None
    assert same_map.policy_transfer_provenance["cross_map"] is False

    with pytest.raises(RuntimeError, match="not spatial-transfer capable"):
        DQNAgent.from_policy_checkpoint(
            str(path), target_game_config=GameConfig(width=10, height=10), device="cpu"
        )


class FixedQ(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


class CountingFixedQ(FixedQ):
    def __init__(self, values: list[float]) -> None:
        super().__init__(values)
        self.forward_calls = 0

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(states)


def test_v3_is_default_and_repeated_greedy_inference_is_deterministic() -> None:
    agent = make_agent()
    assert isinstance(agent.policy_net, SpatialGroupNormDuelingQNetwork)
    assert not any(
        isinstance(module, (nn.BatchNorm2d, nn.Dropout))
        for module in agent.policy_net.modules()
    )
    state = torch.randn(agent.obs_shape)
    agent.policy_net.train()

    actions = [agent.select_action(state, epsilon_override=0.0) for _ in range(4)]

    assert len(set(actions)) == 1
    assert agent.policy_net.training is True


def test_action_mask_applies_to_exploration_and_greedy_selection() -> None:
    agent = make_agent()
    agent.policy_net = FixedQ([1.0, 20.0, 3.0, 100.0])
    state = torch.zeros(agent.obs_shape)
    mask = [True, False, True, False]

    assert agent.select_action(state, epsilon_override=0.0, action_mask=mask) == 2
    assert {
        agent.select_action(state, epsilon_override=1.0, action_mask=mask)
        for _ in range(30)
    } <= {0, 2}
    with pytest.raises(ValueError, match="at least one legal"):
        agent.select_action(state, epsilon_override=0.0, action_mask=[False] * 4)


def test_greedy_evaluation_does_not_consume_python_rng() -> None:
    agent = make_agent()
    state = torch.zeros(agent.obs_shape)
    random.seed(1234)
    before = random.getstate()

    agent.select_action(state, epsilon_override=0.0)

    assert random.getstate() == before


def test_epsilon_decays_linearly_on_behavior_actions_only() -> None:
    agent = make_agent(epsilon_start=1.0, epsilon_final=0.2, epsilon_decay_steps=4)
    state = torch.zeros(agent.obs_shape)
    one_legal_action = [True, False, False, False]

    agent.select_action(state, epsilon_override=0.0, action_mask=one_legal_action)
    assert agent.epsilon == pytest.approx(1.0)
    for expected in (0.8, 0.6, 0.4, 0.2, 0.2):
        agent.select_action(state, action_mask=one_legal_action)
        assert agent.epsilon == pytest.approx(expected)


def test_select_actions_batches_one_forward_and_counts_each_behavior_step() -> None:
    agent = make_agent(epsilon_start=1.0, epsilon_final=0.0, epsilon_decay_steps=8)
    network = CountingFixedQ([1.0, 4.0, 3.0, 2.0])
    agent.policy_net = network
    states = torch.zeros((4, *agent.obs_shape))
    masks = [[True, False, True, False]] * 4

    actions = agent.select_actions(states, action_masks=masks)

    assert len(actions) == 4
    assert set(actions) <= {0, 2}
    assert network.forward_calls == 1
    assert agent.behavior_steps == 4
    assert agent.epsilon == pytest.approx(0.5)


def test_select_actions_accepts_nhwc_numpy_batch() -> None:
    agent = make_agent()
    agent.policy_net = FixedQ([1.0, 2.0, 3.0, 4.0])
    states = np.zeros((3, 12, 12, 20), dtype=np.float32)

    actions = agent.select_actions(states, epsilon_override=0.0)

    assert actions == [3, 3, 3]


def test_frozen_policy_anchor_drives_teacher_replay_and_behavior_schedule() -> None:
    agent = make_agent(epsilon_start=0.2, epsilon_final=0.0, epsilon_decay_steps=8)
    agent.policy_net = FixedQ([1.0, 4.0, 3.0, 2.0])
    agent.snapshot_policy_anchor()
    assert agent.policy_anchor_net is not None
    with torch.no_grad():
        agent.policy_net.values.copy_(torch.tensor([9.0, 1.0, 2.0, 3.0]))
    states = torch.zeros((4, *agent.obs_shape))

    actions = agent.select_anchor_actions(
        states, action_masks=[[True, True, True, True]] * 4
    )

    assert actions == [1, 1, 1, 1]
    assert agent.behavior_steps == 4
    assert agent.epsilon == pytest.approx(0.1)


def test_anchor_loss_and_teacher_state_survive_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        n_step=1,
        policy_anchor_weight=0.5,
        teacher_replay_steps=23,
    )
    agent.snapshot_policy_anchor()
    assert agent.policy_anchor_net is not None
    anchor_before = {
        key: value.detach().clone()
        for key, value in agent.policy_anchor_net.state_dict().items()
    }
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.add_(0.05)
    state = torch.zeros(agent.obs_shape)
    for action in (0, 1):
        agent.remember(state, action, 1.0, state, True)

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["anchor_loss"] > 0
    assert metrics["loss"] == pytest.approx(
        metrics["td_loss"] + 0.5 * metrics["anchor_loss"], rel=1e-5
    )
    path = tmp_path / "anchored.pt"
    agent.save(str(path))
    loaded = DQNAgent.load(str(path), device="cpu")
    assert loaded.policy_anchor_weight == pytest.approx(0.5)
    assert loaded.teacher_replay_steps == 23
    assert loaded.policy_anchor_net is not None
    for key, expected in anchor_before.items():
        assert torch.equal(loaded.policy_anchor_net.state_dict()[key], expected)


def test_policy_anchor_decays_after_teacher_replay_and_restores_progress(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        policy_anchor_weight=0.5,
        policy_anchor_final_weight=0.1,
        policy_anchor_decay_steps=100,
        teacher_replay_steps=20,
    )
    agent.behavior_steps = 20
    assert agent.effective_policy_anchor_weight == pytest.approx(0.5)
    agent.behavior_steps = 70
    assert agent.effective_policy_anchor_weight == pytest.approx(0.3)
    agent.behavior_steps = 140
    assert agent.effective_policy_anchor_weight == pytest.approx(0.1)

    agent.action_mask_mode = "legal_v1"
    path = tmp_path / "decaying-anchor.pt"
    agent.save(str(path))
    loaded = DQNAgent.load(str(path), device="cpu")

    assert loaded.policy_anchor_final_weight == pytest.approx(0.1)
    assert loaded.policy_anchor_decay_steps == 100
    assert loaded.behavior_steps == 140
    assert loaded.effective_policy_anchor_weight == pytest.approx(0.1)


def test_anchor_weight_fails_closed_without_frozen_teacher() -> None:
    agent = make_agent(n_step=1, policy_anchor_weight=0.5)
    state = torch.zeros(agent.obs_shape)
    for action in (0, 1):
        agent.remember(state, action, 1.0, state, True)

    with pytest.raises(RuntimeError, match="no frozen anchor"):
        agent.learn()


def test_completed_high_score_trajectories_are_promoted_and_stratified(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        n_step=2,
        batch_size=4,
        demonstration_capacity=16,
        demonstration_batch_fraction=0.5,
        elite_demonstration_batch_fraction=0.25,
        demonstration_min_score=4,
        demonstration_min_return=5,
        demonstration_elite_score=6,
        demonstration_elite_return=20,
    )
    state = torch.zeros(agent.obs_shape)

    agent.remember(state, 0, 1.0, state, True)
    rejected = agent.finalize_trajectory(score=3, episode_return=10)
    assert rejected == {"quality_tier": 0.0, "promoted_transitions": 0.0}

    for index in range(3):
        agent.remember(state, index % 4, 1.0, state, index == 2)
    success = agent.finalize_trajectory(score=4, episode_return=10)
    assert success == {"quality_tier": 1.0, "promoted_transitions": 3.0}

    for index in range(2):
        agent.remember(state, index, 1.0, state, index == 1)
    elite = agent.finalize_trajectory(score=6, episode_return=25)
    assert elite == {"quality_tier": 2.0, "promoted_transitions": 2.0}
    assert agent.demonstration_replay is not None
    assert agent.demonstration_replay.demonstration_count == 5
    assert agent.demonstration_replay.elite_demonstration_count == 2

    batch = agent.demonstration_replay.sample_demonstrations(5, elite_count=2)
    assert batch.demonstration_mask.all()
    assert batch.quality_tiers.tolist() == [1, 1, 1, 2, 2]
    assert batch.imitation_mask.sum().item() == 3
    assert batch.trajectory_scores[:3].tolist() == [4.0, 4.0, 4.0]
    assert batch.trajectory_returns[3:].tolist() == [25.0, 25.0]

    checkpoint = tmp_path / "demonstrations.pt"
    agent.save(str(checkpoint))
    loaded = DQNAgent.load(str(checkpoint), device="cpu")
    assert loaded.demonstration_capacity == 16
    assert loaded.demonstration_batch_fraction == pytest.approx(0.5)
    assert loaded.elite_demonstration_batch_fraction == pytest.approx(0.25)
    assert loaded.demonstration_min_score == pytest.approx(4)
    assert loaded.demonstration_elite_return == pytest.approx(20)
    assert loaded.demonstration_replay is not None
    assert len(loaded.demonstration_replay) == 0
    assert loaded.demonstration_trajectories_seen == 2


def test_trajectory_promotion_is_atomic_when_a_source_slot_was_overwritten() -> None:
    source = ReplayBuffer(2, torch.device("cpu"), action_dim=2)
    target = ReplayBuffer(8, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 2, 2))
    tokens = [
        source.push(state, 0, 1.0, state, False),
        source.push(state, 1, 1.0, state, True),
    ]
    source.push(state, 0, 0.0, state, False)

    copied = source.copy_trajectory_to(
        tokens,
        target,
        quality_tier=1,
        trajectory_score=4,
        trajectory_return=10,
    )

    assert copied == 0
    assert len(target) == 0


def test_trajectory_larger_than_demo_capacity_is_rejected_atomically() -> None:
    source = ReplayBuffer(4, torch.device("cpu"), action_dim=2)
    target = ReplayBuffer(2, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 2, 2))
    tokens = [source.push(state, 0, 1.0, state, False) for _ in range(3)]

    copied = source.copy_trajectory_to(
        tokens,
        target,
        quality_tier=1,
        trajectory_score=4,
        trajectory_return=10,
    )

    assert copied == 0
    assert len(target) == 0


def test_demo_replay_rejects_weaker_trajectory_without_partial_overwrite() -> None:
    source = ReplayBuffer(8, torch.device("cpu"), action_dim=2)
    target = ReplayBuffer(2, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 2, 2))
    elite_tokens = [source.push(state, action, 1.0, state, False) for action in (0, 1)]
    assert (
        source.copy_trajectory_to(
            elite_tokens,
            target,
            quality_tier=2,
            trajectory_score=8,
            trajectory_return=30,
        )
        == 2
    )
    before = target.sample_demonstrations(2)

    success_tokens = [
        source.push(state, action, 1.0, state, False) for action in (0, 1)
    ]
    assert (
        source.copy_trajectory_to(
            success_tokens,
            target,
            quality_tier=1,
            trajectory_score=20,
            trajectory_return=100,
        )
        == 0
    )
    after = target.sample_demonstrations(2)

    assert after.quality_tiers.tolist() == [2, 2]
    assert after.trajectory_scores.tolist() == before.trajectory_scores.tolist()
    assert after.trajectory_returns.tolist() == before.trajectory_returns.tolist()


def test_demo_replay_replaces_only_strictly_weaker_quality() -> None:
    source = ReplayBuffer(8, torch.device("cpu"), action_dim=2)
    target = ReplayBuffer(2, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 2, 2))
    low_tokens = [source.push(state, action, 1.0, state, False) for action in (0, 1)]
    source.copy_trajectory_to(
        low_tokens,
        target,
        quality_tier=1,
        trajectory_score=4,
        trajectory_return=10,
    )
    high_token = [source.push(state, 1, 1.0, state, False)]

    assert (
        source.copy_trajectory_to(
            high_token,
            target,
            quality_tier=1,
            trajectory_score=5,
            trajectory_return=9,
        )
        == 1
    )
    batch = target.sample_demonstrations(2)
    assert sorted(batch.trajectory_scores.tolist()) == [4.0, 5.0]


def test_demonstration_excludes_configured_terminal_prefix_from_imitation() -> None:
    agent = make_agent(
        n_step=1,
        demonstration_capacity=8,
        demonstration_min_score=1,
        demonstration_min_return=-10,
        demonstration_terminal_exclusion_steps=2,
    )
    state = torch.zeros(agent.obs_shape)
    for index in range(4):
        agent.remember(state, index % 4, 1.0, state, index == 3)

    result = agent.finalize_trajectory(score=4, episode_return=4)
    assert result["promoted_transitions"] == 4
    assert agent.demonstration_replay is not None
    batch = agent.demonstration_replay.sample_demonstrations(4)
    assert batch.imitation_mask.sum().item() == 2


def test_full_demo_stratum_is_uniform_without_replacement_or_fake_is_weights() -> None:
    replay = ReplayBuffer(4, torch.device("cpu"), action_dim=2, alpha=1.0)
    state = torch.zeros((1, 2, 2))
    for index, priority in enumerate((1.0, 10.0, 100.0, 1000.0)):
        replay.push(
            state,
            index % 2,
            1.0,
            state,
            False,
            priority=priority,
            quality_tier=1,
            trajectory_score=4,
            trajectory_return=10,
            imitation_eligible=True,
        )

    batch = replay.sample_demonstrations(4)

    assert sorted(batch.indices.tolist()) == [0, 1, 2, 3]
    assert batch.weights.tolist() == pytest.approx([1.0] * 4)


@pytest.mark.parametrize("fraction", [0.1, 0.75])
def test_demo_fraction_must_reserve_demo_and_regular_rows(fraction: float) -> None:
    with pytest.raises(ValueError, match="at least one demo row"):
        make_agent(
            batch_size=2,
            demonstration_capacity=8,
            demonstration_batch_fraction=fraction,
        )


def test_demonstration_batch_adds_large_margin_imitation_loss() -> None:
    agent = make_agent(
        n_step=1,
        batch_size=4,
        min_replay_size=4,
        demonstration_capacity=16,
        demonstration_batch_fraction=0.5,
        elite_demonstration_batch_fraction=0.0,
        demonstration_min_score=4,
        demonstration_min_return=0,
        imitation_loss_weight=0.5,
        imitation_margin=0.8,
    )
    state = torch.zeros(agent.obs_shape)
    for action in (0, 1):
        agent.remember(state, action, 1.0, state, action == 1)
    agent.finalize_trajectory(score=4, episode_return=5)
    for action in (2, 3):
        agent.remember(state, action, 0.0, state, True)
        agent.finalize_trajectory(score=0, episode_return=-5)

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["imitation_loss"] > 0
    assert metrics["demonstration_batch_fraction"] == pytest.approx(0.5)
    assert metrics["loss"] == pytest.approx(
        metrics["td_loss"] + 0.5 * metrics["imitation_loss"], rel=1e-5
    )


def test_terminal_failure_action_is_excluded_from_imitation_loss() -> None:
    agent = make_agent(
        n_step=1,
        batch_size=2,
        min_replay_size=2,
        demonstration_capacity=8,
        demonstration_batch_fraction=0.5,
        demonstration_min_score=1,
        demonstration_min_return=-10,
        imitation_loss_weight=1.0,
    )
    state = torch.zeros(agent.obs_shape)
    agent.remember(state, 0, -5.0, state, True)
    agent.finalize_trajectory(score=4, episode_return=10)
    agent.remember(state, 1, -5.0, state, True)
    agent.finalize_trajectory(score=0, episode_return=-5)

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["demonstration_batch_fraction"] == pytest.approx(0.5)
    assert metrics["imitation_loss"] == pytest.approx(0.0)


def test_n_step_terminal_flush_emits_all_prefixes() -> None:
    agent = make_agent(gamma=0.5, n_step=3)
    states = [torch.full(agent.obs_shape, float(index)) for index in range(4)]
    for index, reward in enumerate((1.0, 2.0, 3.0)):
        agent.remember(
            states[index],
            0,
            reward,
            states[index + 1],
            done=index == 2,
            next_action_mask=[True, True, False, False],
        )

    assert len(agent.replay_buffer) == 3
    assert len(agent._n_step_buffer) == 0
    assert agent.replay_buffer._rewards is not None
    assert agent.replay_buffer._discounts is not None
    assert agent.replay_buffer._dones is not None
    assert agent.replay_buffer._rewards[:3].tolist() == pytest.approx([2.75, 3.5, 3.0])
    assert agent.replay_buffer._discounts[:3].tolist() == pytest.approx(
        [0.125, 0.25, 0.5]
    )
    assert agent.replay_buffer._dones[:3].tolist() == [1.0, 1.0, 1.0]


def test_n_step_truncation_flushes_without_terminal_bootstrap_or_cross_reset() -> None:
    agent = make_agent(gamma=0.5, n_step=3)
    states = [torch.full(agent.obs_shape, float(index)) for index in range(4)]
    final_mask = [False, True, False, False]
    agent.remember(
        states[0],
        0,
        1.0,
        states[1],
        False,
        next_action_mask=[True, True, False, False],
    )
    agent.remember(
        states[1],
        1,
        2.0,
        states[2],
        False,
        next_action_mask=final_mask,
        episode_end=True,
    )

    assert len(agent._n_step_buffer) == 0
    assert len(agent.replay_buffer) == 2
    assert agent.replay_buffer._rewards is not None
    assert agent.replay_buffer._discounts is not None
    assert agent.replay_buffer._dones is not None
    assert agent.replay_buffer._next_states is not None
    assert agent.replay_buffer._next_action_masks is not None
    assert agent.replay_buffer._rewards[:2].tolist() == pytest.approx([2.0, 2.0])
    assert agent.replay_buffer._discounts[:2].tolist() == pytest.approx([0.25, 0.5])
    assert agent.replay_buffer._dones[:2].tolist() == [0.0, 0.0]
    assert torch.all(agent.replay_buffer._next_states[:2] == states[2].to(torch.float16))
    assert agent.replay_buffer._next_action_masks[:2].tolist() == [final_mask, final_mask]

    agent.remember(states[2], 2, 10.0, states[3], True)
    assert agent.replay_buffer._rewards[2].item() == pytest.approx(10.0)


def test_truncated_transition_bootstraps_in_td_target() -> None:
    agent = make_agent(
        gamma=1.0,
        n_step=1,
        batch_size=1,
        min_replay_size=1,
        target_update_tau=0.0,
        hard_update_interval=100,
    )
    agent.policy_net = FixedQ([0.0, 2.0, 3.0, 100.0])
    agent.target_net = FixedQ([10.0, 20.0, 30.0, 40.0])
    agent.target_net.eval()
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=1e-3)
    state = torch.zeros(agent.obs_shape)
    agent.remember(
        state,
        0,
        0.0,
        state,
        False,
        next_action_mask=[True, False, False, False],
        episode_end=True,
    )

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["td_error"] == pytest.approx(10.0)


def test_terminal_next_action_mask_is_canonical_all_true() -> None:
    agent = make_agent(action_dim=3, n_step=1)
    state = torch.zeros(agent.obs_shape)

    agent.remember(
        state,
        0,
        -1.0,
        state,
        True,
        next_action_mask=[False, False, False],
    )
    batch = agent.replay_buffer.sample(1)

    assert batch.next_action_masks is not None
    assert batch.next_action_masks[0].tolist() == [True, True, True]


def test_parallel_n_step_streams_do_not_mix_transitions() -> None:
    agent = make_agent(gamma=0.5, n_step=2)
    state = torch.zeros(agent.obs_shape)
    mask = [True, True, False, False]

    agent.remember(state, 0, 1.0, state, False, mask, stream_id=0)
    agent.remember(state, 1, 10.0, state, False, mask, stream_id=1)
    agent.remember(state, 0, 2.0, state, True, mask, stream_id=0)
    agent.remember(state, 1, 20.0, state, True, mask, stream_id=1)

    assert len(agent.replay_buffer) == 4
    assert agent.replay_buffer._rewards is not None
    assert agent.replay_buffer._rewards[:4].tolist() == pytest.approx(
        [2.0, 2.0, 20.0, 20.0]
    )
    assert 1 not in agent._n_step_buffers


def test_n_step_queue_snapshots_reusable_encoder_views() -> None:
    agent = make_agent(n_step=2)
    reusable = torch.full(agent.obs_shape, 1.0)
    agent.remember(reusable, 0, 0.0, reusable, False)
    reusable.fill_(9.0)
    agent.remember(reusable, 0, 0.0, reusable, True)

    assert agent.replay_buffer._states is not None
    assert torch.all(agent.replay_buffer._states[0] == 1.0)


def test_prioritized_replay_is_cpu_float16_and_samples_high_priority() -> None:
    replay = ReplayBuffer(8, torch.device("cpu"), action_dim=4, alpha=1.0)
    for index in range(4):
        state = torch.full((3, 4, 4), float(index))
        replay.push(state, index, float(index), state + 1, False, discount=0.9)
    replay.update_priorities([0, 1, 2, 3], [1.0, 1.0, 100.0, 1.0])

    assert replay._states is not None
    assert replay._states.device.type == "cpu"
    assert replay._states.dtype == torch.float16
    draws = [int(replay.sample(1, beta=0.4).indices.item()) for _ in range(200)]
    assert draws.count(2) > 150
    batch = replay.sample(3, beta=0.7)
    assert batch.states.dtype == torch.float32
    assert batch.weights.shape == (3,)
    assert batch.discounts.shape == (3,)
    expected_sum = replay._priorities[:4].pow(replay.alpha).sum()
    assert replay._priority_tree[1] == pytest.approx(expected_sum.item())


def test_pinned_replay_staging_failure_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = ReplayBuffer(8, torch.device("cuda"), action_dim=2, pin_memory=True)
    original_empty = torch.empty

    def fail_pinned_empty(*args: object, **kwargs: object) -> torch.Tensor:
        if kwargs.get("pin_memory"):
            raise RuntimeError("pin allocator unavailable")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_pinned_empty)
    with pytest.warns(RuntimeWarning, match="falling back"):
        staging = replay._staging_tensor("states", torch.zeros((4, 3)), 2)

    assert replay.pin_memory is False
    assert staging.device.type == "cpu"
    assert not staging.is_pinned()


def test_priority_tree_tracks_ring_overwrites_without_sampling_empty_slots() -> None:
    replay = ReplayBuffer(3, torch.device("cpu"), action_dim=2, alpha=0.6)
    for index in range(8):
        state = torch.full((3, 2, 2), float(index))
        replay.push(state, index % 2, 0.0, state, False, priority=index + 1)
    expected_sum = replay._priorities[:3].pow(replay.alpha).sum()
    assert replay._priority_tree[1] == pytest.approx(expected_sum.item(), rel=1e-5)
    for _ in range(50):
        assert int(replay.sample(1).indices.item()) < 3


def test_priority_tree_rebuilds_ancestors_without_update_drift() -> None:
    replay = ReplayBuffer(100, torch.device("cpu"), action_dim=3, alpha=0.6)
    state = torch.zeros((1, 1, 1))
    for _ in range(33):
        replay.push(state, 0, 0.0, state, False)

    indices = torch.arange(64) % 33
    for step in range(100):
        priorities = torch.linspace(0.01, 10.0, 64).roll(step % 64)
        replay.update_priorities(indices, priorities)

    expected = replay._priorities[:33].to(torch.float64).pow(replay.alpha).sum()
    assert replay._priority_tree.dtype == torch.float64
    assert replay._priority_tree[1] == pytest.approx(expected.item(), abs=1e-12)
    assert replay._priority_tree[1] == pytest.approx(
        (replay._priority_tree[2] + replay._priority_tree[3]).item(), abs=1e-12
    )


def test_priority_sampling_clamps_mass_below_padded_tree_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = ReplayBuffer(127, torch.device("cpu"), action_dim=3, alpha=0.6)
    state = torch.zeros((1, 1, 1))
    for _ in range(127):
        replay.push(state, 0, 0.0, state, False)
    replay.update_priorities(torch.arange(127), torch.linspace(0.01, 20.0, 127))

    def upper_boundary_rand(
        *size: int, dtype: torch.dtype, **_: object
    ) -> torch.Tensor:
        return torch.ones(size, dtype=dtype)

    monkeypatch.setattr(torch, "rand", upper_boundary_rand)
    batch = replay.sample(32)
    assert bool((batch.indices < len(replay)).all())
    assert torch.isfinite(batch.weights).all()


def test_priority_batch_uses_largest_absolute_duplicate_and_allows_empty() -> None:
    replay = ReplayBuffer(4, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 1, 1))
    replay.push(state, 0, 0.0, state, False)
    replay.update_priorities([0, 0], [-10.0, 2.0])
    assert replay._priorities[0] == pytest.approx(10.0 + replay.priority_epsilon)
    root_before = replay._priority_tree[1].item()
    replay.update_priorities([], [])
    assert replay._priority_tree[1].item() == root_before


def test_priority_tree_rejects_non_finite_updates() -> None:
    replay = ReplayBuffer(4, torch.device("cpu"), action_dim=2)
    state = torch.zeros((1, 1, 1))
    replay.push(state, 0, 0.0, state, False)
    with pytest.raises(ValueError, match="finite"):
        replay.update_priorities([0], [float("nan")])


def test_double_dqn_target_masks_illegal_actions() -> None:
    agent = make_agent(
        gamma=1.0,
        n_step=1,
        batch_size=1,
        min_replay_size=1,
        target_update_tau=0.0,
        hard_update_interval=100,
    )
    agent.policy_net = FixedQ([0.0, 2.0, 3.0, 100.0])
    agent.target_net = FixedQ([10.0, 20.0, 30.0, 40.0])
    agent.target_net.eval()
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=1e-3)
    state = torch.zeros(agent.obs_shape)
    agent.remember(
        state,
        0,
        0.0,
        state,
        False,
        next_action_mask=[True, False, False, False],
    )

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["td_error"] == pytest.approx(10.0)
    assert set(metrics) >= {
        "loss",
        "td_error",
        "grad_norm",
        "q_mean",
        "sampling_seconds",
        "gpu_wait_seconds",
        "pin_memory",
    }
    assert metrics["sampling_seconds"] >= 0.0
    assert metrics["gpu_wait_seconds"] == 0.0


def test_soft_target_update_synchronizes_batchnorm_buffers() -> None:
    agent = make_agent(
        state_dim=17 * 12 * 12,
        obs_shape=(17, 12, 12),
        network_version=2,
        target_update_tau=0.5,
    )
    for name, buffer in agent.policy_net.named_buffers():
        if buffer.is_floating_point():
            buffer.fill_(4.0 if name.endswith("running_var") else 2.0)
        else:
            buffer.fill_(7)
    for buffer in agent.target_net.buffers():
        buffer.zero_()

    agent._update_target_network()

    target_buffers = dict(agent.target_net.named_buffers())
    for name, policy_buffer in agent.policy_net.named_buffers():
        target_buffer = target_buffers[name]
        if policy_buffer.is_floating_point():
            expected = 2.0 if name.endswith("running_var") else 1.0
            assert torch.allclose(
                target_buffer, torch.full_like(target_buffer, expected)
            )
        else:
            assert torch.equal(target_buffer, policy_buffer)


class FakeSnakeEnv:
    def __init__(self, snake: list[tuple[int, int]]) -> None:
        self.snake = snake
        self.food = (5, 5)
        self.direction = Action.RIGHT
        self.steps_since_food = 0
        self.config = SimpleNamespace(max_idle_steps=90, allow_wrap=False)

    def as_numpy(self) -> np.ndarray:
        grid = np.zeros((6, 6, 3), dtype=np.float32)
        for x, y in self.snake:
            grid[y, x, 0] = 1.0
        fx, fy = self.food
        grid[fy, fx, 1] = 1.0
        hx, hy = self.snake[0]
        grid[hy, hx, 2] = 1.0
        return grid


def test_v3_body_order_and_tail_channels_disambiguate_same_occupancy() -> None:
    first = FakeSnakeEnv([(3, 3), (2, 3), (2, 2), (3, 2)])
    second = FakeSnakeEnv([(3, 3), (3, 2), (2, 2), (2, 3)])

    first_obs = flatten_observation(first, "cpu")
    second_obs = flatten_observation(second, "cpu")

    assert first_obs.shape[0] == 20
    assert torch.equal(first_obs[0], second_obs[0])
    assert not torch.equal(first_obs[17], second_obs[17])
    assert not torch.equal(first_obs[18], second_obs[18])
    assert flatten_observation(first, "cpu", expected_channels=17).shape[0] == 17


def test_v3_horizon_channel_makes_time_limit_state_markov() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6, max_episode_steps=10, seed=3))
    env.reset()
    initial = flatten_observation(env, "cpu")
    env.step(Action.RIGHT)
    after_one_step = flatten_observation(env, "cpu")

    assert initial.shape[0] == 20
    assert torch.all(initial[19] == 0.0)
    assert torch.allclose(after_one_step[19], torch.full_like(after_one_step[19], 0.1))


def test_legacy_v1_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
    legacy = make_agent(
        state_dim=3 * 12 * 12,
        obs_shape=(3, 12, 12),
        network_version=1,
        n_step=1,
        epsilon_final=0.01,
    )
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "policy_state_dict": legacy.policy_net.state_dict(),
            "target_state_dict": legacy.target_net.state_dict(),
            "optimizer_state_dict": legacy.optimizer.state_dict(),
            "metadata": {
                "state_dim": legacy.state_dim,
                "action_dim": legacy.action_dim,
                "hidden_sizes": legacy.hidden_sizes,
                "obs_shape": legacy.obs_shape,
                "network_version": 1,
                "epsilon": 0.25,
                "epsilon_final": 0.01,
                "device": "cpu",
                "game_config": {
                    "width": 12,
                    "height": 12,
                    "initial_length": 3,
                    "reward_step": -0.003,
                    "reward_food": 5.0,
                    "reward_death": -2.0,
                    "allow_wrap": False,
                    "seed": None,
                    "max_idle_steps": 90,
                    "idle_penalty": -5.0,
                },
            },
        },
        path,
    )

    loaded = DQNAgent.load(str(path), device="cpu")

    assert isinstance(loaded.policy_net, BaselineConvDuelingQNetwork)
    assert loaded.network_version == 1
    assert loaded.epsilon == pytest.approx(0.25)
    assert loaded.n_step == 1
    assert loaded.replay_restored is False
    assert loaded.game_config is not None
    assert loaded.game_config.idle_growth_per_food == 0
    assert loaded.game_config.max_episode_steps == 0


def test_checkpoint_is_atomic_and_restores_exploration_state(tmp_path: Path) -> None:
    agent = make_agent(epsilon_decay_steps=10)
    state = torch.zeros(agent.obs_shape)
    for _ in range(3):
        agent.select_action(state, action_mask=[True, False, False, False])
    path = tmp_path / "agent.pt"

    agent.save(str(path))
    loaded = DQNAgent.load(str(path), device="cpu")

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert loaded.behavior_steps == 3
    assert loaded.epsilon == pytest.approx(agent.epsilon)
    assert loaded.replay_restored is False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["pin_memory"] is False
    assert loaded.pin_memory is False


def test_policy_warm_start_migrates_weights_only_across_board_sizes(
    tmp_path: Path,
) -> None:
    source = make_agent(action_dim=3, epsilon_start=0.99, lr=1e-3, n_step=2)
    for parameter in source.policy_net.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.optimizer.step()
    with torch.no_grad():
        for index, value in enumerate(source.policy_net.state_dict().values()):
            if value.is_floating_point():
                value.fill_((index + 1) / 100.0)
    source.epsilon = 0.02
    source.behavior_steps = 123
    source.learn_step_counter = 45
    path = tmp_path / "warm-source.pt"
    source.save(str(path))

    target = make_agent(
        state_dim=20 * 18 * 16,
        action_dim=3,
        obs_shape=(20, 18, 16),
        epsilon_start=0.73,
        lr=7e-4,
        n_step=5,
    )
    observation = torch.zeros(target.obs_shape)
    target.replay_buffer.push(observation, 0, 1.0, observation, False)
    target.remember(observation, 1, 2.0, observation, False, stream_id=9)
    random.seed(993)
    np.random.seed(994)
    torch.manual_seed(995)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    metadata = target.load_policy_weights(str(path))

    source_state = source.policy_net.state_dict()
    target_state = target.policy_net.state_dict()
    synced_target_state = target.target_net.state_dict()
    assert metadata["obs_shape"] == source.obs_shape
    assert target.obs_shape == (20, 18, 16)
    assert source_state.keys() == target_state.keys()
    for key, source_value in source_state.items():
        assert torch.equal(target_state[key], source_value)
        assert torch.equal(synced_target_state[key], source_value)
    assert target.target_net.training is False
    assert target.optimizer.state_dict()["state"] == {}
    assert target.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)
    assert len(target.replay_buffer) == 1
    assert len(target._n_step_buffers[9]) == 1
    assert target.epsilon == pytest.approx(0.73)
    assert target.behavior_steps == 0
    assert target.learn_step_counter == 0
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


@pytest.mark.parametrize(
    ("metadata_key", "incompatible_value", "error_match"),
    [
        ("network_version", 2, "network_version"),
        ("action_dim", 5, "action_dim"),
        ("hidden_sizes", (64,), "hidden_sizes"),
        ("obs_shape", (19, 12, 12), "obs channels"),
    ],
)
def test_policy_warm_start_rejects_incompatible_metadata_without_modification(
    tmp_path: Path,
    metadata_key: str,
    incompatible_value: object,
    error_match: str,
) -> None:
    source = make_agent(action_dim=3)
    source_path = tmp_path / "source.pt"
    source.save(str(source_path))
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    checkpoint["metadata"][metadata_key] = incompatible_value
    incompatible_path = tmp_path / f"incompatible-{metadata_key}.pt"
    torch.save(checkpoint, incompatible_path)
    target = make_agent(action_dim=3)
    before = {
        key: value.clone() for key, value in target.policy_net.state_dict().items()
    }

    with pytest.raises(RuntimeError, match=error_match):
        target.load_policy_weights(str(incompatible_path))

    for key, value in target.policy_net.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("corruption", ["shape", "dtype"])
def test_policy_warm_start_validates_all_tensors_before_modifying_policy(
    tmp_path: Path, corruption: str
) -> None:
    source = make_agent(action_dim=3)
    source_path = tmp_path / "source.pt"
    source.save(str(source_path))
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    keys = sorted(checkpoint["policy_state_dict"])
    checkpoint["policy_state_dict"][keys[0]] = torch.full_like(
        checkpoint["policy_state_dict"][keys[0]], 99
    )
    incompatible_key = keys[-1]
    incompatible_tensor = checkpoint["policy_state_dict"][incompatible_key]
    if corruption == "shape":
        checkpoint["policy_state_dict"][incompatible_key] = (
            incompatible_tensor.flatten()[:1]
        )
    else:
        checkpoint["policy_state_dict"][incompatible_key] = incompatible_tensor.to(
            torch.float64
        )
    incompatible_path = tmp_path / f"incompatible-{corruption}.pt"
    torch.save(checkpoint, incompatible_path)
    target = make_agent(action_dim=3)
    before = {
        key: value.clone() for key, value in target.policy_net.state_dict().items()
    }

    with pytest.raises(RuntimeError, match=corruption):
        target.load_policy_weights(str(incompatible_path))

    for key, value in target.policy_net.state_dict().items():
        assert torch.equal(value, before[key])


def test_policy_warm_start_rejects_source_changed_after_hash_validation(
    tmp_path: Path,
) -> None:
    source = make_agent(action_dim=3)
    source_path = tmp_path / "source.pt"
    source.save(str(source_path))
    target = make_agent(action_dim=3)
    before = {
        key: value.clone() for key, value in target.policy_net.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="SHA-256 authentication"):
        target.load_policy_weights(str(source_path), expected_sha256="0" * 64)

    for key, value in target.policy_net.state_dict().items():
        assert torch.equal(value, before[key])


def test_checkpoint_uses_portable_numpy_rng_state_and_restores_sequence(
    tmp_path: Path,
) -> None:
    agent = make_agent()
    path = tmp_path / "portable_rng.pt"
    np.random.seed(12345)
    agent.save(str(path))
    expected = np.random.random(8)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    serialized_state = checkpoint["rng_state"]["numpy"]["state"]
    assert isinstance(serialized_state, list)
    assert serialized_state
    assert all(type(value) is int for value in serialized_state)

    np.random.seed(999)
    DQNAgent.load(str(path), device="cpu")
    assert np.array_equal(np.random.random(8), expected)
