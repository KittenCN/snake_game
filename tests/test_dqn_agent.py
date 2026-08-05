from __future__ import annotations

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


class FixedQ(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


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


def test_priority_tree_tracks_ring_overwrites_without_sampling_empty_slots() -> None:
    replay = ReplayBuffer(3, torch.device("cpu"), action_dim=2, alpha=0.6)
    for index in range(8):
        state = torch.full((3, 2, 2), float(index))
        replay.push(state, index % 2, 0.0, state, False, priority=index + 1)
    expected_sum = replay._priorities[:3].pow(replay.alpha).sum()
    assert replay._priority_tree[1] == pytest.approx(expected_sum.item(), rel=1e-5)
    for _ in range(50):
        assert int(replay.sample(1).indices.item()) < 3


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
    assert set(metrics) >= {"loss", "td_error", "grad_norm", "q_mean"}


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
