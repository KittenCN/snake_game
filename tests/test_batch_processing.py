from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from batch_processing import (
    BatchObservationEncoder,
    batch_reachable_masks,
    batch_state_potentials,
    encode_observation_batch,
)
from dqn_agent import flatten_observation
from env import GameConfig, RelativeAction, SnakeGameEnv
from train_dqn import state_potential


def _random_environments(
    *, width: int = 12, height: int = 12, count: int = 16
) -> list[SnakeGameEnv]:
    environments: list[SnakeGameEnv] = []
    for index in range(count):
        env = SnakeGameEnv(
            GameConfig(
                width=width,
                height=height,
                initial_length=min(3, width - 1),
                allow_wrap=index % 2 == 1,
                max_idle_steps=90,
                max_episode_steps=width * height * 20,
            )
        )
        env.reset(seed=10_000 + index)
        rng = random.Random(20_000 + index)
        for _ in range(rng.randrange(0, 180)):
            if env.done:
                break
            env.step_relative(RelativeAction(rng.randrange(3)))
        environments.append(env)
    return environments


@pytest.mark.parametrize("expected_channels", [3, 17, 20, 23])
def test_batched_observations_match_scalar_encoder(expected_channels: int) -> None:
    envs = _random_environments()

    actual = encode_observation_batch(envs, expected_channels=expected_channels)
    expected = torch.stack(
        [
            flatten_observation(env, "cpu", expected_channels=expected_channels)
            for env in envs
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.is_contiguous()
    assert actual.device.type == "cpu"


@pytest.mark.parametrize("width,height", [(3, 3), (6, 9), (12, 12)])
def test_batched_potentials_match_scalar_random_states(width: int, height: int) -> None:
    envs = _random_environments(width=width, height=height, count=24)

    actual = batch_state_potentials(envs)
    expected = np.asarray([state_potential(env) for env in envs])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_done_and_missing_food_have_zero_potential() -> None:
    envs = _random_environments(width=5, height=5, count=2)
    envs[0]._done = True
    envs[1]._food = None

    np.testing.assert_array_equal(batch_state_potentials(envs), [0.0, 0.0])


def test_reachable_masks_include_head_and_vacating_tail() -> None:
    envs = _random_environments(width=7, height=7, count=8)
    reachable = batch_reachable_masks(envs)

    for index, env in enumerate(envs):
        hx, hy = env.snake[0]
        tx, ty = env.snake[-1]
        assert reachable[index, hy, hx]
        assert reachable[index, ty, tx]


def test_pin_memory_is_conditional_on_cuda_availability() -> None:
    result = encode_observation_batch(_random_environments(count=2), pin_memory=True)

    assert result.is_pinned() is torch.cuda.is_available()


def test_reusable_encoder_reuses_storage_and_matches_scalar() -> None:
    envs = _random_environments(count=4)
    encoder = BatchObservationEncoder(12, 12, 8, pin_memory=True)

    first = encoder.encode(envs)
    pointer = first.data_ptr()
    first_snapshot = first.clone()
    second = encoder.encode(envs[:2])

    assert second.data_ptr() == pointer
    torch.testing.assert_close(first_snapshot[:2], second, rtol=0.0, atol=0.0)
    assert encoder.is_pinned is torch.cuda.is_available()


def test_reusable_encoder_rejects_capacity_overflow() -> None:
    encoder = BatchObservationEncoder(12, 12, 1)
    with pytest.raises(ValueError, match="exceeds capacity"):
        encoder.encode(_random_environments(count=2))


def test_rejects_empty_or_mixed_size_batches() -> None:
    with pytest.raises(ValueError, match="at least one"):
        encode_observation_batch([])

    small = SnakeGameEnv(GameConfig(width=5, height=5))
    large = SnakeGameEnv(GameConfig(width=6, height=5))
    small.reset(seed=1)
    large.reset(seed=2)
    with pytest.raises(ValueError, match="same board dimensions"):
        batch_state_potentials([small, large])
