from __future__ import annotations

from collections import deque

import pytest

from env import Action, GameConfig, RelativeAction, SnakeGameEnv, relative_to_absolute


def _install_state(
    env: SnakeGameEnv,
    snake: list[tuple[int, int]],
    *,
    direction: Action,
    food: tuple[int, int] | None,
) -> None:
    env._snake = deque(snake)
    env._occupied = set(snake)
    env._direction = direction
    env._food = food
    env._score = 0
    env._steps = 0
    env._steps_since_food = 0
    env._done = False
    env.validate_state_invariants()


def test_entering_vacating_tail_preserves_occupied_cells() -> None:
    env = SnakeGameEnv(GameConfig(width=4, height=4))
    _install_state(
        env,
        [(2, 1), (2, 2), (1, 2), (1, 1)],
        direction=Action.UP,
        food=(0, 0),
    )

    _, _, done, info = env.step(Action.LEFT)

    assert not done
    assert info["requested_action"] is Action.LEFT
    assert info["executed_action"] is Action.LEFT
    assert env.snake == [(1, 1), (2, 1), (2, 2), (1, 2)]
    assert env._occupied == set(env.snake)
    env.validate_state_invariants()


@pytest.mark.parametrize("seed", range(12))
def test_random_episodes_preserve_state_invariants(seed: int) -> None:
    env = SnakeGameEnv(GameConfig(width=8, height=8, seed=seed, max_idle_steps=60))

    for _episode in range(8):
        env.reset()
        env.validate_state_invariants()
        for _step in range(300):
            if env.done:
                break
            env.step(env.sample_action())
            env.validate_state_invariants()
            assert env.food is None or env.food not in set(env.snake)


def test_food_never_spawns_on_the_snake() -> None:
    env = SnakeGameEnv(GameConfig(width=7, height=7, seed=91))

    for _ in range(200):
        observation = env.reset()
        assert observation["food"] not in set(observation["snake"])


def test_long_valid_initial_body_stays_on_board() -> None:
    env = SnakeGameEnv(GameConfig(width=5, height=5, initial_length=4))

    env.reset(seed=1)

    env.validate_state_invariants()
    assert len(env.snake) == 4
    assert all(0 <= x < 5 and 0 <= y < 5 for x, y in env.snake)


def test_horizontal_initial_length_is_not_limited_by_height() -> None:
    env = SnakeGameEnv(GameConfig(width=8, height=3, initial_length=6))
    env.reset(seed=2)
    env.validate_state_invariants()
    assert len(env.snake) == 6


def test_environment_enforces_episode_horizon_as_truncation() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6, max_episode_steps=1))
    env.reset(seed=3)
    _, _, done, info = env.step(Action.RIGHT)
    assert done is True
    assert env.steps == 1
    assert info["event"] == "time_limit"
    assert info["truncated"] is True


@pytest.mark.parametrize(
    ("direction", "relative", "expected"),
    [
        (Action.UP, RelativeAction.STRAIGHT, Action.UP),
        (Action.UP, RelativeAction.LEFT, Action.LEFT),
        (Action.UP, RelativeAction.RIGHT, Action.RIGHT),
        (Action.RIGHT, RelativeAction.LEFT, Action.UP),
        (Action.RIGHT, RelativeAction.RIGHT, Action.DOWN),
        (Action.DOWN, RelativeAction.LEFT, Action.RIGHT),
        (Action.DOWN, RelativeAction.RIGHT, Action.LEFT),
        (Action.LEFT, RelativeAction.LEFT, Action.DOWN),
        (Action.LEFT, RelativeAction.RIGHT, Action.UP),
    ],
)
def test_relative_to_absolute_mapping(
    direction: Action, relative: RelativeAction, expected: Action
) -> None:
    assert relative_to_absolute(direction, relative) is expected


def test_step_reports_requested_and_executed_absolute_actions() -> None:
    env = SnakeGameEnv(GameConfig(width=7, height=7))
    env.reset()

    _, _, _, info = env.step(Action.LEFT)

    assert info["requested_action"] is Action.LEFT
    assert info["executed_action"] is Action.RIGHT


def test_step_relative_reports_mapping() -> None:
    env = SnakeGameEnv(GameConfig(width=7, height=7))
    env.reset()

    _, _, _, info = env.step_relative(RelativeAction.LEFT)

    assert info["requested_relative_action"] is RelativeAction.LEFT
    assert info["requested_action"] is Action.UP
    assert info["executed_action"] is Action.UP


def test_idle_limit_grows_after_eating() -> None:
    env = SnakeGameEnv(
        GameConfig(
            width=7,
            height=7,
            initial_length=2,
            allow_wrap=True,
            max_idle_steps=2,
            idle_growth_per_food=3,
        )
    )
    env.reset()
    head_x, head_y = env.snake[0]
    env._food = ((head_x + 1) % env.config.width, head_y)

    observation, _, done, info = env.step(Action.RIGHT)
    assert info["event"] == "ate_food"
    assert not done
    assert env.score == 1
    assert env.idle_limit == 5
    assert observation["idle_limit"] == 5

    for _ in range(4):
        _, _, done, _ = env.step(Action.RIGHT)
        assert not done
    _, _, done, info = env.step(Action.RIGHT)
    assert done
    assert info["event"] == "idle_timeout"


def test_zero_idle_base_disables_timeout_even_with_growth() -> None:
    env = SnakeGameEnv(
        GameConfig(width=7, height=7, max_idle_steps=0, idle_growth_per_food=10)
    )
    env.reset()
    env._score = 5
    assert env.idle_limit == 0
    assert env.observation()["idle_limit"] == 0


def test_config_seed_produces_reproducible_but_distinct_episode_sequence() -> None:
    first = SnakeGameEnv(GameConfig(width=9, height=9, seed=12345))
    second = SnakeGameEnv(GameConfig(width=9, height=9, seed=12345))

    first_foods = [first.reset()["food"] for _ in range(10)]
    second_foods = [second.reset()["food"] for _ in range(10)]

    assert first_foods == second_foods
    assert len(set(first_foods)) > 1


def test_explicit_reset_seed_replays_the_seeded_episode() -> None:
    env = SnakeGameEnv(GameConfig(width=9, height=9, seed=1))

    first_food = env.reset(seed=777)["food"]
    env.reset()
    replayed_food = env.reset(seed=777)["food"]

    assert replayed_food == first_food


def test_as_numpy_remains_three_channel_compatible() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=5))
    env.reset()
    assert env.as_numpy().shape == (5, 6, 3)


def test_validate_state_invariants_detects_desynchronization() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6))
    env.reset()
    env._occupied.remove(env.snake[0])

    with pytest.raises(RuntimeError, match="occupied cells do not match"):
        env.validate_state_invariants()
