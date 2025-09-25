from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Tuple, Union


GridPosition = Tuple[int, int]


class Action(IntEnum):
    """Enumeration of discrete actions for the snake."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def vector(self) -> GridPosition:
        if self is Action.UP:
            return (0, -1)
        if self is Action.RIGHT:
            return (1, 0)
        if self is Action.DOWN:
            return (0, 1)
        return (-1, 0)

    @classmethod
    def all(cls) -> Tuple["Action", ...]:
        return tuple(cls)


@dataclass
class GameConfig:
    """Configuration options for the snake environment."""

    width: int = 20
    height: int = 20
    initial_length: int = 3
    reward_step: float = -0.003
    reward_food: float = 5.0
    reward_death: float = -2.0
    allow_wrap: bool = False
    seed: Optional[int] = None
    max_idle_steps: int = 0
    idle_penalty: float = -1.0

    def validate(self) -> None:
        if self.width <= 2 or self.height <= 2:
            raise ValueError("Grid must be at least 3x3 to allow movement.")
        if not (1 <= self.initial_length < min(self.width, self.height)):
            raise ValueError(
                "initial_length must be at least 1 and smaller than the grid dimensions."
            )
        if self.max_idle_steps < 0:
            raise ValueError("max_idle_steps must be non-negative")


class SnakeGameEnv:
    """Snake environment suitable for both human play and RL agents."""

    _OPPOSITE = {
        Action.UP: Action.DOWN,
        Action.DOWN: Action.UP,
        Action.LEFT: Action.RIGHT,
        Action.RIGHT: Action.LEFT,
    }

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig()
        self.config.validate()
        self._rng = random.Random(self.config.seed)

        self._snake: Deque[GridPosition] = deque()
        self._occupied: set[GridPosition] = set()
        self._direction: Action = Action.RIGHT
        self._food: Optional[GridPosition] = None
        self._score: int = 0
        self._steps: int = 0
        self._done: bool = False
        self._steps_since_food: int = 0

    # ------------------------------------------------------------------
    # Core environment API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, object]:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            self._rng.seed(seed)
        elif self.config.seed is not None:
            self._rng.seed(self.config.seed)

        self._snake.clear()
        self._occupied.clear()
        self._score = 0
        self._steps = 0
        self._done = False
        self._direction = Action.RIGHT
        self._steps_since_food = 0

        start_x = self.config.width // 2
        start_y = self.config.height // 2
        for offset in range(self.config.initial_length):
            segment = (start_x - offset, start_y)
            self._snake.append(segment)
            self._occupied.add(segment)

        self._spawn_food()
        return self._observation()

    def step(self, action: Union[int, Action]) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        """Advance the environment by one step of the given action."""
        if self._done:
            raise RuntimeError("Episode finished. Call reset() before stepping again.")

        action = self._sanitize_action(action)
        if not self._is_opposite(action):
            self._direction = action

        next_head = self._next_head_position(self._direction)
        self._steps += 1

        reward = self.config.reward_step
        info: Dict[str, object] = {}

        if not self.config.allow_wrap and self._is_out_of_bounds(next_head):
            self._done = True
            reward = self.config.reward_death
            info["event"] = "hit_wall"
            return self._observation(), reward, self._done, info

        if self.config.allow_wrap:
            next_head = (
                next_head[0] % self.config.width,
                next_head[1] % self.config.height,
            )

        if next_head in self._occupied and next_head != self._snake[-1]:
            self._done = True
            reward = self.config.reward_death
            info["event"] = "hit_self"
            return self._observation(), reward, self._done, info

        grew = next_head == self._food
        self._snake.appendleft(next_head)
        self._occupied.add(next_head)

        if grew:
            reward = self.config.reward_food
            self._score += 1
            info["event"] = "ate_food"
            self._spawn_food()
            self._steps_since_food = 0
        else:
            tail = self._snake.pop()
            if tail in self._occupied:
                self._occupied.remove(tail)
            else:
                # Rare desync guard: rebuild occupied cells to the current snake body.
                self._occupied = set(self._snake)
            self._steps_since_food += 1

        if self.config.max_idle_steps > 0 and self._steps_since_food >= self.config.max_idle_steps:
            self._done = True
            reward += self.config.idle_penalty
            info["event"] = "idle_timeout"
            return self._observation(), reward, self._done, info

        if self._food is None:
            self._done = True
            info["event"] = "win"

        return self._observation(), reward, self._done, info

    # ------------------------------------------------------------------
    # Accessors and helpers for agents
    # ------------------------------------------------------------------
    @property
    def direction(self) -> Action:
        return self._direction

    @property
    def snake(self) -> List[GridPosition]:
        return list(self._snake)

    @property
    def food(self) -> Optional[GridPosition]:
        return self._food

    @property
    def score(self) -> int:
        return self._score

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def steps_since_food(self) -> int:
        return self._steps_since_food

    @property
    def done(self) -> bool:
        return self._done

    def legal_actions(self) -> Tuple[Action, ...]:
        return tuple(action for action in Action if action != self._OPPOSITE[self._direction])

    def sample_action(self) -> Action:
        return self._rng.choice(self.legal_actions())

    def is_safe_action(self, action: Union[int, Action]) -> bool:
        candidate = self._sanitize_action(action)
        move_direction = self._direction if self._is_opposite(candidate) else candidate
        head_x, head_y = self._snake[0]
        dx, dy = move_direction.vector
        nx = head_x + dx
        ny = head_y + dy
        if self.config.allow_wrap:
            nx %= self.config.width
            ny %= self.config.height
        else:
            if self._is_out_of_bounds((nx, ny)):
                return False
        target = (nx % self.config.width, ny % self.config.height) if self.config.allow_wrap else (nx, ny)
        tail = self._snake[-1] if self._snake else None
        if tail is not None and target == tail and target != self._food:
            return True
        return target not in self._occupied

    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.config.height, self.config.width, 3)

    def as_numpy(self) -> "numpy.ndarray":  # type: ignore[name-defined]
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "numpy is required for as_numpy(). Install it or skip calling this method."
            ) from exc

        grid = np.zeros(self.observation_shape(), dtype=np.float32)

        for x, y in self._occupied:
            grid[y, x, 0] = 1.0

        if self._food is not None:
            fx, fy = self._food
            grid[fy, fx, 1] = 1.0

        head_x, head_y = self._snake[0]
        grid[head_y, head_x, 2] = 1.0

        return grid

    def render(self, *, to_string: bool = False) -> str:
        symbols = {"empty": " .", "snake": " S", "head": " H", "food": " F"}
        rows: List[str] = []
        snake_body = set(self._snake)
        head = self._snake[0]
        for y in range(self.config.height):
            row_cells: List[str] = []
            for x in range(self.config.width):
                pos = (x, y)
                if pos == head:
                    row_cells.append(symbols["head"])
                elif pos == self._food:
                    row_cells.append(symbols["food"])
                elif pos in snake_body:
                    row_cells.append(symbols["snake"])
                else:
                    row_cells.append(symbols["empty"])
            rows.append("".join(row_cells))
        board = "\n".join(rows)
        if not to_string:
            print(board)
        return board

    def observation(self) -> Dict[str, object]:
        return self._observation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _observation(self) -> Dict[str, object]:
        return {
            "snake": list(self._snake),
            "food": self._food,
            "direction": self._direction,
            "score": self._score,
            "steps": self._steps,
            "done": self._done,
            "width": self.config.width,
            "height": self.config.height,
            "steps_since_food": self._steps_since_food,
        }

    def _spawn_food(self) -> None:
        free_spaces = [
            (x, y)
            for x in range(self.config.width)
            for y in range(self.config.height)
            if (x, y) not in self._occupied
        ]
        if not free_spaces:
            self._food = None
            return
        self._food = self._rng.choice(free_spaces)

    def _sanitize_action(self, action: Union[int, Action]) -> Action:
        if isinstance(action, Action):
            return action
        try:
            return Action(action)
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

    def _is_opposite(self, action: Action) -> bool:
        return action == self._OPPOSITE[self._direction]

    def _next_head_position(self, direction: Action) -> GridPosition:
        head_x, head_y = self._snake[0]
        dx, dy = direction.vector
        return head_x + dx, head_y + dy

    def _is_out_of_bounds(self, position: GridPosition) -> bool:
        x, y = position
        return not (0 <= x < self.config.width and 0 <= y < self.config.height)


__all__ = ["Action", "GameConfig", "SnakeGameEnv"]

