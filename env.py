from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy


GridPosition = Tuple[int, int]
ProjectedState = Tuple[
    Tuple[GridPosition, ...],
    "Action",
    Optional[GridPosition],
    object,
    bool,
]


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


class RelativeAction(IntEnum):
    """Actions expressed relative to the snake's current heading."""

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


def relative_to_absolute(
    direction: Union[int, Action], relative_action: Union[int, RelativeAction]
) -> Action:
    """Convert a relative action into an absolute board direction."""

    try:
        absolute_direction = (
            direction if isinstance(direction, Action) else Action(direction)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid direction: {direction}") from exc
    try:
        relative = (
            relative_action
            if isinstance(relative_action, RelativeAction)
            else RelativeAction(relative_action)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid relative action: {relative_action}") from exc

    if relative is RelativeAction.STRAIGHT:
        return absolute_direction
    turn = -1 if relative is RelativeAction.LEFT else 1
    return Action((int(absolute_direction) + turn) % len(Action))


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
    idle_growth_per_food: int = 2
    idle_limit_floor_steps: int = 0
    max_episode_steps: int = 0

    def validate(self) -> None:
        if self.width <= 2 or self.height <= 2:
            raise ValueError("Grid must be at least 3x3 to allow movement.")
        if not (1 <= self.initial_length < self.width):
            raise ValueError(
                "initial_length must be at least 1 and smaller than the grid width."
            )
        if self.max_idle_steps < 0:
            raise ValueError("max_idle_steps must be non-negative")
        if self.idle_growth_per_food < 0:
            raise ValueError("idle_growth_per_food must be non-negative")
        if self.idle_limit_floor_steps < 0:
            raise ValueError("idle_limit_floor_steps must be non-negative")
        if self.max_idle_steps == 0 and self.idle_limit_floor_steps > 0:
            raise ValueError(
                "idle_limit_floor_steps requires max_idle_steps to be positive"
            )
        if self.max_episode_steps < 0:
            raise ValueError("max_episode_steps must be non-negative")


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

        self._snake.clear()
        self._occupied.clear()
        self._score = 0
        self._steps = 0
        self._done = False
        self._direction = Action.RIGHT
        self._steps_since_food = 0

        # Shift right only when a long configured body would otherwise start
        # outside the board; ordinary games retain the traditional centre start.
        start_x = max(self.config.width // 2, self.config.initial_length - 1)
        start_y = self.config.height // 2
        for offset in range(self.config.initial_length):
            segment = (start_x - offset, start_y)
            self._snake.append(segment)
            self._occupied.add(segment)

        self._spawn_food()
        self.validate_state_invariants()
        return self._observation()

    def step(
        self, action: Union[int, Action]
    ) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        """Advance the environment by one step of the given action."""
        if self._done:
            raise RuntimeError("Episode finished. Call reset() before stepping again.")

        requested_action = self._sanitize_action(action)
        if not self._is_opposite(requested_action):
            self._direction = requested_action
        executed_action = self._direction

        next_head = self._next_head_position(executed_action)
        self._steps += 1

        reward = self.config.reward_step
        info: Dict[str, object] = {
            "requested_action": requested_action,
            "executed_action": executed_action,
        }

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

        grew = next_head == self._food
        tail = self._snake[-1]
        can_enter_vacating_tail = not grew and next_head == tail
        if next_head in self._occupied and not can_enter_vacating_tail:
            self._done = True
            reward = self.config.reward_death
            info["event"] = "hit_self"
            return self._observation(), reward, self._done, info

        if not grew:
            removed_tail = self._snake.pop()
            self._occupied.remove(removed_tail)
            self._steps_since_food += 1

        self._snake.appendleft(next_head)
        self._occupied.add(next_head)

        if grew:
            reward = self.config.reward_food
            self._score += 1
            info["event"] = "ate_food"
            self._spawn_food()
            self._steps_since_food = 0

        if self.idle_limit > 0 and self._steps_since_food >= self.idle_limit:
            self._done = True
            reward += self.config.idle_penalty
            info["event"] = "idle_timeout"
            return self._observation(), reward, self._done, info

        if self._food is None:
            self._done = True
            info["event"] = "win"
        elif (
            self.config.max_episode_steps > 0
            and self._steps >= self.config.max_episode_steps
        ):
            self._done = True
            info["event"] = "time_limit"
            info["truncated"] = True

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
    def idle_limit(self) -> int:
        dynamic_limit = (
            self.config.max_idle_steps + self._score * self.config.idle_growth_per_food
            if self.config.max_idle_steps > 0
            else 0
        )
        return max(dynamic_limit, self.config.idle_limit_floor_steps)

    def relative_survival_mask(self) -> Tuple[bool, ...]:
        """Return non-fatal relative actions, or all actions when death is forced."""
        mask = tuple(
            self.is_safe_action(relative_to_absolute(self._direction, relative))
            for relative in RelativeAction
        )
        return mask if any(mask) else tuple(True for _ in RelativeAction)

    def relative_topology_survival_mask(self) -> Tuple[bool, ...]:
        """Prefer moves with a safe continuation and a route back to the tail.

        The projection is pure: it predicts growth, food spawning and wrapping from a
        copied RNG state without advancing the live environment.  If every topology
        candidate is rejected, the mask falls back to immediate one-step safety; only
        a position with no immediately safe action falls back to all actions.
        """

        initial: ProjectedState = (
            tuple(self._snake),
            self._direction,
            self._food,
            self._rng.getstate(),
            False,
        )
        one_step = [False] * len(RelativeAction)
        two_step = [False] * len(RelativeAction)
        topology = [False] * len(RelativeAction)
        for relative in RelativeAction:
            absolute = relative_to_absolute(self._direction, relative)
            projected = self._project_state(initial, absolute)
            if projected is None:
                continue
            index = int(relative)
            one_step[index] = True
            if projected[4]:
                two_step[index] = True
                topology[index] = True
                continue
            next_direction = projected[1]
            two_step[index] = any(
                self._project_state(
                    projected, relative_to_absolute(next_direction, next_relative)
                )
                is not None
                for next_relative in RelativeAction
            )
            topology[index] = two_step[index] and self._projected_tail_reachable(
                projected
            )

        if any(topology):
            return tuple(topology)
        if any(two_step):
            return tuple(two_step)
        if any(one_step):
            return tuple(one_step)
        return tuple(True for _ in RelativeAction)

    @property
    def done(self) -> bool:
        return self._done

    def legal_actions(self) -> Tuple[Action, ...]:
        return tuple(
            action for action in Action if action != self._OPPOSITE[self._direction]
        )

    def sample_action(self) -> Action:
        return self._rng.choice(self.legal_actions())

    def step_relative(
        self, action: Union[int, RelativeAction]
    ) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        """Advance one step using a relative straight/left/right action."""

        try:
            relative = (
                action if isinstance(action, RelativeAction) else RelativeAction(action)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid relative action: {action}") from exc
        absolute = relative_to_absolute(self._direction, relative)
        observation, reward, done, info = self.step(absolute)
        info["requested_relative_action"] = relative
        return observation, reward, done, info

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
        target = (
            (nx % self.config.width, ny % self.config.height)
            if self.config.allow_wrap
            else (nx, ny)
        )
        tail = self._snake[-1] if self._snake else None
        if tail is not None and target == tail and target != self._food:
            return True
        return target not in self._occupied

    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.config.height, self.config.width, 3)

    def as_numpy(self) -> "numpy.ndarray":
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

    def validate_state_invariants(self) -> None:
        """Raise ``RuntimeError`` when the internal board state is inconsistent."""

        snake_cells = list(self._snake)
        snake_set = set(snake_cells)
        errors: List[str] = []
        if not snake_cells:
            errors.append("snake is empty")
        if len(snake_cells) != len(snake_set):
            errors.append("snake contains duplicate cells")
        if snake_set != self._occupied:
            errors.append("occupied cells do not match snake cells")
        out_of_bounds = [cell for cell in snake_cells if self._is_out_of_bounds(cell)]
        if out_of_bounds:
            errors.append(f"snake contains out-of-bounds cells: {out_of_bounds}")
        if self._food is not None:
            if self._is_out_of_bounds(self._food):
                errors.append(f"food is out of bounds: {self._food}")
            if self._food in snake_set:
                errors.append(f"food overlaps snake: {self._food}")
        if errors:
            raise RuntimeError("Invalid snake environment state: " + "; ".join(errors))

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
            "idle_limit": self.idle_limit,
            "max_episode_steps": self.config.max_episode_steps,
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

    def _project_state(
        self, state: ProjectedState, action: Union[int, Action]
    ) -> Optional[ProjectedState]:
        """Project one action without mutating the live environment or its RNG."""

        snake, direction, food, rng_state, won = state
        if won or not snake:
            return state
        candidate = action if isinstance(action, Action) else Action(action)
        move_direction = (
            direction if candidate == self._OPPOSITE[direction] else candidate
        )
        head_x, head_y = snake[0]
        dx, dy = move_direction.vector
        target = (head_x + dx, head_y + dy)
        if self.config.allow_wrap:
            target = (target[0] % self.config.width, target[1] % self.config.height)
        elif self._is_out_of_bounds(target):
            return None

        grew = target == food
        tail = snake[-1]
        if target in set(snake) and not (target == tail and not grew):
            return None
        projected_snake = (
            (target, *snake) if grew else (target, *snake[:-1])
        )
        projected_food = food
        projected_rng_state = rng_state
        projected_won = False
        if grew:
            occupied = set(projected_snake)
            free_spaces = [
                (x, y)
                for x in range(self.config.width)
                for y in range(self.config.height)
                if (x, y) not in occupied
            ]
            if not free_spaces:
                projected_food = None
                projected_won = True
            else:
                projected_rng = random.Random()
                projected_rng.setstate(rng_state)
                projected_food = projected_rng.choice(free_spaces)
                projected_rng_state = projected_rng.getstate()
        return (
            tuple(projected_snake),
            move_direction,
            projected_food,
            projected_rng_state,
            projected_won,
        )

    def _projected_tail_reachable(self, state: ProjectedState) -> bool:
        snake, _, _, _, won = state
        if won or len(snake) <= 1:
            return True
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake[1:-1])
        reachable = {head}
        frontier = deque([head])
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in (Action.UP.vector, Action.RIGHT.vector, Action.DOWN.vector, Action.LEFT.vector):
                candidate = (x + dx, y + dy)
                if self.config.allow_wrap:
                    candidate = (
                        candidate[0] % self.config.width,
                        candidate[1] % self.config.height,
                    )
                elif self._is_out_of_bounds(candidate):
                    continue
                if candidate == tail:
                    return True
                if candidate in blocked or candidate in reachable:
                    continue
                reachable.add(candidate)
                frontier.append(candidate)
        return False

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


__all__ = [
    "Action",
    "RelativeAction",
    "relative_to_absolute",
    "GameConfig",
    "SnakeGameEnv",
]
