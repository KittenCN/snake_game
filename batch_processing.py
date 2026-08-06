"""Vectorized observation and topology helpers for parallel Snake environments.

The functions in this module intentionally return CPU data.  Callers can request
pinned storage and then transfer a complete batch to CUDA with
``tensor.to(device, non_blocking=True)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

try:
    from .env import Action, SnakeGameEnv
except ImportError:  # pragma: no cover - direct script compatibility
    from env import Action, SnakeGameEnv


OBSERVATION_CHANNELS = 20


def _validate_environments(envs: Sequence[SnakeGameEnv]) -> tuple[int, int, int]:
    if not envs:
        raise ValueError("at least one environment is required")
    height = envs[0].config.height
    width = envs[0].config.width
    for index, env in enumerate(envs):
        if (env.config.height, env.config.width) != (height, width):
            raise ValueError(
                "all environments must have the same board dimensions; "
                f"environment {index} is {env.config.width}x{env.config.height}, "
                f"expected {width}x{height}"
            )
        if not env.snake:
            raise ValueError(
                f"environment {index} has an empty snake; call reset() first"
            )
    return len(envs), height, width


def _cpu_output(
    shape: tuple[int, ...], *, pin_memory: bool
) -> tuple[torch.Tensor, np.ndarray]:
    # A CPU-only PyTorch build rejects pin_memory allocations.  Falling back to
    # ordinary CPU storage keeps this API usable in tests and CPU inference.
    use_pinned = bool(pin_memory and torch.cuda.is_available())
    try:
        tensor = torch.empty(shape, dtype=torch.float32, pin_memory=use_pinned)
    except RuntimeError:
        # Some CUDA-capable installations do not have a functioning pinned
        # allocator (for example, a CPU-only worker sharing CUDA metadata).
        tensor = torch.empty(shape, dtype=torch.float32)
    tensor.zero_()
    return tensor, tensor.numpy()


def encode_observation_batch(
    envs: Sequence[SnakeGameEnv],
    *,
    expected_channels: int | None = None,
    pin_memory: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode same-sized environments as one ``(N, C, H, W)`` CPU tensor.

    The 20-channel result is semantically identical to calling
    :func:`dqn_agent.flatten_observation` for each environment.  Grid-wide
    channels are populated with NumPy broadcasting and advanced indexing;
    Python iteration is limited to collecting each snake's variable-length
    segment list and its four action-safety flags.
    """

    batch_size, height, width = _validate_environments(envs)
    channels = OBSERVATION_CHANNELS if expected_channels is None else expected_channels
    if channels <= 0:
        raise ValueError("expected_channels must be positive")

    required_shape = (batch_size, channels, height, width)
    if out is None:
        output, array = _cpu_output(required_shape, pin_memory=pin_memory)
    else:
        if (
            out.device.type != "cpu"
            or out.dtype != torch.float32
            or tuple(out.shape) != required_shape
        ):
            raise ValueError(
                "out must be a CPU float32 tensor with shape "
                f"{required_shape}; got {out.device}/{out.dtype}/{tuple(out.shape)}"
            )
        output = out
        output.zero_()
        array = output.numpy()
    encoded = array[:, : min(channels, OBSERVATION_CHANNELS)]

    # Stable v3 prefix: board, coordinates, food displacement, direction.
    if channels > 3:
        encoded[:, 3] = torch.linspace(-1.0, 1.0, width, dtype=torch.float32).numpy()[
            None, :
        ]
    if channels > 4:
        encoded[:, 4] = torch.linspace(-1.0, 1.0, height, dtype=torch.float32).numpy()[
            :, None
        ]

    head_x = np.empty(batch_size, dtype=np.intp)
    head_y = np.empty(batch_size, dtype=np.intp)
    food_x = np.empty(batch_size, dtype=np.intp)
    food_y = np.empty(batch_size, dtype=np.intp)
    lengths = np.empty(batch_size, dtype=np.float32)
    idle_progress = np.zeros(batch_size, dtype=np.float32)
    horizon_progress = np.zeros(batch_size, dtype=np.float32)
    directions = np.empty(batch_size, dtype=np.intp)
    safe = np.empty((batch_size, len(Action)), dtype=np.float32)

    for batch_index, env in enumerate(envs):
        snake = env.snake
        segment_count = len(snake)
        xs = np.fromiter(
            (cell[0] for cell in snake), dtype=np.intp, count=segment_count
        )
        ys = np.fromiter(
            (cell[1] for cell in snake), dtype=np.intp, count=segment_count
        )
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys = xs[valid], ys[valid]

        if channels > 0:
            encoded[batch_index, 0, ys, xs] = 1.0
        hx, hy = snake[0]
        head_x[batch_index], head_y[batch_index] = hx, hy
        if channels > 2 and 0 <= hx < width and 0 <= hy < height:
            encoded[batch_index, 2, hy, hx] = 1.0

        if env.food is None:
            fx, fy = hx, hy
        else:
            fx, fy = env.food
            if channels > 1:
                encoded[batch_index, 1, fy, fx] = 1.0
        food_x[batch_index], food_y[batch_index] = fx, fy
        directions[batch_index] = int(env.direction)
        lengths[batch_index] = min(1.0, max(0.0, segment_count / (height * width)))

        if env.idle_limit > 0:
            idle_progress[batch_index] = min(1.0, env.steps_since_food / env.idle_limit)
        if env.config.max_episode_steps > 0:
            horizon_progress[batch_index] = min(
                1.0, env.steps / env.config.max_episode_steps
            )
        safe[batch_index] = [float(env.is_safe_action(action)) for action in Action]

        if channels > 17:
            tx, ty = snake[-1]
            if 0 <= tx < width and 0 <= ty < height:
                encoded[batch_index, 17, ty, tx] = 1.0
        if channels > 18 and segment_count:
            order = (np.arange(1, segment_count + 1, dtype=np.float32) / segment_count)[
                valid
            ]
            encoded[batch_index, 18, ys, xs] = order

    if channels > 5:
        encoded[:, 5] = ((food_x - head_x) / max(1, width - 1))[:, None, None]
    if channels > 6:
        encoded[:, 6] = ((food_y - head_y) / max(1, height - 1))[:, None, None]
    if channels > 7:
        direction_end = min(channels, 11)
        for channel in range(7, direction_end):
            encoded[:, channel] = (directions == channel - 7)[:, None, None]
    if channels > 11:
        encoded[:, 11] = lengths[:, None, None]
    if channels > 12:
        encoded[:, 12] = idle_progress[:, None, None]
    if channels > 13:
        danger_end = min(channels, 17)
        encoded[:, 13:danger_end] = safe[:, : danger_end - 13, None, None]
    if channels > 19:
        encoded[:, 19] = horizon_progress[:, None, None]

    return output


class BatchObservationEncoder:
    """Reusable CPU staging buffer for repeated parallel-environment encoding.

    The returned tensor is a view into shared storage and is overwritten by the
    next :meth:`encode` call.  This is safe for the intended training pipeline:
    enqueue its non-blocking CUDA copy before encoding the next environment step.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_batch_size: int,
        *,
        pin_memory: bool = False,
        channels: int = OBSERVATION_CHANNELS,
    ) -> None:
        if width <= 0 or height <= 0 or max_batch_size <= 0 or channels <= 0:
            raise ValueError(
                "dimensions, max_batch_size, and channels must be positive"
            )
        self.width = width
        self.height = height
        self.max_batch_size = max_batch_size
        self.channels = channels
        self._buffer, _ = _cpu_output(
            (max_batch_size, channels, height, width), pin_memory=pin_memory
        )

    @property
    def is_pinned(self) -> bool:
        return self._buffer.is_pinned()

    def encode(self, envs: Sequence[SnakeGameEnv]) -> torch.Tensor:
        if len(envs) > self.max_batch_size:
            raise ValueError(
                f"batch size {len(envs)} exceeds capacity {self.max_batch_size}"
            )
        view = self._buffer[: len(envs)]
        return encode_observation_batch(
            envs,
            expected_channels=self.channels,
            out=view,
        )


def batch_reachable_masks(envs: Sequence[SnakeGameEnv]) -> np.ndarray:
    """Return vectorized head-reachability masks with each tail treated as free."""

    batch_size, height, width = _validate_environments(envs)
    blocked = np.zeros((batch_size, height, width), dtype=np.bool_)
    reachable = np.zeros_like(blocked)
    wraps = np.empty(batch_size, dtype=np.bool_)

    for index, env in enumerate(envs):
        snake = env.snake
        if len(snake) > 2:
            body = snake[1:-1]
            xs = np.fromiter((cell[0] for cell in body), dtype=np.intp, count=len(body))
            ys = np.fromiter((cell[1] for cell in body), dtype=np.intp, count=len(body))
            blocked[index, ys, xs] = True
        hx, hy = snake[0]
        reachable[index, hy, hx] = True
        wraps[index] = env.config.allow_wrap

    wrap_selector = wraps[:, None, None]
    # A flood fill needs at most H*W expansions.  Neighbour construction and
    # membership tests are performed for the entire batch at once.
    for _ in range(height * width):
        wrap_neighbours = (
            np.roll(reachable, 1, axis=1)
            | np.roll(reachable, -1, axis=1)
            | np.roll(reachable, 1, axis=2)
            | np.roll(reachable, -1, axis=2)
        )
        bounded_neighbours = np.zeros_like(reachable)
        bounded_neighbours[:, 1:] |= reachable[:, :-1]
        bounded_neighbours[:, :-1] |= reachable[:, 1:]
        bounded_neighbours[:, :, 1:] |= reachable[:, :, :-1]
        bounded_neighbours[:, :, :-1] |= reachable[:, :, 1:]
        neighbours = np.where(wrap_selector, wrap_neighbours, bounded_neighbours)
        expanded = reachable | (neighbours & ~blocked)
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    return reachable


def batch_state_potentials(envs: Sequence[SnakeGameEnv]) -> np.ndarray:
    """Compute the training potential for a same-sized environment batch."""

    batch_size, height, width = _validate_environments(envs)
    reachable = batch_reachable_masks(envs)
    reachable_count = reachable.sum(axis=(1, 2), dtype=np.int64)
    blocked_count = np.empty(batch_size, dtype=np.int64)
    distance = np.zeros(batch_size, dtype=np.float64)
    tail_reachable = np.zeros(batch_size, dtype=np.float64)
    active = np.empty(batch_size, dtype=np.bool_)

    for index, env in enumerate(envs):
        snake = env.snake
        blocked_count[index] = max(0, len(snake) - 2)
        active[index] = not env.done and env.food is not None
        tx, ty = snake[-1]
        tail_reachable[index] = float(reachable[index, ty, tx])
        if env.food is not None:
            hx, hy = snake[0]
            fx, fy = env.food
            dx, dy = abs(hx - fx), abs(hy - fy)
            if env.config.allow_wrap:
                dx = min(dx, width - dx)
                dy = min(dy, height - dy)
            distance[index] = dx + dy

    max_distance = max(1, width + height - 2)
    food_closeness = 1.0 - np.minimum(1.0, distance / max_distance)
    traversable = np.maximum(1, width * height - blocked_count)
    space_ratio = np.minimum(1.0, reachable_count / traversable)
    potentials = 0.55 * food_closeness + 0.30 * space_ratio + 0.15 * tail_reachable
    return np.where(active, potentials, 0.0).astype(np.float64, copy=False)


__all__ = [
    "BatchObservationEncoder",
    "OBSERVATION_CHANNELS",
    "batch_reachable_masks",
    "batch_state_potentials",
    "encode_observation_batch",
]
