"""DQN agent implementation for the snake game."""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import random
import tempfile
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Deque, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

try:
    from .env import Action, GameConfig
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from env import Action, GameConfig


if TYPE_CHECKING:
    from .env import SnakeGameEnv


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled replay batch already transferred to the learner device."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    discounts: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    next_action_masks: torch.Tensor | None
    demonstration_mask: torch.Tensor
    imitation_mask: torch.Tensor
    quality_tiers: torch.Tensor
    trajectory_scores: torch.Tensor
    trajectory_returns: torch.Tensor


@dataclass(frozen=True)
class PolicyCheckpointSnapshot:
    """Authenticated, single-read policy checkpoint snapshot."""

    checkpoint_bytes: bytes
    checkpoint_sha256: str
    checkpoint: Mapping[str, Any]
    metadata: Mapping[str, Any]
    policy_state_dict: Mapping[str, torch.Tensor]


_NETWORK_SCHEMAS: dict[int, dict[str, Any]] = {
    1: {
        "observation_schema": "snake_grid_v1_3ch",
        "observation_channels": 3,
        "action_schema": "absolute_actions_v1",
        "action_dim": 4,
        "spatial_transfer_capable": False,
    },
    2: {
        "observation_schema": "snake_grid_v2_17ch",
        "observation_channels": 17,
        "action_schema": "absolute_actions_v1",
        "action_dim": 4,
        "spatial_transfer_capable": False,
    },
    3: {
        "observation_schema": "snake_grid_v3_20ch",
        "observation_channels": 20,
        "action_schema": "relative_actions_v1",
        "action_dim": 3,
        "spatial_transfer_capable": True,
    },
}


class ReplayBuffer:
    """Lazy CPU ring buffer with proportional prioritized replay.

    Observations are deliberately stored as float16 on CPU.  Keeping hundreds of
    thousands of individual CUDA tensors both exhausts VRAM and causes allocator
    fragmentation; batches are transferred to the learner device only when sampled.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        *,
        action_dim: int | None = None,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
        pin_memory: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if alpha < 0.0:
            raise ValueError("PER alpha must be non-negative")
        self.capacity = int(capacity)
        self.device = device
        self.action_dim = action_dim
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.pin_memory = bool(pin_memory and device.type == "cuda")
        self._size = 0
        self._position = 0
        self._states: torch.Tensor | None = None
        self._next_states: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._rewards: torch.Tensor | None = None
        self._dones: torch.Tensor | None = None
        self._discounts: torch.Tensor | None = None
        self._next_action_masks: torch.Tensor | None = None
        self._quality_tiers = torch.zeros(
            self.capacity, dtype=torch.uint8, device="cpu"
        )
        self._trajectory_scores = torch.zeros(
            self.capacity, dtype=torch.float32, device="cpu"
        )
        self._trajectory_returns = torch.zeros(
            self.capacity, dtype=torch.float32, device="cpu"
        )
        self._slot_versions = torch.zeros(self.capacity, dtype=torch.long, device="cpu")
        self._imitation_eligible = torch.zeros(
            self.capacity, dtype=torch.bool, device="cpu"
        )
        self._priorities = torch.zeros(self.capacity, dtype=torch.float32, device="cpu")
        self._tree_capacity = 1 << (self.capacity - 1).bit_length()
        # The tree is tiny compared with observation storage. Float64 prevents
        # probability-mass rounding from crossing into padded leaves when the
        # configured capacity is not a power of two.
        self._priority_tree = torch.zeros(
            self._tree_capacity * 2, dtype=torch.float64, device="cpu"
        )
        self._max_priority = 1.0
        self._staging: dict[str, torch.Tensor] = {}
        self._pin_memory_warning_emitted = False

    def __len__(self) -> int:
        return self._size

    @property
    def storage_device(self) -> torch.device:
        return torch.device("cpu")

    def _allocate(
        self, state: torch.Tensor, next_action_mask: torch.Tensor | None
    ) -> None:
        shape = tuple(int(dim) for dim in state.shape)
        self._states = torch.empty(
            (self.capacity, *shape), dtype=torch.float16, device="cpu"
        )
        self._next_states = torch.empty_like(self._states)
        self._actions = torch.empty(self.capacity, dtype=torch.long, device="cpu")
        self._rewards = torch.empty(self.capacity, dtype=torch.float32, device="cpu")
        self._dones = torch.empty(self.capacity, dtype=torch.float32, device="cpu")
        self._discounts = torch.empty(self.capacity, dtype=torch.float32, device="cpu")
        mask_dim = self.action_dim
        if mask_dim is None and next_action_mask is not None:
            mask_dim = int(next_action_mask.numel())
            self.action_dim = mask_dim
        if mask_dim is not None:
            self._next_action_masks = torch.ones(
                (self.capacity, mask_dim), dtype=torch.bool, device="cpu"
            )

    def _staging_tensor(
        self, name: str, source: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Gather into reusable pinned storage so H2D can actually be asynchronous."""
        shape = (batch_size, *source.shape[1:])
        staging = self._staging.get(name)
        if (
            staging is None
            or tuple(staging.shape) != shape
            or staging.dtype != source.dtype
        ):
            try:
                staging = torch.empty(
                    shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
            except RuntimeError as exc:
                if not self._pin_memory_warning_emitted:
                    warnings.warn(
                        f"pinned replay staging allocation failed; falling back to "
                        f"pageable CPU memory: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._pin_memory_warning_emitted = True
                self.pin_memory = False
                self._staging.clear()
                staging = torch.empty(shape, dtype=source.dtype, device="cpu")
            self._staging[name] = staging
        return staging

    def _gather(
        self, name: str, source: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if not self.pin_memory:
            return source[indices]
        staging = self._staging_tensor(name, source, int(indices.numel()))
        torch.index_select(source, 0, indices, out=staging)
        return staging

    @staticmethod
    def _cpu_scalar(
        value: torch.Tensor | int | float | bool, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device="cpu").reshape(())

    def _set_priority(self, index: int, priority: float) -> None:
        """Update one leaf and rebuild its ancestors without cumulative drift."""
        if not math.isfinite(float(priority)):
            raise ValueError("replay priority must be finite")
        raw_priority = max(abs(float(priority)), self.priority_epsilon)
        self._priorities[index] = raw_priority
        raw_priority = float(self._priorities[index])
        self._max_priority = max(self._max_priority, raw_priority)
        scaled_priority = 1.0 if self.alpha == 0.0 else raw_priority**self.alpha
        tree_index = self._tree_capacity + index
        self._priority_tree[tree_index] = scaled_priority
        tree_index //= 2
        while tree_index:
            self._priority_tree[tree_index] = (
                self._priority_tree[tree_index * 2]
                + self._priority_tree[tree_index * 2 + 1]
            )
            tree_index //= 2

    def _set_priority_batch(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        """Update sampled priorities with vectorized work at each tree level."""
        # Sampling is with replacement. Collapse duplicate indices first so a
        # leaf is written once; duplicate transitions have the same TD target,
        # and keeping the largest error is the conservative PER choice.
        if indices.numel() == 0:
            return
        if not bool(torch.isfinite(priorities).all()):
            raise ValueError("replay priorities must be finite")
        priorities = priorities.abs()
        unique_indices, inverse = torch.unique(
            indices, sorted=False, return_inverse=True
        )
        unique_priorities = torch.zeros(
            unique_indices.numel(), dtype=torch.float32, device="cpu"
        )
        unique_priorities.scatter_reduce_(
            0, inverse, priorities, reduce="amax", include_self=False
        )
        unique_priorities = unique_priorities.add(self.priority_epsilon)
        self._priorities[unique_indices] = unique_priorities
        self._max_priority = max(
            self._max_priority, float(unique_priorities.max().item())
        )

        scaled = (
            torch.ones_like(unique_priorities, dtype=torch.float64)
            if self.alpha == 0.0
            else unique_priorities.to(torch.float64).pow(self.alpha)
        )
        tree_indices = self._tree_capacity + unique_indices
        self._priority_tree[tree_indices] = scaled
        while int(tree_indices[0]) > 1:
            parents = torch.unique(tree_indices // 2)
            self._priority_tree[parents] = (
                self._priority_tree[parents * 2] + self._priority_tree[parents * 2 + 1]
            )
            tree_indices = parents

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor | int,
        reward: torch.Tensor | float,
        next_state: torch.Tensor,
        done: torch.Tensor | float | bool,
        *,
        discount: torch.Tensor | float = 1.0,
        next_action_mask: torch.Tensor | Sequence[bool] | None = None,
        priority: float | None = None,
        quality_tier: int = 0,
        trajectory_score: float = 0.0,
        trajectory_return: float = 0.0,
        imitation_eligible: bool = False,
    ) -> tuple[int, int]:
        if quality_tier not in (0, 1, 2):
            raise ValueError(
                "quality_tier must be 0 (regular), 1 (success), or 2 (elite)"
            )
        if not math.isfinite(trajectory_score) or not math.isfinite(trajectory_return):
            raise ValueError("trajectory score and return must be finite")
        state_cpu = state.detach().to(device="cpu", dtype=torch.float16)
        next_state_cpu = next_state.detach().to(device="cpu", dtype=torch.float16)
        mask_cpu = (
            torch.as_tensor(next_action_mask, dtype=torch.bool, device="cpu").flatten()
            if next_action_mask is not None
            else None
        )
        if self._states is None:
            self._allocate(state_cpu, mask_cpu)
        assert self._states is not None and self._next_states is not None
        assert self._actions is not None and self._rewards is not None
        assert self._dones is not None and self._discounts is not None
        if tuple(state_cpu.shape) != tuple(self._states.shape[1:]):
            raise ValueError(
                f"state shape {tuple(state_cpu.shape)} does not match replay shape "
                f"{tuple(self._states.shape[1:])}"
            )
        if tuple(next_state_cpu.shape) != tuple(self._next_states.shape[1:]):
            raise ValueError("next_state shape does not match replay observation shape")

        idx = self._position
        self._states[idx].copy_(state_cpu)
        self._next_states[idx].copy_(next_state_cpu)
        self._actions[idx] = self._cpu_scalar(action, torch.long)
        self._rewards[idx] = self._cpu_scalar(reward, torch.float32)
        self._dones[idx] = self._cpu_scalar(done, torch.float32)
        self._discounts[idx] = self._cpu_scalar(discount, torch.float32)
        self._quality_tiers[idx] = quality_tier
        self._trajectory_scores[idx] = trajectory_score
        self._trajectory_returns[idx] = trajectory_return
        self._slot_versions[idx] += 1
        self._imitation_eligible[idx] = bool(imitation_eligible)
        slot_version = int(self._slot_versions[idx].item())
        if self._next_action_masks is not None:
            if mask_cpu is None:
                self._next_action_masks[idx].fill_(True)
            else:
                if mask_cpu.numel() != self._next_action_masks.shape[1]:
                    raise ValueError(
                        "next_action_mask length does not match action_dim"
                    )
                if not bool(mask_cpu.any()):
                    raise ValueError(
                        "next_action_mask must contain at least one legal action"
                    )
                self._next_action_masks[idx].copy_(mask_cpu)

        if priority is None:
            priority_value = self._max_priority
        else:
            priority_value = abs(float(priority)) + self.priority_epsilon
        self._set_priority(idx, priority_value)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx, slot_version

    @property
    def demonstration_count(self) -> int:
        if self._size == 0:
            return 0
        return int((self._quality_tiers[: self._size] > 0).sum().item())

    @property
    def elite_demonstration_count(self) -> int:
        if self._size == 0:
            return 0
        return int((self._quality_tiers[: self._size] == 2).sum().item())

    def copy_trajectory_to(
        self,
        tokens: Sequence[tuple[int, int]],
        target: "ReplayBuffer",
        *,
        quality_tier: int,
        trajectory_score: float,
        trajectory_return: float,
    ) -> int:
        """Copy a completed trajectory if none of its ring slots were overwritten."""
        if quality_tier not in (1, 2):
            raise ValueError("demonstration trajectories require quality tier 1 or 2")
        if not tokens:
            return 0
        if len(tokens) > target.capacity:
            return 0
        assert self._states is not None and self._next_states is not None
        assert self._actions is not None and self._rewards is not None
        assert self._dones is not None and self._discounts is not None
        valid_indices: list[int] = []
        for index, version in tokens:
            if not 0 <= index < self._size:
                return 0
            if int(self._slot_versions[index].item()) != int(version):
                return 0
            valid_indices.append(index)

        for index in valid_indices:
            next_mask = (
                self._next_action_masks[index]
                if self._next_action_masks is not None
                else None
            )
            target.push(
                self._states[index],
                self._actions[index],
                self._rewards[index],
                self._next_states[index],
                self._dones[index],
                discount=self._discounts[index],
                next_action_mask=next_mask,
                priority=float(self._priorities[index].item()),
                quality_tier=quality_tier,
                trajectory_score=trajectory_score,
                trajectory_return=trajectory_return,
                imitation_eligible=bool(self._imitation_eligible[index].item()),
            )
        return len(valid_indices)

    def _build_batch(
        self, indices_cpu: torch.Tensor, weights: torch.Tensor
    ) -> ReplayBatch:
        assert self._states is not None and self._next_states is not None
        assert self._actions is not None and self._rewards is not None
        assert self._dones is not None and self._discounts is not None
        non_blocking = self.device.type == "cuda" and self.pin_memory
        masks = None
        if self._next_action_masks is not None:
            masks = self._gather(
                "next_action_masks", self._next_action_masks, indices_cpu
            ).to(self.device, non_blocking=non_blocking)
        quality_tiers = self._gather(
            "quality_tiers", self._quality_tiers, indices_cpu
        ).to(self.device, non_blocking=non_blocking)
        return ReplayBatch(
            states=self._gather("states", self._states, indices_cpu).to(
                self.device, dtype=torch.float32, non_blocking=non_blocking
            ),
            actions=self._gather("actions", self._actions, indices_cpu).to(
                self.device, non_blocking=non_blocking
            ),
            rewards=self._gather("rewards", self._rewards, indices_cpu).to(
                self.device, non_blocking=non_blocking
            ),
            next_states=self._gather("next_states", self._next_states, indices_cpu).to(
                self.device, dtype=torch.float32, non_blocking=non_blocking
            ),
            dones=self._gather("dones", self._dones, indices_cpu).to(
                self.device, non_blocking=non_blocking
            ),
            discounts=self._gather("discounts", self._discounts, indices_cpu).to(
                self.device, non_blocking=non_blocking
            ),
            weights=weights.to(
                self.device, dtype=torch.float32, non_blocking=non_blocking
            ),
            indices=indices_cpu,
            next_action_masks=masks,
            demonstration_mask=quality_tiers > 0,
            imitation_mask=(quality_tiers > 0)
            & self._gather(
                "imitation_eligible", self._imitation_eligible, indices_cpu
            ).to(self.device, non_blocking=non_blocking),
            quality_tiers=quality_tiers,
            trajectory_scores=self._gather(
                "trajectory_scores", self._trajectory_scores, indices_cpu
            ).to(self.device, non_blocking=non_blocking),
            trajectory_returns=self._gather(
                "trajectory_returns", self._trajectory_returns, indices_cpu
            ).to(self.device, non_blocking=non_blocking),
        )

    def sample(
        self,
        batch_size: int,
        *,
        beta: float = 0.4,
        normalize_weights: bool = True,
    ) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("not enough replay samples")
        assert self._states is not None and self._next_states is not None
        assert self._actions is not None and self._rewards is not None
        assert self._dones is not None and self._discounts is not None

        total_priority = float(self._priority_tree[1])
        if not math.isfinite(total_priority) or total_priority <= 0.0:
            raise RuntimeError("replay priority sum is invalid")
        # Stratified proportional sampling keeps work O(batch * log capacity),
        # instead of rescanning all 50k+ priorities for every gradient update.
        masses = (
            torch.arange(batch_size, dtype=torch.float64)
            + torch.rand(batch_size, dtype=torch.float64)
        ) * (total_priority / batch_size)
        # Guard the half-open sampling interval even if a future RNG/backend
        # rounds the final stratified mass up to the root sum.
        upper_bound = torch.nextafter(
            torch.tensor(total_priority, dtype=torch.float64),
            torch.tensor(-math.inf, dtype=torch.float64),
        )
        masses.clamp_max_(upper_bound)
        tree_indices = torch.ones(batch_size, dtype=torch.long)
        while int(tree_indices[0]) < self._tree_capacity:
            left = tree_indices * 2
            left_sums = self._priority_tree[left]
            right_sums = self._priority_tree[left + 1]
            go_right = (masses >= left_sums) & (right_sums > 0.0)
            masses = torch.where(go_right, masses - left_sums, masses)
            tree_indices = left + go_right.to(torch.long)
            selected_sums = torch.where(go_right, right_sums, left_sums)
            selected_upper_bounds = torch.nextafter(
                selected_sums, torch.full_like(selected_sums, -math.inf)
            )
            masses = torch.minimum(masses, selected_upper_bounds)
        indices_cpu = tree_indices - self._tree_capacity
        if bool((indices_cpu >= self._size).any()):
            raise RuntimeError("priority tree selected an uninitialized replay slot")
        leaf_priorities = self._priority_tree[tree_indices]
        sample_probabilities = leaf_priorities / total_priority
        weights = (self._size * sample_probabilities).pow(-max(0.0, float(beta)))
        if normalize_weights:
            weights = weights / weights.max().clamp_min(1e-12)

        return self._build_batch(indices_cpu, weights)

    def sample_demonstrations(
        self,
        batch_size: int,
        *,
        elite_count: int = 0,
    ) -> ReplayBatch:
        """Sample fixed success/elite quotas uniformly without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= elite_count <= batch_size:
            raise ValueError("elite_count must be between zero and batch_size")
        if self.demonstration_count == 0:
            raise ValueError("demonstration replay is empty")

        tiers = self._quality_tiers[: self._size]
        success_pool = torch.nonzero(tiers == 1, as_tuple=False).flatten()
        elite_pool = torch.nonzero(tiers == 2, as_tuple=False).flatten()
        desired_elite_count = elite_count
        elite_count = min(desired_elite_count, int(elite_pool.numel()))
        success_count = min(batch_size - elite_count, int(success_pool.numel()))
        remaining = batch_size - elite_count - success_count
        if remaining > 0:
            extra_elite = min(remaining, int(elite_pool.numel()) - elite_count)
            elite_count += extra_elite
            remaining -= extra_elite
        if remaining > 0:
            extra_success = min(remaining, int(success_pool.numel()) - success_count)
            success_count += extra_success
            remaining -= extra_success
        if remaining > 0:
            raise ValueError("not enough unique demonstration transitions")

        def sample_pool(
            pool: torch.Tensor, count: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if count == 0:
                return (
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.float64),
                )
            selected_positions = torch.randperm(pool.numel())[:count]
            selected = pool[selected_positions]
            importance = torch.ones(count, dtype=torch.float64)
            return selected, importance

        success_indices, success_weights = sample_pool(success_pool, success_count)
        elite_indices, elite_weights = sample_pool(elite_pool, elite_count)
        indices = torch.cat((success_indices, elite_indices))
        weights = torch.cat((success_weights, elite_weights))
        return self._build_batch(indices, weights)

    def update_priorities(
        self,
        indices: torch.Tensor | Sequence[int],
        priorities: torch.Tensor | Sequence[float],
    ) -> None:
        indices_cpu = torch.as_tensor(indices, dtype=torch.long, device="cpu").flatten()
        priorities_cpu = torch.as_tensor(
            priorities, dtype=torch.float32, device="cpu"
        ).flatten()
        if indices_cpu.numel() != priorities_cpu.numel():
            raise ValueError("indices and priorities must have the same length")
        if bool((indices_cpu < 0).any()) or bool((indices_cpu >= self._size).any()):
            raise IndexError("replay priority index out of range")
        self._set_priority_batch(indices_cpu, priorities_cpu)


class BaselineConvDuelingQNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],  # (C, H, W)
        action_dim: int,
        hidden_sizes: Sequence[int],
        *,
        use_dueling: bool = True,
    ) -> None:
        super().__init__()
        channels, height, width = obs_shape
        self.use_dueling = use_dueling

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = self._conv_out_dim(channels, height, width)

        mlp_layers: list[nn.Module] = []
        input_dim = conv_out_dim
        for hidden in hidden_sizes:
            mlp_layers.extend([nn.Linear(input_dim, hidden), nn.ReLU()])
            input_dim = hidden
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        feature_dim = input_dim

        if self.use_dueling:
            head_dim = max(64, feature_dim // 2)
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(feature_dim, action_dim)

    def _conv_out_dim(self, channels: int, height: int, width: int) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            out = self.conv(sample)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        features = self.mlp(x)
        if self.use_dueling:
            value = self.value_head(features)
            advantages = self.advantage_head(features)
            advantages = advantages - advantages.mean(dim=1, keepdim=True)
            return value + advantages
        return self.q_head(features)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class EnhancedConvDuelingQNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],  # (C, H, W)
        action_dim: int,
        hidden_sizes: Sequence[int],
        *,
        use_dueling: bool = True,
    ) -> None:
        super().__init__()
        channels, height, width = obs_shape
        self.use_dueling = use_dueling

        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.res1 = ResidualBlock(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.res2 = ResidualBlock(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
        )
        self.res3 = ResidualBlock(192)
        self.head = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        conv_out_dim = self._conv_out_dim(channels, height, width)

        mlp_layers: list[nn.Module] = []
        input_dim = conv_out_dim
        for hidden in hidden_sizes:
            mlp_layers.extend(
                [
                    nn.Linear(input_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(p=0.1),
                ]
            )
            input_dim = hidden
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        feature_dim = input_dim

        if self.use_dueling:
            head_dim = max(64, feature_dim // 2)
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim),
                nn.SiLU(),
                nn.Linear(head_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim),
                nn.SiLU(),
                nn.Linear(head_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(feature_dim, action_dim)

    def _conv_out_dim(self, channels: int, height: int, width: int) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            sample = self.stem(sample)
            sample = self.res1(sample)
            sample = self.down1(sample)
            sample = self.res2(sample)
            sample = self.down2(sample)
            sample = self.res3(sample)
            sample = self.head(sample)
            return sample.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.head(x)
        x = torch.flatten(x, start_dim=1)
        features = self.mlp(x)
        if self.use_dueling:
            value = self.value_head(features)
            advantages = self.advantage_head(features)
            advantages = advantages - advantages.mean(dim=1, keepdim=True)
            return value + advantages
        return self.q_head(features)


class GroupNormResidualBlock(nn.Module):
    """Residual block without running statistics, suitable for off-policy RL."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SpatialGroupNormDuelingQNetwork(nn.Module):
    """Version 3 network that retains coarse spatial layout through a 3x3 head."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        hidden_sizes: Sequence[int],
        *,
        use_dueling: bool = True,
    ) -> None:
        super().__init__()
        channels, _, _ = obs_shape
        self.use_dueling = use_dueling
        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.res1 = GroupNormResidualBlock(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.res2 = GroupNormResidualBlock(96)
        self.down2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.res3 = GroupNormResidualBlock(128)
        self.spatial_head = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )

        input_dim = 128 * 3 * 3
        mlp_layers: list[nn.Module] = []
        for hidden in hidden_sizes:
            mlp_layers.extend(
                [nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            )
            input_dim = hidden
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        feature_dim = input_dim

        if self.use_dueling:
            head_dim = max(64, feature_dim // 2)
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 1)
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(feature_dim, head_dim),
                nn.SiLU(),
                nn.Linear(head_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.spatial_head(x)
        features = self.mlp(torch.flatten(x, start_dim=1))
        if self.use_dueling:
            value = self.value_head(features)
            advantages = self.advantage_head(features)
            return value + advantages - advantages.mean(dim=1, keepdim=True)
        return self.q_head(features)


class DQNAgent:
    """Deep Q-Network agent tailored for the snake environment."""

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Accelerator device {resolved} was requested, but this PyTorch build "
                "cannot access CUDA or ROCm. Install a matching accelerator build or "
                "use --device cpu explicitly."
            )
        return resolved

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        lr: float = 2e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 200_000,
        min_replay_size: int = 5_000,
        target_update_interval: int = 5_000,
        target_update_tau: float = 0.006,
        hard_update_interval: int = 0,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        dueling_hidden: int | None = None,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: float = 0.997,
        epsilon_decay_steps: int = 250_000,
        n_step: int = 3,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 500_000,
        per_priority_epsilon: float = 1e-5,
        device: str | torch.device | None = None,
        game_config: GameConfig | None = None,
        obs_shape: Tuple[int, int, int] | None = None,
        network_version: int = 3,
        amp_enabled: bool | None = None,
        pin_memory: bool | None = None,
        policy_anchor_weight: float = 0.0,
        teacher_replay_steps: int = 0,
        demonstration_capacity: int = 0,
        demonstration_batch_fraction: float = 0.0,
        elite_demonstration_batch_fraction: float = 0.0,
        demonstration_min_score: float = 4.0,
        demonstration_min_return: float = 0.0,
        demonstration_elite_score: float = 6.0,
        demonstration_elite_return: float = 20.0,
        imitation_loss_weight: float = 0.0,
        imitation_margin: float = 0.8,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.min_replay_size = min_replay_size
        self.target_update_interval = target_update_interval
        self.target_update_tau = max(0.0, target_update_tau)
        self.hard_update_interval = hard_update_interval
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.dueling_hidden = dueling_hidden
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        if epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.behavior_steps = 0
        self.n_step = int(n_step)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = max(1, int(per_beta_frames))
        self.per_priority_epsilon = float(per_priority_epsilon)
        self.device = self._resolve_device(device)
        self.pin_memory = (
            self.device.type == "cuda" if pin_memory is None else bool(pin_memory)
        ) and self.device.type == "cuda"
        self.game_config = game_config
        if network_version not in _NETWORK_SCHEMAS:
            raise ValueError(
                "network_version must be one of "
                f"{sorted(_NETWORK_SCHEMAS)}; got {network_version!r}"
            )
        self.network_version = network_version
        if not math.isfinite(policy_anchor_weight) or policy_anchor_weight < 0:
            raise ValueError("policy_anchor_weight must be finite and non-negative")
        if teacher_replay_steps < 0:
            raise ValueError("teacher_replay_steps must be non-negative")
        if teacher_replay_steps > replay_capacity:
            raise ValueError("teacher_replay_steps must not exceed replay_capacity")
        if demonstration_capacity < 0:
            raise ValueError("demonstration_capacity must be non-negative")
        if not 0.0 <= demonstration_batch_fraction < 1.0:
            raise ValueError("demonstration_batch_fraction must be in [0, 1)")
        if (
            not 0.0
            <= elite_demonstration_batch_fraction
            <= demonstration_batch_fraction
        ):
            raise ValueError(
                "elite_demonstration_batch_fraction must be between zero and "
                "demonstration_batch_fraction"
            )
        requested_demo_rows = int(round(batch_size * demonstration_batch_fraction))
        if demonstration_batch_fraction > 0 and not (
            1 <= requested_demo_rows < batch_size
        ):
            raise ValueError(
                "demonstration_batch_fraction must reserve at least one demo row "
                "and one regular replay row for the configured batch_size"
            )
        requested_elite_rows = int(
            round(batch_size * elite_demonstration_batch_fraction)
        )
        if elite_demonstration_batch_fraction > 0 and requested_elite_rows < 1:
            raise ValueError(
                "elite_demonstration_batch_fraction must reserve at least one row "
                "for the configured batch_size"
            )
        trajectory_thresholds = (
            demonstration_min_score,
            demonstration_min_return,
            demonstration_elite_score,
            demonstration_elite_return,
        )
        if not all(math.isfinite(value) for value in trajectory_thresholds):
            raise ValueError("demonstration score/return thresholds must be finite")
        if demonstration_elite_score < demonstration_min_score:
            raise ValueError("demonstration_elite_score must not be below the minimum")
        if demonstration_elite_return < demonstration_min_return:
            raise ValueError("demonstration_elite_return must not be below the minimum")
        if not math.isfinite(imitation_loss_weight) or imitation_loss_weight < 0:
            raise ValueError("imitation_loss_weight must be finite and non-negative")
        if not math.isfinite(imitation_margin) or imitation_margin <= 0:
            raise ValueError("imitation_margin must be finite and positive")
        if demonstration_batch_fraction > 0 and demonstration_capacity == 0:
            raise ValueError(
                "demonstration_capacity must be positive when demonstration sampling is enabled"
            )
        if imitation_loss_weight > 0 and demonstration_batch_fraction == 0:
            raise ValueError(
                "demonstration_batch_fraction must be positive when imitation loss is enabled"
            )
        self.policy_anchor_weight = float(policy_anchor_weight)
        self.teacher_replay_steps = int(teacher_replay_steps)
        self.demonstration_capacity = int(demonstration_capacity)
        self.demonstration_batch_fraction = float(demonstration_batch_fraction)
        self.elite_demonstration_batch_fraction = float(
            elite_demonstration_batch_fraction
        )
        self.demonstration_min_score = float(demonstration_min_score)
        self.demonstration_min_return = float(demonstration_min_return)
        self.demonstration_elite_score = float(demonstration_elite_score)
        self.demonstration_elite_return = float(demonstration_elite_return)
        self.imitation_loss_weight = float(imitation_loss_weight)
        self.imitation_margin = float(imitation_margin)
        self._configure_amp(amp_enabled)

        if obs_shape is None:
            raise ValueError("obs_shape must be provided for convolutional network")
        self.obs_shape = obs_shape  # (C, H, W)

        if self.target_update_tau <= 0.0 and self.hard_update_interval <= 0:
            self.hard_update_interval = self.target_update_interval

        if self.network_version == 3:
            network_cls = SpatialGroupNormDuelingQNetwork
        elif self.network_version == 2:
            network_cls = EnhancedConvDuelingQNetwork
        else:  # network version 1
            network_cls = BaselineConvDuelingQNetwork
        self.policy_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_anchor_net: nn.Module | None = None

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr, weight_decay=2e-5)
        self.replay_buffer = ReplayBuffer(
            replay_capacity,
            self.device,
            action_dim=action_dim,
            alpha=self.per_alpha,
            priority_epsilon=self.per_priority_epsilon,
            pin_memory=self.pin_memory,
        )
        self.demonstration_replay: ReplayBuffer | None = None
        if self.demonstration_capacity > 0:
            self.demonstration_replay = ReplayBuffer(
                self.demonstration_capacity,
                self.device,
                action_dim=action_dim,
                alpha=self.per_alpha,
                priority_epsilon=self.per_priority_epsilon,
                pin_memory=self.pin_memory,
            )
        self._n_step_buffers: dict[
            int,
            Deque[
                tuple[
                    torch.Tensor,
                    int,
                    float,
                    torch.Tensor,
                    bool,
                    torch.Tensor | None,
                ]
            ],
        ] = {}
        self._n_step_buffer: Deque[
            tuple[torch.Tensor, int, float, torch.Tensor, bool, torch.Tensor | None]
        ] = self._n_step_buffers.setdefault(0, deque())
        self._episode_replay_tokens: dict[int, list[tuple[int, int]]] = {}
        self.demonstration_trajectories_seen = 0
        self.demonstration_transitions_promoted = 0
        self.replay_restored = False
        self.learn_step_counter = 0
        self.policy_transfer_provenance: dict[str, Any] | None = None

    def snapshot_policy_anchor(self) -> None:
        """Freeze the current policy as an immutable teacher for conservative updates."""
        anchor = copy.deepcopy(self.policy_net).to(self.device)
        anchor.eval()
        anchor.requires_grad_(False)
        self.policy_anchor_net = anchor

    @property
    def policy_anchor_enabled(self) -> bool:
        return self.policy_anchor_net is not None

    def configure_amp(self, enabled: bool | None = None) -> None:
        self._configure_amp(enabled)

    def configure_replay_pin_memory(self, enabled: bool | None = None) -> None:
        """Configure pinned sample staging; existing replay observations stay intact."""
        requested = self.device.type == "cuda" if enabled is None else bool(enabled)
        self.pin_memory = requested and self.device.type == "cuda"
        self.replay_buffer.pin_memory = self.pin_memory
        self.replay_buffer._staging.clear()
        if self.demonstration_replay is not None:
            self.demonstration_replay.pin_memory = self.pin_memory
            self.demonstration_replay._staging.clear()

    def _configure_amp(self, enabled: bool | None) -> None:
        if enabled is None:
            enabled = self.device.type == "cuda"
        if enabled and self.device.type != "cuda":
            enabled = False
        self.amp_enabled = bool(enabled)
        if self.amp_enabled:
            try:
                self.grad_scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
                self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(
        self,
        state: np.ndarray | torch.Tensor,
        *,
        epsilon_override: float | None = None,
        action_mask: torch.Tensor | Sequence[bool] | None = None,
    ) -> int:
        return self.select_actions(
            state,
            epsilon_override=epsilon_override,
            action_masks=action_mask,
        )[0]

    def select_actions(
        self,
        states: Sequence[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
        *,
        epsilon_override: float | None = None,
        action_masks: (
            torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]] | None
        ) = None,
    ) -> list[int]:
        """Select actions for an environment batch with exactly one network forward."""
        state_tensor = self._ensure_batch_tensor(states)
        batch_size = int(state_tensor.shape[0])
        legal_masks = self._normalize_action_masks(action_masks, batch_size)
        legal_masks_cpu = legal_masks.to(device="cpu")
        epsilon = self.epsilon if epsilon_override is None else epsilon_override
        was_training = self.policy_net.training
        self.policy_net.eval()
        try:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                q_values = q_values.masked_fill(~legal_masks, -torch.inf)
            actions = q_values.argmax(dim=1).tolist()
        finally:
            self.policy_net.train(was_training)

        if epsilon > 0.0:
            for index in range(batch_size):
                if random.random() < epsilon:
                    legal_actions = torch.nonzero(
                        legal_masks_cpu[index], as_tuple=False
                    ).flatten()
                    actions[index] = int(
                        legal_actions[random.randrange(legal_actions.numel())].item()
                    )
        if epsilon_override is None:
            self.behavior_steps += batch_size
            self._update_epsilon()
        return [int(action) for action in actions]

    def select_anchor_actions(
        self,
        states: Sequence[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
        *,
        action_masks: (
            torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]] | None
        ) = None,
        advance_behavior_steps: bool = True,
    ) -> list[int]:
        """Select greedy actions from the frozen warm-start teacher."""
        if self.policy_anchor_net is None:
            raise RuntimeError("policy anchor is not configured")
        state_tensor = self._ensure_batch_tensor(states)
        batch_size = int(state_tensor.shape[0])
        legal_masks = self._normalize_action_masks(action_masks, batch_size)
        with torch.no_grad():
            q_values = self.policy_anchor_net(state_tensor)
            q_values = q_values.masked_fill(~legal_masks, -torch.inf)
        actions = [int(action) for action in q_values.argmax(dim=1).tolist()]
        if advance_behavior_steps:
            self.behavior_steps += batch_size
            self._update_epsilon()
        return actions

    def remember(
        self,
        state: np.ndarray | torch.Tensor,
        action: int | torch.Tensor,
        reward: float | torch.Tensor,
        next_state: np.ndarray | torch.Tensor,
        done: bool | float | torch.Tensor,
        next_action_mask: torch.Tensor | Sequence[bool] | None = None,
        *,
        stream_id: int = 0,
    ) -> None:
        state_t = self._ensure_cpu_observation(state)
        next_state_t = self._ensure_cpu_observation(next_state)
        action_value = int(torch.as_tensor(action).reshape(()).item())
        reward_value = float(torch.as_tensor(reward).reshape(()).item())
        done_value = bool(torch.as_tensor(done).reshape(()).item())
        mask_t: torch.Tensor | None = None
        if next_action_mask is not None:
            raw_mask = torch.as_tensor(
                next_action_mask, dtype=torch.bool, device="cpu"
            ).flatten()
            if raw_mask.numel() != self.action_dim:
                raise ValueError(
                    f"next_action_mask must contain {self.action_dim} entries; "
                    f"got {raw_mask.numel()}"
                )
            if not bool(raw_mask.any()):
                if not done_value:
                    raise ValueError(
                        "non-terminal next_action_mask must contain a legal action"
                    )
                raw_mask = torch.ones_like(raw_mask)
            mask_t = raw_mask.detach()
        stream_id = int(stream_id)
        stream_buffer = self._n_step_buffers.setdefault(stream_id, deque())
        stream_buffer.append(
            (state_t, action_value, reward_value, next_state_t, done_value, mask_t)
        )
        if done_value:
            while stream_buffer:
                self._emit_n_step_transition(stream_id)
        elif len(stream_buffer) >= self.n_step:
            self._emit_n_step_transition(stream_id)

    def finalize_trajectory(
        self,
        *,
        score: float,
        episode_return: float,
        stream_id: int = 0,
    ) -> dict[str, float]:
        """Promote a completed high-score trajectory into persistent demo replay."""
        if not math.isfinite(score) or not math.isfinite(episode_return):
            raise ValueError("trajectory score and return must be finite")
        stream_id = int(stream_id)
        if self._n_step_buffers.get(stream_id):
            raise RuntimeError(
                "cannot finalize a trajectory before its n-step buffer is empty"
            )
        tokens = self._episode_replay_tokens.pop(stream_id, [])
        quality_tier = 0
        if (
            score >= self.demonstration_min_score
            and episode_return >= self.demonstration_min_return
        ):
            quality_tier = 1
            if (
                score >= self.demonstration_elite_score
                and episode_return >= self.demonstration_elite_return
            ):
                quality_tier = 2

        promoted = 0
        if quality_tier > 0 and self.demonstration_replay is not None:
            promoted = self.replay_buffer.copy_trajectory_to(
                tokens,
                self.demonstration_replay,
                quality_tier=quality_tier,
                trajectory_score=float(score),
                trajectory_return=float(episode_return),
            )
            if promoted > 0:
                self.demonstration_trajectories_seen += 1
                self.demonstration_transitions_promoted += promoted
        return {
            "quality_tier": float(quality_tier if promoted > 0 else 0),
            "promoted_transitions": float(promoted),
        }

    @staticmethod
    def _concatenate_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
        def concatenate_optional(
            left: torch.Tensor | None, right: torch.Tensor | None
        ) -> torch.Tensor | None:
            if left is None and right is None:
                return None
            if left is None or right is None:
                raise RuntimeError("replay batches disagree about action-mask storage")
            return torch.cat((left, right), dim=0)

        return ReplayBatch(
            states=torch.cat((first.states, second.states), dim=0),
            actions=torch.cat((first.actions, second.actions), dim=0),
            rewards=torch.cat((first.rewards, second.rewards), dim=0),
            next_states=torch.cat((first.next_states, second.next_states), dim=0),
            dones=torch.cat((first.dones, second.dones), dim=0),
            discounts=torch.cat((first.discounts, second.discounts), dim=0),
            weights=torch.cat((first.weights, second.weights), dim=0),
            indices=torch.cat((first.indices, second.indices), dim=0),
            next_action_masks=concatenate_optional(
                first.next_action_masks, second.next_action_masks
            ),
            demonstration_mask=torch.cat(
                (first.demonstration_mask, second.demonstration_mask), dim=0
            ),
            imitation_mask=torch.cat(
                (first.imitation_mask, second.imitation_mask), dim=0
            ),
            quality_tiers=torch.cat((first.quality_tiers, second.quality_tiers), dim=0),
            trajectory_scores=torch.cat(
                (first.trajectory_scores, second.trajectory_scores), dim=0
            ),
            trajectory_returns=torch.cat(
                (first.trajectory_returns, second.trajectory_returns), dim=0
            ),
        )

    def learn(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None
        if self.policy_anchor_weight > 0 and self.policy_anchor_net is None:
            raise RuntimeError(
                "policy anchor weight is enabled but no frozen anchor is configured"
            )
        beta_progress = min(1.0, self.learn_step_counter / self.per_beta_frames)
        beta = self.per_beta_start + beta_progress * (1.0 - self.per_beta_start)
        sampling_started = time.perf_counter()
        requested_demo_count = int(
            round(self.batch_size * self.demonstration_batch_fraction)
        )
        available_demo_count = (
            self.demonstration_replay.demonstration_count
            if self.demonstration_replay is not None
            else 0
        )
        # Ramp the quota with unique stored transitions. This prevents one new
        # success from being duplicated across a large fraction of the batch.
        demo_count = min(requested_demo_count, available_demo_count)
        general_count = self.batch_size - demo_count
        general_batch = self.replay_buffer.sample(
            general_count, beta=beta, normalize_weights=False
        )
        demo_batch: ReplayBatch | None = None
        if demo_count > 0:
            assert self.demonstration_replay is not None
            elite_count = min(
                demo_count,
                int(round(self.batch_size * self.elite_demonstration_batch_fraction)),
            )
            demo_batch = self.demonstration_replay.sample_demonstrations(
                demo_count,
                elite_count=elite_count,
            )
            batch = self._concatenate_batches(general_batch, demo_batch)
        else:
            batch = general_batch
        batch = replace(
            batch,
            weights=batch.weights / batch.weights.max().clamp_min(1e-12),
        )
        sampling_seconds = time.perf_counter() - sampling_started

        scaler = (
            self.grad_scaler
            if (self.amp_enabled and self.grad_scaler is not None)
            else None
        )

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            policy_q_values = self.policy_net(batch.states)
            q_values = policy_q_values.gather(
                1, batch.actions.long().unsqueeze(1)
            ).squeeze(1)
            with torch.no_grad():
                if self.use_double_dqn:
                    was_training = self.policy_net.training
                    self.policy_net.eval()
                    try:
                        policy_next_q = self.policy_net(batch.next_states)
                    finally:
                        self.policy_net.train(was_training)
                    if batch.next_action_masks is not None:
                        policy_next_q = policy_next_q.masked_fill(
                            ~batch.next_action_masks, -torch.inf
                        )
                    next_actions = policy_next_q.argmax(dim=1, keepdim=True)
                    next_q_values = (
                        self.target_net(batch.next_states)
                        .gather(1, next_actions)
                        .squeeze(1)
                    )
                else:
                    target_next_q = self.target_net(batch.next_states)
                    if batch.next_action_masks is not None:
                        target_next_q = target_next_q.masked_fill(
                            ~batch.next_action_masks, -torch.inf
                        )
                    next_q_values = target_next_q.max(dim=1).values
            targets = (
                batch.rewards + batch.discounts * (1 - batch.dones) * next_q_values
            )
            targets = targets.to(q_values.dtype)
            td_errors = targets - q_values
            element_losses = F.smooth_l1_loss(q_values, targets, reduction="none")
            td_loss = (batch.weights * element_losses).mean()
            if self.policy_anchor_net is not None and self.policy_anchor_weight > 0:
                with torch.no_grad():
                    anchor_q_values = self.policy_anchor_net(batch.states)
                anchor_loss = F.smooth_l1_loss(
                    policy_q_values, anchor_q_values, reduction="mean"
                )
            else:
                anchor_loss = torch.zeros((), device=self.device, dtype=td_loss.dtype)
            if bool(batch.imitation_mask.any()):
                demo_q_values = policy_q_values[batch.imitation_mask]
                demo_actions = batch.actions[batch.imitation_mask].long()
                margins = torch.full_like(demo_q_values, self.imitation_margin)
                margins.scatter_(1, demo_actions.unsqueeze(1), 0.0)
                competing_values = (demo_q_values + margins).max(dim=1).values
                chosen_demo_values = demo_q_values.gather(
                    1, demo_actions.unsqueeze(1)
                ).squeeze(1)
                imitation_elements = (competing_values - chosen_demo_values).clamp_min(
                    0.0
                )
                quality_weights = torch.where(
                    batch.quality_tiers[batch.imitation_mask] == 2,
                    torch.full_like(imitation_elements, 1.5),
                    torch.ones_like(imitation_elements),
                )
                imitation_loss = (
                    imitation_elements * quality_weights
                ).sum() / quality_weights.sum().clamp_min(1.0)
            else:
                imitation_loss = torch.zeros(
                    (), device=self.device, dtype=td_loss.dtype
                )
            loss = (
                td_loss
                + self.policy_anchor_weight * anchor_loss
                + self.imitation_loss_weight * imitation_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=5.0
            )
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=5.0
            )
            self.optimizer.step()

        # General replay PER already requires TD errors on the host. Measure this
        # existing synchronization point rather than adding torch.cuda.synchronize().
        gpu_wait_started = time.perf_counter()
        absolute_td_errors = td_errors.detach().abs()
        td_errors_cpu = absolute_td_errors.to(device="cpu", dtype=torch.float32)
        loss_value = float(loss.detach().cpu().item())
        td_loss_value = float(td_loss.detach().cpu().item())
        anchor_loss_value = float(anchor_loss.detach().cpu().item())
        imitation_loss_value = float(imitation_loss.detach().cpu().item())
        td_error_value = float(absolute_td_errors.mean().detach().cpu().item())
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
        q_mean_value = float(q_values.detach().mean().cpu().item())
        gpu_wait_seconds = (
            time.perf_counter() - gpu_wait_started
            if self.device.type == "cuda"
            else 0.0
        )
        self.replay_buffer.update_priorities(
            general_batch.indices,
            td_errors_cpu[:general_count],
        )
        self.learn_step_counter += 1
        self._update_target_network()
        return {
            "loss": loss_value,
            "td_loss": td_loss_value,
            "anchor_loss": anchor_loss_value,
            "anchor_weight": self.policy_anchor_weight,
            "imitation_loss": imitation_loss_value,
            "imitation_weight": self.imitation_loss_weight,
            "demonstration_batch_fraction": demo_count / self.batch_size,
            "elite_demonstration_batch_fraction": float(
                (batch.quality_tiers == 2).sum().detach().cpu().item()
            )
            / self.batch_size,
            "td_error": td_error_value,
            "grad_norm": grad_norm_value,
            "q_mean": q_mean_value,
            "per_beta": float(beta),
            "sampling_seconds": sampling_seconds,
            "gpu_wait_seconds": gpu_wait_seconds,
            "pin_memory": float(self.replay_buffer.pin_memory),
        }

    def save(self, path: str) -> None:
        numpy_state = np.random.get_state()
        checkpoint = {
            "checkpoint_schema_version": 4,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "policy_anchor_state_dict": (
                self.policy_anchor_net.state_dict()
                if self.policy_anchor_net is not None
                else None
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_scaler_state_dict": (
                self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            ),
            "rng_state": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    # Python ints are portable across old/new PyTorch serializers.
                    # Some PyTorch releases cannot pickle torch.uint32 storage.
                    "state": [int(value) for value in numpy_state[1]],
                    "position": int(numpy_state[2]),
                    "has_gauss": int(numpy_state[3]),
                    "cached_gaussian": float(numpy_state[4]),
                },
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            "metadata": {
                "checkpoint_schema_version": 4,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "lr": self.lr,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "replay_capacity": self.replay_capacity,
                "min_replay_size": self.min_replay_size,
                "target_update_interval": self.target_update_interval,
                "target_update_tau": self.target_update_tau,
                "hard_update_interval": self.hard_update_interval,
                "use_double_dqn": self.use_double_dqn,
                "use_dueling": self.use_dueling,
                "dueling_hidden": self.dueling_hidden,
                "epsilon_start": self.epsilon_start,
                "epsilon_final": self.epsilon_final,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "epsilon": self.epsilon,
                "behavior_steps": self.behavior_steps,
                "n_step": self.n_step,
                "per_alpha": self.per_alpha,
                "per_beta_start": self.per_beta_start,
                "per_beta_frames": self.per_beta_frames,
                "per_priority_epsilon": self.per_priority_epsilon,
                "replay_size": len(self.replay_buffer),
                "replay_restored": False,
                "amp_enabled": self.amp_enabled,
                "pin_memory": self.replay_buffer.pin_memory,
                "device": str(self.device),
                "learn_step_counter": self.learn_step_counter,
                "obs_shape": self.obs_shape,
                "network_version": self.network_version,
                "observation_schema": _NETWORK_SCHEMAS[self.network_version][
                    "observation_schema"
                ],
                "action_schema": _NETWORK_SCHEMAS[self.network_version][
                    "action_schema"
                ],
                "spatial_transfer_capable": _NETWORK_SCHEMAS[self.network_version][
                    "spatial_transfer_capable"
                ],
                "policy_anchor_weight": self.policy_anchor_weight,
                "teacher_replay_steps": self.teacher_replay_steps,
                "demonstration_capacity": self.demonstration_capacity,
                "demonstration_batch_fraction": self.demonstration_batch_fraction,
                "elite_demonstration_batch_fraction": self.elite_demonstration_batch_fraction,
                "demonstration_min_score": self.demonstration_min_score,
                "demonstration_min_return": self.demonstration_min_return,
                "demonstration_elite_score": self.demonstration_elite_score,
                "demonstration_elite_return": self.demonstration_elite_return,
                "imitation_loss_weight": self.imitation_loss_weight,
                "imitation_margin": self.imitation_margin,
                "demonstration_replay_size": (
                    len(self.demonstration_replay)
                    if self.demonstration_replay is not None
                    else 0
                ),
                "demonstration_trajectories_seen": self.demonstration_trajectories_seen,
                "demonstration_transitions_promoted": self.demonstration_transitions_promoted,
                "policy_anchor_enabled": self.policy_anchor_net is not None,
                "game_config": asdict(self.game_config) if self.game_config else None,
                "policy_transfer_provenance": self.policy_transfer_provenance,
            },
        }
        target_path = os.path.abspath(os.fspath(path))
        target_dir = os.path.dirname(target_path) or os.curdir
        os.makedirs(target_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(target_path)}.",
            suffix=".tmp",
            dir=target_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, target_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _read_policy_checkpoint_snapshot(
        path: str, *, expected_sha256: str | None = None
    ) -> PolicyCheckpointSnapshot:
        """Read and deserialize a policy source exactly once before validation."""
        with open(path, "rb") as stream:
            checkpoint_bytes = stream.read()
        actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if expected_sha256 is not None:
            normalized_sha256 = expected_sha256.lower()
            if len(normalized_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in normalized_sha256
            ):
                raise ValueError(
                    "expected_sha256 must be a 64-character hexadecimal digest"
                )
            if actual_sha256 != normalized_sha256:
                raise RuntimeError(
                    "policy checkpoint failed SHA-256 authentication: "
                    f"expected SHA-256 {normalized_sha256}, got {actual_sha256}"
                )
        checkpoint_stream = io.BytesIO(checkpoint_bytes)
        try:
            checkpoint = torch.load(
                checkpoint_stream, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            checkpoint_stream.seek(0)
            checkpoint = torch.load(checkpoint_stream, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise RuntimeError(  # noqa: TRY004 - checkpoint incompatibility contract
                "policy checkpoint must contain a mapping"
            )
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(  # noqa: TRY004 - checkpoint incompatibility contract
                "policy checkpoint is missing mapping metadata"
            )
        policy_state = checkpoint.get("policy_state_dict")
        if not isinstance(policy_state, dict):
            raise RuntimeError(  # noqa: TRY004 - checkpoint incompatibility contract
                "policy checkpoint is missing policy_state_dict"
            )
        return PolicyCheckpointSnapshot(
            checkpoint_bytes=checkpoint_bytes,
            checkpoint_sha256=actual_sha256,
            checkpoint=checkpoint,
            metadata=metadata,
            policy_state_dict=policy_state,
        )

    def _apply_policy_checkpoint_snapshot(
        self, snapshot: PolicyCheckpointSnapshot
    ) -> dict[str, Any]:
        """Validate a snapshot fully, then atomically replace policy/target weights."""
        metadata = snapshot.metadata
        policy_state = snapshot.policy_state_dict

        incompatibilities: list[str] = []
        source_network_version = metadata.get("network_version")
        source_schema = _NETWORK_SCHEMAS.get(source_network_version)
        if source_schema is None:
            incompatibilities.append(
                "unsupported network_version "
                f"{source_network_version!r}; supported versions are "
                f"{sorted(_NETWORK_SCHEMAS)}"
            )
        if source_network_version != self.network_version:
            incompatibilities.append(
                "network_version "
                f"{source_network_version!r} != {self.network_version!r}"
            )
        source_action_dim = metadata.get("action_dim")
        if source_action_dim != self.action_dim:
            incompatibilities.append(
                f"action_dim {source_action_dim!r} != {self.action_dim!r}"
            )
        source_hidden_sizes = metadata.get("hidden_sizes")
        if not isinstance(source_hidden_sizes, (list, tuple)):
            incompatibilities.append("hidden_sizes is missing or invalid")
        elif tuple(source_hidden_sizes) != self.hidden_sizes:
            incompatibilities.append(
                f"hidden_sizes {tuple(source_hidden_sizes)!r} != {self.hidden_sizes!r}"
            )
        source_obs_shape = metadata.get("obs_shape")
        if (
            not isinstance(source_obs_shape, (list, tuple))
            or len(source_obs_shape) != 3
        ):
            incompatibilities.append("obs_shape is missing or invalid")
        elif source_obs_shape[0] != self.obs_shape[0]:
            incompatibilities.append(
                f"obs channels {source_obs_shape[0]!r} != {self.obs_shape[0]!r}"
            )
        is_cross_map = (
            isinstance(source_obs_shape, (list, tuple))
            and len(source_obs_shape) == 3
            and tuple(source_obs_shape[1:]) != tuple(self.obs_shape[1:])
        )
        if (
            is_cross_map
            and source_schema is not None
            and isinstance(source_obs_shape, (list, tuple))
        ):
            expected_action_dim = source_schema["action_dim"]
            if source_action_dim != expected_action_dim:
                incompatibilities.append(
                    f"network v{source_network_version} action schema requires "
                    f"action_dim={expected_action_dim}; got {source_action_dim!r}"
                )
            source_action_schema = metadata.get(
                "action_schema", source_schema["action_schema"]
            )
            if source_action_schema != source_schema["action_schema"]:
                incompatibilities.append(
                    f"action_schema {source_action_schema!r} != "
                    f"{source_schema['action_schema']!r}"
                )
            expected_channels = source_schema["observation_channels"]
            if len(source_obs_shape) == 3 and source_obs_shape[0] != expected_channels:
                incompatibilities.append(
                    f"network v{source_network_version} observation schema requires "
                    f"{expected_channels} channels; got {source_obs_shape[0]!r}"
                )
            source_observation_schema = metadata.get(
                "observation_schema", source_schema["observation_schema"]
            )
            if source_observation_schema != source_schema["observation_schema"]:
                incompatibilities.append(
                    f"observation_schema {source_observation_schema!r} != "
                    f"{source_schema['observation_schema']!r}"
                )
            if (
                not source_schema["spatial_transfer_capable"]
                or metadata.get(
                    "spatial_transfer_capable",
                    source_schema["spatial_transfer_capable"],
                )
                is not True
            ):
                incompatibilities.append(
                    f"network_version {source_network_version} is not spatial-transfer capable"
                )

        current_state = self.policy_net.state_dict()
        source_keys = set(policy_state)
        current_keys = set(current_state)
        missing_keys = sorted(current_keys - source_keys)
        unexpected_keys = sorted(source_keys - current_keys)
        if missing_keys:
            incompatibilities.append(f"missing state_dict keys: {missing_keys}")
        if unexpected_keys:
            incompatibilities.append(f"unexpected state_dict keys: {unexpected_keys}")
        for key in sorted(current_keys & source_keys):
            source_value = policy_state[key]
            current_value = current_state[key]
            if not isinstance(source_value, torch.Tensor):
                incompatibilities.append(f"state_dict[{key!r}] is not a tensor")
                continue
            if source_value.shape != current_value.shape:
                incompatibilities.append(
                    f"state_dict[{key!r}] shape {tuple(source_value.shape)!r} "
                    f"!= {tuple(current_value.shape)!r}"
                )
            if source_value.dtype != current_value.dtype:
                incompatibilities.append(
                    f"state_dict[{key!r}] dtype {source_value.dtype} "
                    f"!= {current_value.dtype}"
                )
            if source_value.layout != current_value.layout:
                incompatibilities.append(
                    f"state_dict[{key!r}] layout {source_value.layout} "
                    f"!= {current_value.layout}"
                )

        if incompatibilities:
            raise RuntimeError(
                "incompatible policy checkpoint: " + "; ".join(incompatibilities)
            )

        # Strict loading cannot encounter a structural mismatch after the checks
        # above, so no incompatible checkpoint can partially modify this agent.
        self.policy_net.load_state_dict(policy_state, strict=True)
        self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)
        self.target_net.eval()
        return dict(metadata)

    def load_policy_weights(
        self, path: str, *, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        """Load policy weights into an already constructed compatible agent.

        Prefer :meth:`from_policy_checkpoint` when the destination map or agent
        does not already exist.  This compatibility method shares the same
        single-read, SHA-authenticated validation contract.
        """
        snapshot = self._read_policy_checkpoint_snapshot(
            path, expected_sha256=expected_sha256
        )
        return self._apply_policy_checkpoint_snapshot(snapshot)

    @staticmethod
    def _checkpoint_game_config(metadata: Mapping[str, Any]) -> GameConfig | None:
        raw_config = metadata.get("game_config")
        if not isinstance(raw_config, Mapping):
            return None
        config_data = dict(raw_config)
        config_data.setdefault("idle_growth_per_food", 0)
        config_data.setdefault("max_episode_steps", 0)
        return GameConfig(**config_data)

    @classmethod
    def from_policy_checkpoint(
        cls,
        path: str,
        *,
        target_game_config: GameConfig | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        target_max_episode_steps: int | None = None,
        device: str | torch.device | None = None,
        expected_sha256: str | None = None,
        agent_options: Mapping[str, Any] | None = None,
    ) -> "DQNAgent":
        """Construct a fresh target-map agent from policy weights only.

        The checkpoint is captured as one immutable byte snapshot and optionally
        SHA-authenticated.  Only architecture metadata and policy tensors cross
        the boundary: optimizer/scaler state, replay, counters, exploration,
        teacher state, and checkpoint RNG are never restored.
        """
        if (target_width is None) != (target_height is None):
            raise ValueError("target_width and target_height must be provided together")
        if target_game_config is not None and target_width is not None:
            raise ValueError(
                "Provide target_game_config or target_width/target_height, not both"
            )
        if target_game_config is not None and target_max_episode_steps is not None:
            raise ValueError(
                "target_max_episode_steps cannot be combined with target_game_config"
            )
        if target_max_episode_steps is not None and target_max_episode_steps <= 0:
            raise ValueError("target_max_episode_steps must be positive")
        snapshot = cls._read_policy_checkpoint_snapshot(
            path, expected_sha256=expected_sha256
        )
        metadata = snapshot.metadata
        source_obs_shape = metadata.get("obs_shape")
        if (
            not isinstance(source_obs_shape, (list, tuple))
            or len(source_obs_shape) != 3
            or any(
                not isinstance(value, int) or value <= 0 for value in source_obs_shape
            )
        ):
            raise RuntimeError("policy checkpoint obs_shape is missing or invalid")
        source_config = cls._checkpoint_game_config(metadata)
        if source_config is not None and (
            source_config.width != int(source_obs_shape[2])
            or source_config.height != int(source_obs_shape[1])
        ):
            raise RuntimeError(
                "policy checkpoint game_config dimensions conflict with obs_shape: "
                f"game_config={source_config.width}x{source_config.height}, "
                f"obs_shape={source_obs_shape[2]}x{source_obs_shape[1]}"
            )
        if target_game_config is None:
            base_config = source_config or GameConfig(
                width=int(source_obs_shape[2]), height=int(source_obs_shape[1])
            )
            target_game_config = GameConfig(**asdict(base_config))
            if target_width is not None and target_height is not None:
                target_game_config.width = int(target_width)
                target_game_config.height = int(target_height)
                target_game_config.max_episode_steps = (
                    int(target_max_episode_steps)
                    if target_max_episode_steps is not None
                    else int(target_width) * int(target_height) * 20
                )
            elif target_max_episode_steps is not None:
                target_game_config.max_episode_steps = int(target_max_episode_steps)
        target_game_config.validate()

        structural_fields = {
            "state_dim",
            "action_dim",
            "hidden_sizes",
            "use_dueling",
            "dueling_hidden",
            "device",
            "game_config",
            "obs_shape",
            "network_version",
        }
        options = dict(agent_options or {})
        forbidden = sorted(structural_fields.intersection(options))
        if forbidden:
            raise ValueError(
                "Policy transfer inherits checkpoint network structure; remove agent_options: "
                + ", ".join(forbidden)
            )
        target_obs_shape = (
            int(source_obs_shape[0]),
            int(target_game_config.height),
            int(target_game_config.width),
        )
        try:
            action_dim = int(metadata["action_dim"])
            hidden_sizes = tuple(int(value) for value in metadata["hidden_sizes"])
            network_version = int(metadata["network_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "policy checkpoint network structure metadata is missing or invalid"
            ) from exc
        source_schema = _NETWORK_SCHEMAS.get(network_version)
        if source_schema is None:
            raise RuntimeError(
                f"unsupported policy checkpoint network_version {network_version!r}; "
                f"supported versions are {sorted(_NETWORK_SCHEMAS)}"
            )
        cross_map = int(target_game_config.width) != int(source_obs_shape[2]) or int(
            target_game_config.height
        ) != int(source_obs_shape[1])
        if cross_map:
            if not source_schema["spatial_transfer_capable"]:
                raise RuntimeError(
                    f"network_version {network_version} is not spatial-transfer capable; "
                    "use the checkpoint's original map dimensions"
                )
            expected_action_dim = source_schema["action_dim"]
            expected_channels = source_schema["observation_channels"]
            source_action_schema = metadata.get(
                "action_schema", source_schema["action_schema"]
            )
            source_observation_schema = metadata.get(
                "observation_schema", source_schema["observation_schema"]
            )
            if (
                action_dim != expected_action_dim
                or int(source_obs_shape[0]) != expected_channels
                or source_action_schema != source_schema["action_schema"]
                or source_observation_schema != source_schema["observation_schema"]
                or metadata.get("spatial_transfer_capable", True) is not True
            ):
                raise RuntimeError(
                    "policy checkpoint does not satisfy its declared spatial-transfer "
                    "action/observation schema"
                )
        agent = cls(
            state_dim=int(np.prod(target_obs_shape)),
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            use_dueling=bool(metadata.get("use_dueling", True)),
            dueling_hidden=metadata.get("dueling_hidden"),
            device=device,
            game_config=target_game_config,
            obs_shape=target_obs_shape,
            network_version=network_version,
            **options,
        )
        embedded_metadata = agent._apply_policy_checkpoint_snapshot(snapshot)
        source_map = {
            "width": (
                source_config.width if source_config else int(source_obs_shape[2])
            ),
            "height": (
                source_config.height if source_config else int(source_obs_shape[1])
            ),
        }
        target_map = {
            "width": target_game_config.width,
            "height": target_game_config.height,
        }
        agent.policy_transfer_provenance = {
            "transfer_mode": "policy_only",
            "source_path": os.path.abspath(os.fspath(path)),
            "checkpoint_sha256": snapshot.checkpoint_sha256,
            "sha256_verified": expected_sha256 is not None,
            "source_checkpoint_schema_version": embedded_metadata.get(
                "checkpoint_schema_version"
            ),
            "source_network_version": network_version,
            "source_action_dim": action_dim,
            "source_hidden_sizes": list(hidden_sizes),
            "source_obs_shape": list(source_obs_shape),
            "target_obs_shape": list(target_obs_shape),
            "source_game_config": (
                asdict(source_config) if source_config is not None else None
            ),
            "target_game_config": asdict(target_game_config),
            "source_map": source_map,
            "target_map": target_map,
            "cross_map": source_map != target_map,
            "optimizer_restored": False,
            "replay_restored": False,
            "rng_restored": False,
        }
        return agent

    @classmethod
    def validate_policy_sidecar_identity(
        cls,
        sidecar_metadata: Mapping[str, Any],
        transfer_provenance: Mapping[str, Any],
    ) -> None:
        """Reject sidecar architecture claims that disagree with policy bytes."""
        expected = {
            "network_version": transfer_provenance.get("source_network_version"),
            "action_dim": transfer_provenance.get("source_action_dim"),
            "obs_shape": transfer_provenance.get("source_obs_shape"),
        }
        conflicts = {
            key: (sidecar_metadata[key], value)
            for key, value in expected.items()
            if key in sidecar_metadata and sidecar_metadata[key] != value
        }
        if conflicts:
            raise RuntimeError(
                "Policy source sidecar conflicts with authenticated checkpoint bytes: "
                + ", ".join(
                    f"{key}=sidecar:{values[0]!r}/checkpoint:{values[1]!r}"
                    for key, values in conflicts.items()
                )
            )

    @classmethod
    def restore_training_checkpoint(
        cls, path: str, *, device: str | torch.device | None = None
    ) -> "DQNAgent":
        """Restore the complete resumable training state from a checkpoint."""
        # Loading to CPU first avoids inheriting a stale checkpoint device and
        # lets module/optimizer loading move state coherently to the requested device.
        resolved_device = (
            cls._resolve_device(device) if device is not None else torch.device("cpu")
        )
        try:
            checkpoint = torch.load(
                path, map_location=resolved_device, weights_only=True
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            checkpoint = torch.load(path, map_location=resolved_device)
        metadata = checkpoint["metadata"]
        obs_shape = (
            tuple(metadata.get("obs_shape")) if metadata.get("obs_shape") else None
        )
        game_config_data = metadata.get("game_config")
        if game_config_data:
            game_config_data = dict(game_config_data)
            # Missing fields retain historical fixed-idle/no-time-horizon
            # semantics instead of inheriting current dataclass defaults.
            game_config_data.setdefault("idle_growth_per_food", 0)
            game_config_data.setdefault("max_episode_steps", 0)
        # Checkpoint device strings describe the machine that created the artifact,
        # not a portable runtime requirement. An omitted override follows the current
        # machine's auto-detected device; an explicit override must be available.
        runtime_device = cls._resolve_device(device)
        agent = cls(
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            hidden_sizes=tuple(metadata.get("hidden_sizes", (256, 256))),
            lr=metadata.get("lr", 2e-4),
            gamma=metadata.get("gamma", 0.99),
            batch_size=metadata.get("batch_size", 64),
            replay_capacity=metadata.get("replay_capacity", 200_000),
            min_replay_size=metadata.get("min_replay_size", 5_000),
            target_update_interval=metadata.get("target_update_interval", 5_000),
            target_update_tau=metadata.get("target_update_tau", 0.006),
            hard_update_interval=metadata.get("hard_update_interval", 0),
            use_double_dqn=metadata.get("use_double_dqn", True),
            use_dueling=metadata.get("use_dueling", True),
            dueling_hidden=metadata.get("dueling_hidden"),
            epsilon_start=metadata.get("epsilon_start", 1.0),
            epsilon_final=metadata.get("epsilon_final", 0.01),
            epsilon_decay=metadata.get("epsilon_decay", 0.997),
            epsilon_decay_steps=metadata.get("epsilon_decay_steps", 250_000),
            n_step=metadata.get("n_step", 1),
            per_alpha=metadata.get("per_alpha", 0.6),
            per_beta_start=metadata.get("per_beta_start", 0.4),
            per_beta_frames=metadata.get("per_beta_frames", 500_000),
            per_priority_epsilon=metadata.get("per_priority_epsilon", 1e-5),
            device=runtime_device,
            game_config=GameConfig(**game_config_data) if game_config_data else None,
            obs_shape=obs_shape,
            network_version=metadata.get("network_version", 1),
            amp_enabled=metadata.get("amp_enabled"),
            # Pinned memory is a runtime/device property and replay is not
            # restored, so configure it from the selected device every time.
            pin_memory=None,
            policy_anchor_weight=metadata.get("policy_anchor_weight", 0.0),
            teacher_replay_steps=metadata.get("teacher_replay_steps", 0),
            demonstration_capacity=metadata.get("demonstration_capacity", 0),
            demonstration_batch_fraction=metadata.get(
                "demonstration_batch_fraction", 0.0
            ),
            elite_demonstration_batch_fraction=metadata.get(
                "elite_demonstration_batch_fraction", 0.0
            ),
            demonstration_min_score=metadata.get("demonstration_min_score", 4.0),
            demonstration_min_return=metadata.get("demonstration_min_return", 0.0),
            demonstration_elite_score=metadata.get("demonstration_elite_score", 6.0),
            demonstration_elite_return=metadata.get("demonstration_elite_return", 20.0),
            imitation_loss_weight=metadata.get("imitation_loss_weight", 0.0),
            imitation_margin=metadata.get("imitation_margin", 0.8),
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["policy_state_dict"])
        )
        anchor_state = checkpoint.get("policy_anchor_state_dict")
        if anchor_state is not None:
            agent.snapshot_policy_anchor()
            assert agent.policy_anchor_net is not None
            agent.policy_anchor_net.load_state_dict(anchor_state, strict=True)
        elif metadata.get("policy_anchor_enabled"):
            raise RuntimeError(
                "checkpoint metadata requires a policy anchor but its state is missing"
            )
        if checkpoint.get("optimizer_state_dict") is not None:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("grad_scaler_state_dict")
        if scaler_state is not None and agent.grad_scaler is not None:
            agent.grad_scaler.load_state_dict(scaler_state)
        agent.learn_step_counter = metadata.get("learn_step_counter", 0)
        agent.demonstration_trajectories_seen = int(
            metadata.get("demonstration_trajectories_seen", 0)
        )
        agent.demonstration_transitions_promoted = int(
            metadata.get("demonstration_transitions_promoted", 0)
        )
        raw_transfer_provenance = metadata.get("policy_transfer_provenance")
        agent.policy_transfer_provenance = (
            dict(raw_transfer_provenance)
            if isinstance(raw_transfer_provenance, Mapping)
            else None
        )
        agent.epsilon = metadata.get("epsilon", metadata.get("epsilon_final", 0.01))
        if "behavior_steps" in metadata:
            agent.behavior_steps = int(metadata["behavior_steps"])
        else:
            span = agent.epsilon_start - agent.epsilon_final
            progress = (
                0.0 if span <= 0 else (agent.epsilon_start - agent.epsilon) / span
            )
            agent.behavior_steps = int(
                max(0.0, min(1.0, progress)) * agent.epsilon_decay_steps
            )
        agent.replay_restored = False
        agent._restore_rng_state(checkpoint.get("rng_state"))
        return agent

    @classmethod
    def load(cls, path: str, *, device: str | torch.device | None = None) -> "DQNAgent":
        """Compatibility alias for :meth:`restore_training_checkpoint`."""
        return cls.restore_training_checkpoint(path, device=device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_target_network(self) -> None:
        if self.target_update_tau > 0.0:
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    target_param.data.mul_(1 - self.target_update_tau).add_(
                        param.data, alpha=self.target_update_tau
                    )
                policy_buffers = dict(self.policy_net.named_buffers())
                for name, target_buffer in self.target_net.named_buffers():
                    policy_buffer = policy_buffers[name]
                    if target_buffer.is_floating_point() or target_buffer.is_complex():
                        target_buffer.mul_(1 - self.target_update_tau).add_(
                            policy_buffer, alpha=self.target_update_tau
                        )
                    else:
                        target_buffer.copy_(policy_buffer)
        elif (
            self.hard_update_interval > 0
            and self.learn_step_counter % self.hard_update_interval == 0
        ):
            self.target_net.load_state_dict(self.policy_net.state_dict())
        elif (
            self.target_update_interval > 0
            and self.learn_step_counter % self.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _update_epsilon(self) -> None:
        progress = min(1.0, self.behavior_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (
            self.epsilon_final - self.epsilon_start
        )

    def _normalize_action_mask(
        self, action_mask: torch.Tensor | Sequence[bool] | None
    ) -> torch.Tensor:
        if action_mask is None:
            return torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
        mask = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).flatten()
        if mask.numel() != self.action_dim:
            raise ValueError(
                f"action_mask must contain {self.action_dim} entries; got {mask.numel()}"
            )
        if not bool(mask.any()):
            raise ValueError("action_mask must contain at least one legal action")
        return mask

    def _normalize_action_masks(
        self,
        action_masks: torch.Tensor | Sequence[bool] | Sequence[Sequence[bool]] | None,
        batch_size: int,
    ) -> torch.Tensor:
        if action_masks is None:
            return torch.ones(
                (batch_size, self.action_dim), dtype=torch.bool, device=self.device
            )
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        if masks.dim() == 1 and batch_size == 1:
            masks = masks.unsqueeze(0)
        if tuple(masks.shape) != (batch_size, self.action_dim):
            raise ValueError(
                "action_masks must have shape "
                f"({batch_size}, {self.action_dim}); got {tuple(masks.shape)}"
            )
        if not bool(masks.any(dim=1).all()):
            raise ValueError("each action mask must contain at least one legal action")
        return masks

    def _emit_n_step_transition(self, stream_id: int = 0) -> None:
        stream_buffer = self._n_step_buffers.get(stream_id)
        if not stream_buffer:
            return
        accumulated_reward = 0.0
        steps = 0
        final_next_state = stream_buffer[0][3]
        final_done = False
        final_mask = stream_buffer[0][5]
        for _, _, reward, next_state, done, next_mask in list(stream_buffer)[
            : self.n_step
        ]:
            accumulated_reward += (self.gamma**steps) * reward
            steps += 1
            final_next_state = next_state
            final_done = done
            final_mask = next_mask
            if done:
                break
        state, action, _, _, action_was_terminal, _ = stream_buffer[0]
        token = self.replay_buffer.push(
            state,
            action,
            accumulated_reward,
            final_next_state,
            final_done,
            discount=self.gamma**steps,
            next_action_mask=final_mask,
            imitation_eligible=not action_was_terminal,
        )
        if self.demonstration_replay is not None:
            self._episode_replay_tokens.setdefault(stream_id, []).append(token)
        stream_buffer.popleft()
        if not stream_buffer and stream_id != 0:
            del self._n_step_buffers[stream_id]

    @staticmethod
    def _restore_rng_state(rng_state: Any) -> None:
        if not isinstance(rng_state, dict):
            return
        python_state = rng_state.get("python")
        if python_state is not None:
            random.setstate(python_state)
        numpy_state = rng_state.get("numpy")
        if isinstance(numpy_state, dict) and numpy_state.get("state") is not None:
            state_tensor = torch.as_tensor(numpy_state["state"], device="cpu")
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    state_tensor.numpy().astype(np.uint32, copy=False),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        torch_state = rng_state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(
                torch.as_tensor(torch_state, device="cpu", dtype=torch.uint8)
            )
        cuda_states = rng_state.get("torch_cuda")
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(
                [
                    torch.as_tensor(state, device="cpu", dtype=torch.uint8)
                    for state in cuda_states
                ]
            )

    def _ensure_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(self.device, dtype=torch.float32)
        else:
            tensor = torch.from_numpy(value).to(self.device, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.view(self.obs_shape)
        elif tensor.dim() == 3 and tensor.shape[-1] == self.obs_shape[0]:
            tensor = tensor.permute(2, 0, 1)
        return tensor

    def _ensure_cpu_observation(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device="cpu", dtype=torch.float32)
        else:
            tensor = torch.from_numpy(value).to(dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.view(self.obs_shape)
        elif tensor.dim() == 3 and tensor.shape[-1] == self.obs_shape[0]:
            tensor = tensor.permute(2, 0, 1)
        if tuple(tensor.shape) != tuple(self.obs_shape):
            raise ValueError(
                f"observation must have shape {self.obs_shape}; got {tuple(tensor.shape)}"
            )
        # Callers commonly pass views into a reusable batched encoder.  The
        # n-step queue must own a snapshot because that staging view is
        # overwritten on the next environment tick.
        return tensor.contiguous().clone()

    def _ensure_batch_tensor(
        self,
        values: Sequence[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(values, torch.Tensor) and values.device.type != "cpu":
            device_batch = values.detach()
            if device_batch.dim() == 1:
                device_batch = device_batch.view(1, *self.obs_shape)
            elif device_batch.dim() == 2 and device_batch.shape[1] == self.state_dim:
                device_batch = device_batch.view(-1, *self.obs_shape)
            elif device_batch.dim() == 3:
                if tuple(device_batch.shape) == tuple(self.obs_shape):
                    device_batch = device_batch.unsqueeze(0)
                elif device_batch.shape[-1] == self.obs_shape[0] and tuple(
                    device_batch.shape[:2]
                ) == tuple(self.obs_shape[1:]):
                    device_batch = device_batch.permute(2, 0, 1).unsqueeze(0)
            elif device_batch.dim() == 4 and tuple(device_batch.shape[1:]) != tuple(
                self.obs_shape
            ):
                if device_batch.shape[-1] == self.obs_shape[0] and tuple(
                    device_batch.shape[1:3]
                ) == tuple(self.obs_shape[1:]):
                    device_batch = device_batch.permute(0, 3, 1, 2)
            if device_batch.dim() != 4 or tuple(device_batch.shape[1:]) != tuple(
                self.obs_shape
            ):
                raise ValueError(
                    f"batched observations must have shape (N, {self.obs_shape}); "
                    f"got {tuple(values.shape)}"
                )
            return device_batch.to(self.device, dtype=torch.float32).contiguous()
        if isinstance(values, (np.ndarray, torch.Tensor)):
            raw = (
                values.detach()
                if isinstance(values, torch.Tensor)
                else torch.from_numpy(values)
            )
            if raw.dim() in (1, 3):
                observations = [self._ensure_cpu_observation(raw)]
                batch_cpu = torch.stack(observations)
            elif raw.dim() == 2 and raw.shape[1] == self.state_dim:
                batch_cpu = raw.to(device="cpu", dtype=torch.float32).view(
                    -1, *self.obs_shape
                )
            elif raw.dim() == 4:
                batch_cpu = raw.to(device="cpu", dtype=torch.float32)
                if tuple(batch_cpu.shape[1:]) != tuple(self.obs_shape):
                    if batch_cpu.shape[-1] == self.obs_shape[0] and tuple(
                        batch_cpu.shape[1:3]
                    ) == tuple(self.obs_shape[1:]):
                        batch_cpu = batch_cpu.permute(0, 3, 1, 2)
                    else:
                        raise ValueError(
                            f"batched observations must have shape (N, {self.obs_shape}); "
                            f"got {tuple(raw.shape)}"
                        )
            else:
                raise ValueError(
                    f"unsupported batched observation shape {tuple(raw.shape)}"
                )
            batch_cpu = batch_cpu.contiguous()
        else:
            if not values:
                raise ValueError("states batch must not be empty")
            batch_cpu = torch.stack(
                [self._ensure_cpu_observation(value) for value in values]
            )
        if int(batch_cpu.shape[0]) <= 0:
            raise ValueError("states batch must not be empty")
        non_blocking = False
        if self.pin_memory and not batch_cpu.is_pinned():
            try:
                batch_cpu = batch_cpu.pin_memory()
            except RuntimeError as exc:
                warnings.warn(
                    f"pinned action batch allocation failed; using pageable memory: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.configure_replay_pin_memory(False)
            else:
                non_blocking = True
        elif batch_cpu.is_pinned():
            non_blocking = self.device.type == "cuda"
        return batch_cpu.to(self.device, dtype=torch.float32, non_blocking=non_blocking)


def flatten_observation(
    source: Union[np.ndarray, "SnakeGameEnv"],
    device: torch.device | str,
    *,
    expected_channels: int | None = None,
) -> torch.Tensor:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    env: SnakeGameEnv | None
    if hasattr(source, "as_numpy"):
        env = source  # type: ignore[assignment]
        grid = env.as_numpy()
    else:
        env = None
        grid = source

    if grid.ndim != 3:
        raise ValueError(
            f"Expected observation with 3 dimensions (H, W, C); got shape {grid.shape}"
        )

    height, width, _ = grid.shape
    base_tensor = torch.from_numpy(grid).permute(2, 0, 1).to(dtype=torch.float32)
    channels: list[torch.Tensor] = [base_tensor]

    xs = (
        torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        .view(1, 1, width)
        .expand(1, height, width)
    )
    ys = (
        torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        .view(1, height, 1)
        .expand(1, height, width)
    )
    channels.extend([xs, ys])

    head_mask = (
        base_tensor[2] if base_tensor.shape[0] >= 3 else torch.zeros((height, width))
    )
    head_indices = torch.nonzero(head_mask, as_tuple=False)
    if head_indices.numel() > 0:
        head_y, head_x = head_indices[0].tolist()
    else:
        head_y = height // 2
        head_x = width // 2

    food_x: int
    food_y: int
    if env is not None and env.food is not None:
        food_x, food_y = env.food
    else:
        food_mask = (
            base_tensor[1]
            if base_tensor.shape[0] >= 2
            else torch.zeros((height, width))
        )
        food_indices = torch.nonzero(food_mask, as_tuple=False)
        if food_indices.numel() > 0:
            food_y, food_x = food_indices[0].tolist()
        else:
            food_x = head_x
            food_y = head_y

    denom_x = max(1, width - 1)
    denom_y = max(1, height - 1)
    norm_dx = float(food_x - head_x) / denom_x
    norm_dy = float(food_y - head_y) / denom_y
    food_dx_channel = torch.full((1, height, width), norm_dx, dtype=torch.float32)
    food_dy_channel = torch.full((1, height, width), norm_dy, dtype=torch.float32)
    channels.extend([food_dx_channel, food_dy_channel])

    direction_channels = torch.zeros((4, height, width), dtype=torch.float32)
    if env is not None:
        direction_idx = int(env.direction)
        direction_channels[direction_idx].fill_(1.0)
    channels.append(direction_channels)

    if env is not None:
        snake_length = len(env.snake)
    else:
        snake_length = int(base_tensor[0].sum().item())
    max_length = max(1, height * width)
    length_ratio = min(1.0, max(0.0, snake_length / max_length))
    length_channel = torch.full(
        (1, height, width), float(length_ratio), dtype=torch.float32
    )
    channels.append(length_channel)

    idle_progress = 0.0
    if env is not None:
        idle_limit = int(getattr(env, "idle_limit", env.config.max_idle_steps))
        if idle_limit > 0:
            idle_progress = min(1.0, env.steps_since_food / idle_limit)
    idle_channel = torch.full((1, height, width), idle_progress, dtype=torch.float32)
    channels.append(idle_channel)

    danger_channels = torch.ones((4, height, width), dtype=torch.float32)
    if env is not None:
        occupied = base_tensor[0] > 0.5
        tail = env.snake[-1] if env.snake else None
        food_target = env.food
        current_dir = env.direction
        opposite_dir = Action((int(current_dir) + 2) % 4)
        for idx, action in enumerate(Action):
            move_dir = current_dir if action == opposite_dir else action
            dx, dy = move_dir.vector
            nx = head_x + dx
            ny = head_y + dy
            if env.config.allow_wrap:
                nx %= width
                ny %= height
            else:
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    danger_channels[idx].fill_(0.0)
                    continue
            target = (nx, ny)
            if not env.config.allow_wrap and (
                nx < 0 or nx >= width or ny < 0 or ny >= height
            ):
                danger_channels[idx].fill_(0.0)
                continue
            if (
                tail is not None
                and target == tail
                and (food_target is None or target != food_target)
            ):
                continue
            if occupied[int(ny), int(nx)]:
                danger_channels[idx].fill_(0.0)
    channels.append(danger_channels)

    # Version 3 keeps the original 17 channels as a stable prefix and appends
    # Markov-critical body and finite-horizon information. Legacy callers can request 3 or 17
    # channels through expected_channels and receive the exact old prefix.
    tail_channel = torch.zeros((1, height, width), dtype=torch.float32)
    body_order_channel = torch.zeros((1, height, width), dtype=torch.float32)
    if env is not None and env.snake:
        snake_segments = env.snake
        tail_x, tail_y = snake_segments[-1]
        if 0 <= tail_x < width and 0 <= tail_y < height:
            tail_channel[0, tail_y, tail_x] = 1.0
        segment_count = len(snake_segments)
        for index, (segment_x, segment_y) in enumerate(snake_segments):
            if 0 <= segment_x < width and 0 <= segment_y < height:
                # Head is near zero and tail is one; +1 keeps every occupied
                # segment distinguishable from empty space.
                body_order_channel[0, segment_y, segment_x] = (
                    index + 1
                ) / segment_count
    channels.extend([tail_channel, body_order_channel])

    horizon_progress = 0.0
    if env is not None:
        max_episode_steps = int(getattr(env.config, "max_episode_steps", 0))
        if max_episode_steps > 0:
            horizon_progress = min(
                1.0, int(getattr(env, "steps", 0)) / max_episode_steps
            )
    horizon_channel = torch.full(
        (1, height, width), horizon_progress, dtype=torch.float32
    )
    channels.append(horizon_channel)

    stacked = torch.cat(channels, dim=0)
    if expected_channels is not None:
        if stacked.shape[0] > expected_channels:
            stacked = stacked[:expected_channels]
        elif stacked.shape[0] < expected_channels:
            pad = torch.zeros(
                (expected_channels - stacked.shape[0], height, width),
                dtype=stacked.dtype,
            )
            stacked = torch.cat([stacked, pad], dim=0)

    return stacked.contiguous().to(device=device)


__all__ = [
    "DQNAgent",
    "ReplayBatch",
    "ReplayBuffer",
    "SpatialGroupNormDuelingQNetwork",
    "flatten_observation",
]
