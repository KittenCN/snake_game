"""DQN agent implementation for the snake game."""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Sequence, Tuple, TYPE_CHECKING, Union

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
        self._size = 0
        self._position = 0
        self._states: torch.Tensor | None = None
        self._next_states: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._rewards: torch.Tensor | None = None
        self._dones: torch.Tensor | None = None
        self._discounts: torch.Tensor | None = None
        self._next_action_masks: torch.Tensor | None = None
        self._priorities = torch.zeros(self.capacity, dtype=torch.float32, device="cpu")
        self._tree_capacity = 1 << (self.capacity - 1).bit_length()
        self._priority_tree = torch.zeros(
            self._tree_capacity * 2, dtype=torch.float32, device="cpu"
        )
        self._max_priority = 1.0

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

    @staticmethod
    def _cpu_scalar(
        value: torch.Tensor | int | float | bool, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device="cpu").reshape(())

    def _set_priority(self, index: int, priority: float) -> None:
        """Update one proportional-PER leaf and its O(log capacity) sum path."""
        raw_priority = max(abs(float(priority)), self.priority_epsilon)
        self._priorities[index] = raw_priority
        self._max_priority = max(self._max_priority, raw_priority)
        scaled_priority = 1.0 if self.alpha == 0.0 else raw_priority**self.alpha
        tree_index = self._tree_capacity + index
        delta = scaled_priority - float(self._priority_tree[tree_index])
        while tree_index:
            self._priority_tree[tree_index] += delta
            tree_index //= 2

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
    ) -> None:
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

    def sample(self, batch_size: int, *, beta: float = 0.4) -> ReplayBatch:
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
            torch.arange(batch_size, dtype=torch.float32) + torch.rand(batch_size)
        ) * (total_priority / batch_size)
        tree_indices = torch.ones(batch_size, dtype=torch.long)
        while int(tree_indices[0]) < self._tree_capacity:
            left = tree_indices * 2
            left_sums = self._priority_tree[left]
            go_right = masses >= left_sums
            masses = torch.where(go_right, masses - left_sums, masses)
            tree_indices = left + go_right.to(torch.long)
        indices_cpu = tree_indices - self._tree_capacity
        if bool((indices_cpu >= self._size).any()):
            raise RuntimeError("priority tree selected an uninitialized replay slot")
        leaf_priorities = self._priority_tree[tree_indices]
        sample_probabilities = leaf_priorities / total_priority
        weights = (self._size * sample_probabilities).pow(-max(0.0, float(beta)))
        weights = weights / weights.max().clamp_min(1e-12)

        non_blocking = self.device.type == "cuda"
        masks = None
        if self._next_action_masks is not None:
            masks = self._next_action_masks[indices_cpu].to(
                self.device, non_blocking=non_blocking
            )
        return ReplayBatch(
            states=self._states[indices_cpu].to(
                self.device, dtype=torch.float32, non_blocking=non_blocking
            ),
            actions=self._actions[indices_cpu].to(
                self.device, non_blocking=non_blocking
            ),
            rewards=self._rewards[indices_cpu].to(
                self.device, non_blocking=non_blocking
            ),
            next_states=self._next_states[indices_cpu].to(
                self.device, dtype=torch.float32, non_blocking=non_blocking
            ),
            dones=self._dones[indices_cpu].to(self.device, non_blocking=non_blocking),
            discounts=self._discounts[indices_cpu].to(
                self.device, non_blocking=non_blocking
            ),
            weights=weights.to(self.device, non_blocking=non_blocking),
            indices=indices_cpu,
            next_action_masks=masks,
        )

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
        for index, priority in zip(indices_cpu.tolist(), priorities_cpu.tolist()):
            self._set_priority(index, abs(priority) + self.priority_epsilon)


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
            return torch.device("cpu")
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
        self.game_config = game_config
        self.network_version = network_version
        self._configure_amp(amp_enabled)

        if obs_shape is None:
            raise ValueError("obs_shape must be provided for convolutional network")
        self.obs_shape = obs_shape  # (C, H, W)

        if self.target_update_tau <= 0.0 and self.hard_update_interval <= 0:
            self.hard_update_interval = self.target_update_interval

        if self.network_version >= 3:
            network_cls = SpatialGroupNormDuelingQNetwork
        elif self.network_version == 2:
            network_cls = EnhancedConvDuelingQNetwork
        else:
            network_cls = BaselineConvDuelingQNetwork
        self.policy_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr, weight_decay=2e-5)
        self.replay_buffer = ReplayBuffer(
            replay_capacity,
            self.device,
            action_dim=action_dim,
            alpha=self.per_alpha,
            priority_epsilon=self.per_priority_epsilon,
        )
        self._n_step_buffer: Deque[
            tuple[torch.Tensor, int, float, torch.Tensor, bool, torch.Tensor | None]
        ] = deque()
        self.replay_restored = False
        self.learn_step_counter = 0

    def configure_amp(self, enabled: bool | None = None) -> None:
        self._configure_amp(enabled)

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
        epsilon = self.epsilon if epsilon_override is None else epsilon_override
        legal_mask = self._normalize_action_mask(action_mask)
        legal_actions = torch.nonzero(legal_mask, as_tuple=False).flatten()
        if epsilon > 0.0 and random.random() < epsilon:
            chosen = int(legal_actions[random.randrange(legal_actions.numel())].item())
        else:
            state_tensor = self._ensure_tensor(state).unsqueeze(0)
            was_training = self.policy_net.training
            self.policy_net.eval()
            try:
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    q_values = q_values.masked_fill(
                        ~legal_mask.unsqueeze(0), -torch.inf
                    )
                chosen = int(q_values.argmax(dim=1).item())
            finally:
                self.policy_net.train(was_training)
        if epsilon_override is None:
            self.behavior_steps += 1
            self._update_epsilon()
        return chosen

    def remember(
        self,
        state: np.ndarray | torch.Tensor,
        action: int | torch.Tensor,
        reward: float | torch.Tensor,
        next_state: np.ndarray | torch.Tensor,
        done: bool | float | torch.Tensor,
        next_action_mask: torch.Tensor | Sequence[bool] | None = None,
    ) -> None:
        state_t = self._ensure_tensor(state)
        next_state_t = self._ensure_tensor(next_state)
        action_value = int(torch.as_tensor(action).reshape(()).item())
        reward_value = float(torch.as_tensor(reward).reshape(()).item())
        done_value = bool(torch.as_tensor(done).reshape(()).item())
        mask_t: torch.Tensor | None = None
        if next_action_mask is not None:
            raw_mask = torch.as_tensor(
                next_action_mask, dtype=torch.bool, device=self.device
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
        self._n_step_buffer.append(
            (state_t, action_value, reward_value, next_state_t, done_value, mask_t)
        )
        if done_value:
            while self._n_step_buffer:
                self._emit_n_step_transition()
        elif len(self._n_step_buffer) >= self.n_step:
            self._emit_n_step_transition()

    def learn(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None
        beta_progress = min(1.0, self.learn_step_counter / self.per_beta_frames)
        beta = self.per_beta_start + beta_progress * (1.0 - self.per_beta_start)
        batch = self.replay_buffer.sample(self.batch_size, beta=beta)

        scaler = (
            self.grad_scaler
            if (self.amp_enabled and self.grad_scaler is not None)
            else None
        )

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            q_values = (
                self.policy_net(batch.states)
                .gather(1, batch.actions.long().unsqueeze(1))
                .squeeze(1)
            )
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
            loss = (batch.weights * element_losses).mean()

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

        self.replay_buffer.update_priorities(
            batch.indices,
            td_errors.detach().abs().to(device="cpu", dtype=torch.float32),
        )
        self.learn_step_counter += 1
        self._update_target_network()
        return {
            "loss": float(loss.detach().cpu().item()),
            "td_error": float(td_errors.detach().abs().mean().cpu().item()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            "q_mean": float(q_values.detach().mean().cpu().item()),
            "per_beta": float(beta),
        }

    def save(self, path: str) -> None:
        numpy_state = np.random.get_state()
        checkpoint = {
            "checkpoint_schema_version": 3,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_scaler_state_dict": (
                self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            ),
            "rng_state": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "state": torch.from_numpy(numpy_state[1].copy()),
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
                "checkpoint_schema_version": 3,
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
                "device": str(self.device),
                "learn_step_counter": self.learn_step_counter,
                "obs_shape": self.obs_shape,
                "network_version": self.network_version,
                "game_config": asdict(self.game_config) if self.game_config else None,
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

    @classmethod
    def load(cls, path: str, *, device: str | torch.device | None = None) -> "DQNAgent":
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
        saved_device = device if device is not None else metadata.get("device")
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
            device=cls._resolve_device(saved_device),
            game_config=GameConfig(**game_config_data) if game_config_data else None,
            obs_shape=obs_shape,
            network_version=metadata.get("network_version", 1),
            amp_enabled=metadata.get("amp_enabled"),
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["policy_state_dict"])
        )
        if checkpoint.get("optimizer_state_dict") is not None:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("grad_scaler_state_dict")
        if scaler_state is not None and agent.grad_scaler is not None:
            agent.grad_scaler.load_state_dict(scaler_state)
        agent.learn_step_counter = metadata.get("learn_step_counter", 0)
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

    def _emit_n_step_transition(self) -> None:
        if not self._n_step_buffer:
            return
        accumulated_reward = 0.0
        steps = 0
        final_next_state = self._n_step_buffer[0][3]
        final_done = False
        final_mask = self._n_step_buffer[0][5]
        for _, _, reward, next_state, done, next_mask in list(self._n_step_buffer)[
            : self.n_step
        ]:
            accumulated_reward += (self.gamma**steps) * reward
            steps += 1
            final_next_state = next_state
            final_done = done
            final_mask = next_mask
            if done:
                break
        state, action, _, _, _, _ = self._n_step_buffer[0]
        self.replay_buffer.push(
            state,
            action,
            accumulated_reward,
            final_next_state,
            final_done,
            discount=self.gamma**steps,
            next_action_mask=final_mask,
        )
        self._n_step_buffer.popleft()

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
