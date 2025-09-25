"""DQN agent implementation for the snake game."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict
from typing import Deque, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
from torch import nn
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


class ReplayBuffer:
    """Experience replay buffer that stores tensors directly on device."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._buffer: Deque[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self._buffer.append(
            (
                state.detach().to(self.device),
                action.detach().to(self.device),
                reward.detach().to(self.device),
                next_state.detach().to(self.device),
                done.detach().to(self.device),
            )
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states, dim=0),
            torch.stack(actions, dim=0),
            torch.stack(rewards, dim=0),
            torch.stack(next_states, dim=0),
            torch.stack(dones, dim=0),
        )



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
        epsilon_final: float = 0.01,
        epsilon_decay: float = 0.997,
        device: str | torch.device | None = None,
        game_config: GameConfig | None = None,
        obs_shape: Tuple[int, int, int] | None = None,
        network_version: int = 2,
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
        self.epsilon_decay = epsilon_decay
        self.device = self._resolve_device(device)
        self.game_config = game_config
        self.network_version = network_version
        self._configure_amp(amp_enabled)

        if obs_shape is None:
            raise ValueError("obs_shape must be provided for convolutional network")
        self.obs_shape = obs_shape  # (C, H, W)

        if self.target_update_tau <= 0.0 and self.hard_update_interval <= 0:
            self.hard_update_interval = self.target_update_interval

        network_cls = EnhancedConvDuelingQNetwork if self.network_version >= 2 else BaselineConvDuelingQNetwork
        self.policy_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net = network_cls(
            self.obs_shape, action_dim, hidden_sizes, use_dueling=self.use_dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr, weight_decay=2e-5)
        self.replay_buffer = ReplayBuffer(replay_capacity, self.device)
        self.learn_step_counter = 0

        self.loss_fn = nn.SmoothL1Loss()

    def configure_amp(self, enabled: bool | None = None) -> None:
        self._configure_amp(enabled)

    def _configure_amp(self, enabled: bool | None) -> None:
        if enabled is None:
            enabled = self.device.type == "cuda"
        if enabled and self.device.type != "cuda":
            enabled = False
        self.amp_enabled = bool(enabled)
        if self.amp_enabled:
            self.grad_scaler = torch.amp.GradScaler()
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
    ) -> int:
        epsilon = self.epsilon if epsilon_override is None else epsilon_override
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = self._ensure_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray | torch.Tensor,
        action: int | torch.Tensor,
        reward: float | torch.Tensor,
        next_state: np.ndarray | torch.Tensor,
        done: bool | float | torch.Tensor,
    ) -> None:
        state_t = self._ensure_tensor(state)
        next_state_t = self._ensure_tensor(next_state)
        action_t = torch.as_tensor(action, dtype=torch.long, device=self.device).view(-1)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1)
        self.replay_buffer.push(state_t, action_t, reward_t, next_state_t, done_t)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        scaler = self.grad_scaler if (self.amp_enabled and self.grad_scaler is not None) else None

        with torch.amp.autocast(device_type=self.device.type,enabled=self.amp_enabled):
            q_values = self.policy_net(states).gather(1, actions.long()).squeeze(1)
            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards.squeeze(1) + self.gamma * (1 - dones.squeeze(1)) * next_q_values
            targets = targets.to(q_values.dtype)
            loss = self.loss_fn(q_values, targets)

        loss_value = float(loss.detach().cpu().item())

        self.optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
            self.optimizer.step()

        self.learn_step_counter += 1
        self._update_target_network()
        self._decay_epsilon()
        return loss_value

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": {
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
                    "epsilon": self.epsilon,
                    "amp_enabled": self.amp_enabled,
                    "device": str(self.device),
                    "learn_step_counter": self.learn_step_counter,
                    "obs_shape": self.obs_shape,
                    "network_version": self.network_version,
                    "game_config": asdict(self.game_config) if self.game_config else None,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str | torch.device | None = None) -> "DQNAgent":
        checkpoint = torch.load(path, map_location=cls._resolve_device(device))
        metadata = checkpoint["metadata"]
        obs_shape = tuple(metadata.get("obs_shape")) if metadata.get("obs_shape") else None
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
            device=cls._resolve_device(device or metadata.get("device")),
            game_config=GameConfig(**metadata["game_config"]) if metadata.get("game_config") else None,
            obs_shape=obs_shape,
            network_version=metadata.get("network_version", 1),
            amp_enabled=metadata.get("amp_enabled"),
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.learn_step_counter = metadata.get("learn_step_counter", 0)
        agent.epsilon = metadata.get("epsilon", metadata.get("epsilon_final", 0.01))
        return agent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_target_network(self) -> None:
        if self.target_update_tau > 0.0:
            with torch.no_grad():
                for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.mul_(1 - self.target_update_tau).add_(param.data, alpha=self.target_update_tau)
        elif self.hard_update_interval > 0 and self.learn_step_counter % self.hard_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        elif self.target_update_interval > 0 and self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

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
        raise ValueError(f"Expected observation with 3 dimensions (H, W, C); got shape {grid.shape}")

    height, width, _ = grid.shape
    base_tensor = torch.from_numpy(grid).permute(2, 0, 1).to(dtype=torch.float32)
    channels: list[torch.Tensor] = [base_tensor]

    xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32).view(1, 1, width).expand(1, height, width)
    ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32).view(1, height, 1).expand(1, height, width)
    channels.extend([xs, ys])

    head_mask = base_tensor[2] if base_tensor.shape[0] >= 3 else torch.zeros((height, width))
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
        food_mask = base_tensor[1] if base_tensor.shape[0] >= 2 else torch.zeros((height, width))
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
    length_channel = torch.full((1, height, width), float(length_ratio), dtype=torch.float32)
    channels.append(length_channel)

    idle_progress = 0.0
    if env is not None and env.config.max_idle_steps > 0:
        idle_progress = min(1.0, env.steps_since_food / env.config.max_idle_steps)
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
            if not env.config.allow_wrap and (nx < 0 or nx >= width or ny < 0 or ny >= height):
                danger_channels[idx].fill_(0.0)
                continue
            if tail is not None and target == tail and (food_target is None or target != food_target):
                continue
            if occupied[int(ny), int(nx)]:
                danger_channels[idx].fill_(0.0)
    channels.append(danger_channels)

    stacked = torch.cat(channels, dim=0)
    if expected_channels is not None:
        if stacked.shape[0] > expected_channels:
            stacked = stacked[:expected_channels]
        elif stacked.shape[0] < expected_channels:
            pad = torch.zeros((expected_channels - stacked.shape[0], height, width), dtype=stacked.dtype)
            stacked = torch.cat([stacked, pad], dim=0)

    return stacked.contiguous().to(device=device)


__all__ = ["DQNAgent", "ReplayBuffer", "flatten_observation"]
