"""DQN agent implementation for the snake game."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict
from typing import Deque, Sequence, Tuple

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


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        *,
        dueling: bool = False,
        dueling_hidden: int | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden), nn.ReLU()])
            input_dim = hidden
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        feature_dim = input_dim if layers else state_dim

        self.dueling = dueling
        if dueling:
            hidden_dim = dueling_hidden or feature_dim
            self.value_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        if self.dueling:
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
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        replay_capacity: int = 50_000,
        min_replay_size: int = 1_000,
        target_update_interval: int = 1_000,
        target_update_tau: float = 0.0,
        hard_update_interval: int = 0,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        dueling_hidden: int | None = None,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str | torch.device | None = None,
        game_config: GameConfig | None = None,
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

        if self.target_update_tau <= 0.0 and self.hard_update_interval <= 0:
            self.hard_update_interval = self.target_update_interval

        self.policy_net = QNetwork(
            state_dim,
            action_dim,
            hidden_sizes,
            dueling=self.use_dueling,
            dueling_hidden=self.dueling_hidden,
        ).to(self.device)
        self.target_net = QNetwork(
            state_dim,
            action_dim,
            hidden_sizes,
            dueling=self.use_dueling,
            dueling_hidden=self.dueling_hidden,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.replay_buffer = ReplayBuffer(replay_capacity, self.device)
        self.learn_step_counter = 0

        self.loss_fn = nn.SmoothL1Loss()

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
        state_tensor = self._ensure_tensor(state).view(1, -1)
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

        q_values = self.policy_net(states).gather(1, actions.long()).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards.squeeze(1) + self.gamma * (1 - dones.squeeze(1)) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        self._update_target_network()
        self._decay_epsilon()
        return float(loss.item())

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
                    "device": str(self.device),
                    "learn_step_counter": self.learn_step_counter,
                    "game_config": asdict(self.game_config) if self.game_config else None,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str | torch.device | None = None) -> "DQNAgent":
        checkpoint = torch.load(path, map_location=cls._resolve_device(device))
        metadata = checkpoint["metadata"]
        agent = cls(
            state_dim=metadata["state_dim"],
            action_dim=metadata["action_dim"],
            hidden_sizes=tuple(metadata.get("hidden_sizes", (256, 256))),
            lr=metadata.get("lr", 1e-3),
            gamma=metadata.get("gamma", 0.99),
            batch_size=metadata.get("batch_size", 128),
            replay_capacity=metadata.get("replay_capacity", 1),
            min_replay_size=metadata.get("min_replay_size", 1),
            target_update_interval=metadata.get("target_update_interval", 1_000),
            target_update_tau=metadata.get("target_update_tau", 0.0),
            hard_update_interval=metadata.get("hard_update_interval", 0),
            use_double_dqn=metadata.get("use_double_dqn", True),
            use_dueling=metadata.get("use_dueling", True),
            dueling_hidden=metadata.get("dueling_hidden"),
            epsilon_start=metadata.get("epsilon_start", 0.0),
            epsilon_final=metadata.get("epsilon_final", 0.0),
            epsilon_decay=metadata.get("epsilon_decay", 1.0),
            device=cls._resolve_device(device or metadata.get("device")),
            game_config=GameConfig(**metadata["game_config"]) if metadata.get("game_config") else None,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.learn_step_counter = metadata.get("learn_step_counter", 0)
        agent.epsilon = metadata.get("epsilon", metadata.get("epsilon_final", 0.0))
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
        return tensor.view(-1)


def flatten_observation(grid: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(grid).to(device=device, dtype=torch.float32).view(-1)


__all__ = ["DQNAgent", "ReplayBuffer", "flatten_observation"]
