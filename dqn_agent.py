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
    """Simple experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class DQNAgent:
    """Deep Q-Network agent tailored for the snake environment."""

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
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str | None = None,
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
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.game_config = game_config

        self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.learn_step_counter = 0

        self.loss_fn = nn.SmoothL1Loss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, *, epsilon_override: float | None = None) -> int:
        """Epsilon-greedy policy."""
        epsilon = self.epsilon if epsilon_override is None else epsilon_override
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * (1 - dones_t) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

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
                    "epsilon_start": self.epsilon_start,
                    "epsilon_final": self.epsilon_final,
                    "epsilon_decay": self.epsilon_decay,
                    "device": str(self.device),
                    "game_config": asdict(self.game_config) if self.game_config else None,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str | None = None) -> "DQNAgent":
        checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
            epsilon_start=metadata.get("epsilon_start", 0.0),
            epsilon_final=metadata.get("epsilon_final", 0.0),
            epsilon_decay=1.0,
            device=device or metadata.get("device"),
            game_config=GameConfig(**metadata["game_config"]) if metadata.get("game_config") else None,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = metadata.get("epsilon_final", 0.0)
        return agent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)


def flatten_observation(grid: np.ndarray) -> np.ndarray:
    return grid.astype(np.float32).flatten()


__all__ = ["DQNAgent", "ReplayBuffer", "flatten_observation"]
