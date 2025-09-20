"""Snake game environment package."""

from .env import Action, GameConfig, SnakeGameEnv
from .gui import SnakeGameGUI
from .dqn_agent import DQNAgent, flatten_observation

__all__ = ["Action", "GameConfig", "SnakeGameEnv", "SnakeGameGUI", "DQNAgent", "flatten_observation"]
