"""Snake game environment package."""
try:
    from .env import Action, GameConfig, SnakeGameEnv
    from .gui import SnakeGameGUI
    from .dqn_agent import DQNAgent, flatten_observation
except ImportError:
    import os
    import sys
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from env import Action, GameConfig, SnakeGameEnv
    from gui import SnakeGameGUI
    from dqn_agent import DQNAgent, flatten_observation

__all__ = ["Action", "GameConfig", "SnakeGameEnv", "SnakeGameGUI", "DQNAgent", "flatten_observation"]
