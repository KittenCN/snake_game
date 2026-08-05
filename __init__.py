"""Snake game environment package."""

try:
    from .env import (
        Action,
        GameConfig,
        RelativeAction,
        SnakeGameEnv,
        relative_to_absolute,
    )
    from .dqn_agent import DQNAgent, flatten_observation
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from env import (
        Action,
        GameConfig,
        RelativeAction,
        SnakeGameEnv,
        relative_to_absolute,
    )
    from dqn_agent import DQNAgent, flatten_observation


def __getattr__(name: str):
    """Keep Tkinter optional for headless training and analysis imports."""
    if name != "SnakeGameGUI":
        raise AttributeError(name)
    try:
        from .gui import SnakeGameGUI
    except ImportError:
        from gui import SnakeGameGUI
    return SnakeGameGUI


__all__ = [
    "Action",
    "RelativeAction",
    "relative_to_absolute",
    "GameConfig",
    "SnakeGameEnv",
    "SnakeGameGUI",
    "DQNAgent",
    "flatten_observation",
]
