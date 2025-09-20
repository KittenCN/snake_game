"""Simple demo that runs the environment with a random policy."""

from __future__ import annotations

import time

try:
    from .env import SnakeGameEnv
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from env import SnakeGameEnv


def main() -> None:
    env = SnakeGameEnv()
    observation = env.reset()
    env.render()
    while not observation["done"]:
        action = env.sample_action()
        observation, reward, done, info = env.step(action)
        print(f"Action: {action.name}, reward: {reward:+.2f}, info: {info}")
        env.render()
        time.sleep(0.2)
        if done:
            break


if __name__ == "__main__":
    main()
