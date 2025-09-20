"""Simple CLI runner for the snake game environment."""

from __future__ import annotations

import argparse
import time

from .env import Action, GameConfig, SnakeGameEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake game CLI runner")
    parser.add_argument("--width", type=int, default=20, help="Grid width")
    parser.add_argument("--height", type=int, default=20, help="Grid height")
    parser.add_argument("--speed", type=float, default=0.2, help="Delay between steps (seconds)")
    parser.add_argument("--wrap", action="store_true", help="Enable wrap-around edges")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--auto",
        type=int,
        default=0,
        metavar="STEPS",
        help="Run a random agent for the given number of steps (0 = interactive mode)",
    )
    return parser.parse_args()


def interactive_loop(env: SnakeGameEnv, speed: float) -> None:
    print("Controls: w(up), d(right), s(down), a(left), q(quit)")
    env.render()

    action_map = {
        "w": Action.UP,
        "d": Action.RIGHT,
        "s": Action.DOWN,
        "a": Action.LEFT,
    }

    while not env.done:
        key = input("move> ").strip().lower()
        if key == "q":
            print("Exiting game.")
            break
        if key not in action_map:
            print("Invalid input. Use w/a/s/d or q to quit.")
            continue
        obs, reward, done, info = env.step(action_map[key])
        env.render()
        print(
            f"score={obs['score']} reward={reward:.2f} steps={obs['steps']} event={info.get('event', 'continue')}"
        )
        time.sleep(speed)


def auto_loop(env: SnakeGameEnv, speed: float, steps: int) -> None:
    env.render()
    for _ in range(steps):
        if env.done:
            break
        obs, reward, done, info = env.step(env.sample_action())
        env.render()
        print(
            f"score={obs['score']} reward={reward:.2f} steps={obs['steps']} event={info.get('event', 'continue')}"
        )
        time.sleep(speed)


def main() -> None:
    args = parse_args()
    config = GameConfig(width=args.width, height=args.height, allow_wrap=args.wrap, seed=args.seed)
    env = SnakeGameEnv(config)
    env.reset()

    if args.auto > 0:
        auto_loop(env, args.speed, args.auto)
    else:
        interactive_loop(env, args.speed)


if __name__ == "__main__":
    main()
