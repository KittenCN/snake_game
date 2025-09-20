"""Tkinter-based GUI for the snake game environment."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional

try:
    from .env import Action, GameConfig, SnakeGameEnv
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from env import Action, GameConfig, SnakeGameEnv


class SnakeGameGUI:
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        *,
        cell_size: int = 25,
        speed_ms: int = 150,
        title: str = "Snake Game",
        master: Optional[tk.Tk] = None,
        controller: Optional[Callable[[SnakeGameEnv], Action]] = None,
        on_episode_end: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config or GameConfig()
        self.env = SnakeGameEnv(self.config)
        self.cell_size = cell_size
        self.speed_ms = max(20, speed_ms)
        self._pending_action: Action = Action.RIGHT
        self._running = False
        self._after_id: Optional[str] = None
        self._controller = controller
        self._on_episode_end = on_episode_end
        self._episode_reward: float = 0.0

        self.root = master or tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        width_px = self.config.width * self.cell_size
        height_px = self.config.height * self.cell_size
        self.canvas = tk.Canvas(
            self.root,
            width=width_px,
            height=height_px,
            bg="#111111",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=(0, 10))
        tk.Button(control_frame, text="Restart", command=self.reset).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Quit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        if self._controller is None:
            self.root.bind("<KeyPress>", self._on_key_press)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.reset()
        self.root.mainloop()

    def reset(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.env.reset()
        self._pending_action = self.env.direction
        self._running = True
        self._episode_reward = 0.0
        self._update_status(reward=0.0, info={})
        self._draw_board()
        self._after_id = self.root.after(self.speed_ms, self._tick)
        if self._controller is None:
            self.root.focus_force()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if not self._running:
            return

        if self._controller is not None:
            action = self._controller(self.env)
        else:
            action = self._pending_action
        if not isinstance(action, Action):
            action = Action(action)

        observation, reward, done, info = self.env.step(action)
        self._episode_reward += reward
        self._draw_board()
        self._update_status(reward=reward, info=info)

        if done:
            self._running = False
            self._after_id = None
            summary = {
                "reward": self._episode_reward,
                "score": observation["score"],
                "steps": observation["steps"],
                "info": info,
            }
            if self._on_episode_end is not None:
                self._on_episode_end(summary)
            if self._controller is None:
                messagebox.showinfo(
                    "Game Over",
                    f"Event: {info.get('event', 'finished')}\nScore: {observation['score']}\nSteps: {observation['steps']}",
                    parent=self.root,
                )
        else:
            self._after_id = self.root.after(self.speed_ms, self._tick)

    def _draw_board(self) -> None:
        self.canvas.delete("cell")
        snake = self.env.snake
        food = self.env.food
        for idx, (x, y) in enumerate(snake):
            x0 = x * self.cell_size
            y0 = y * self.cell_size
            x1 = x0 + self.cell_size
            y1 = y0 + self.cell_size
            color = "#43a047" if idx > 0 else "#1b5e20"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags="cell")
        if food is not None:
            fx0 = food[0] * self.cell_size
            fy0 = food[1] * self.cell_size
            fx1 = fx0 + self.cell_size
            fy1 = fy0 + self.cell_size
            self.canvas.create_oval(fx0, fy0, fx1, fy1, fill="#fdd835", outline="", tags="cell")

    def _update_status(self, *, reward: float, info: dict) -> None:
        self.status_var.set(
            f"Score: {self.env.score}    Steps: {self.env.steps}    Reward: {reward:+.2f}    Event: {info.get('event', 'playing')}"
        )

    def _on_key_press(self, event: tk.Event) -> None:
        keymap = {
            "Up": Action.UP,
            "Right": Action.RIGHT,
            "Down": Action.DOWN,
            "Left": Action.LEFT,
            "w": Action.UP,
            "d": Action.RIGHT,
            "s": Action.DOWN,
            "a": Action.LEFT,
        }
        action = keymap.get(event.keysym) or keymap.get(event.keysym.lower())
        if action is None or self.env.done:
            return
        if action in self.env.legal_actions():
            self._pending_action = action


# ----------------------------------------------------------------------
# CLI helper for manual play
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake game GUI")
    parser.add_argument("--width", type=int, default=20, help="Grid width")
    parser.add_argument("--height", type=int, default=20, help="Grid height")
    parser.add_argument("--wrap", action="store_true", help="Enable wrap-around edges")
    parser.add_argument("--speed", type=int, default=150, help="Tick speed in milliseconds")
    parser.add_argument("--cell-size", type=int, default=25, help="Size of each cell in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(width=args.width, height=args.height, allow_wrap=args.wrap, seed=args.seed)
    gui = SnakeGameGUI(config, cell_size=args.cell_size, speed_ms=args.speed)
    gui.start()


if __name__ == "__main__":
    main()
