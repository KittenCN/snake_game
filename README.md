# Snake Game Environment

This package contains a pure-Python implementation of the Snake game with a reinforcement-learning-friendly API.

```python
from snake_game import SnakeGameEnv, Action, GameConfig

env = SnakeGameEnv(GameConfig(width=10, height=10))
obs = env.reset(seed=123)

for _ in range(100):
    action = env.sample_action()
    obs, reward, done, info = env.step(action)
    if done:
        break

board = env.render(to_string=True)
print(board)
```

## Key API methods

- `reset(seed: Optional[int]) -> observation`
- `step(action) -> (observation, reward, done, info)`
- `legal_actions() -> Tuple[Action, ...]`
- `sample_action() -> Action`
- `observation()` to query the current state without stepping
- `as_numpy()` to obtain an `(H, W, 3)` tensor (requires `numpy`)
- `render(to_string: bool = False)` for ASCII visualisation

Run a random-agent demo:

```bash
python -m snake_game
```

The demo prints the board state after each action; stop the program with `Ctrl+C` when finished.

## PyTorch DQN Agent
## PyTorch DQN Agent

A Deep Q-Network baseline is provided in `snake_game.dqn_agent`.

Train it with:
```bash
python -m snake_game.train_dqn --episodes 1000 --width 12 --height 12
```

Once trained (default checkpoint `models/dqn_snake.pt`), launch GUI inference with all defaults:
```bash
python -m snake_game.play_dqn
```

Use console rendering instead of the GUI:
```bash
python -m snake_game.play_dqn --console --render
```

Install dependencies such as `torch` and `numpy` before training.
