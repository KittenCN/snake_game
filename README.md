# Snake DQN Toolkit / 贪吃蛇 DQN 工具集

## Overview / 项目概览
- **English:** A full reinforcement learning toolkit that pairs a feature-rich Snake environment with a PyTorch Deep Q-Network agent, supporting modern training tricks, advanced observation features, and safe inference.
- **中文：** 这是一个完整的强化学习工具集，结合了功能丰富的贪吃蛇环境与 PyTorch DQN 智能体，内置现代训练技巧、增强的观测特征以及安全推理功能。

## Core Components / 核心组件
1. **Environment (`env.py`)
   - **EN:** Pure-Python Snake simulator with wrap support, idle penalties, safety checks, and numpy-friendly exports.
   - **中文：** 纯 Python 的贪吃蛇环境，支持穿墙、空闲惩罚、安全行动检测及 numpy 观测输出。
2. **Agent (`dqn_agent.py`)
   - **EN:** Switchable CNN backbones (`--network-version`), dueling + double DQN, prioritized observation encoding, and replay experience buffer.
   - **中文：** 可切换的 CNN 主干网络（`--network-version`）、双优势 DQN、强化的观测编码以及经验回放缓存。
3. **Training Pipeline (`train_dqn.py`)
   - **EN:** Segment logging, evaluation checkpoints, optional early stop, train-metric best snapshots, and resume/curriculum utilities.
   - **中文：** 支持分段日志、评估检查点、可选早停、基于训练指标的最佳模型保存，以及断点续训与课程式设置。
4. **Inference Runner (`play_dqn.py`)
   - **EN:** GUI / console play, deterministic or safety-checked control, supports custom devices and seeds.
   - **中文：** GUI 或命令行运行，支持安全控制或纯贪心策略，可自定义设备与随机种子。

## Installation / 安装
```
pip install -r requirements.txt
```
- **EN:** Ensure `torch`, `numpy`, and `tkinter` (for GUI) are available in your environment.
- **中文：** 请确认运行环境中已安装 `torch`、`numpy` 和 `tkinter`（GUI 模式需要）。

## Training Workflow / 训练流程
```
python -m snake_game.train_dqn \
  --episodes 2000 \
  --width 12 --height 12 \
  --network-version 2 \
  --train-best-metric reward \
  --log-dir runs
```
- **EN:**
  - `--network-version`: choose the enhanced residual CNN (`2`) or the legacy CNN (`1`).
  - `--train-best-*`: toggles training-best checkpoints; model files saved as `*_best_reward.pt` or `*_best_score.pt`.
  - Evaluations run every `--eval-interval` episodes; best eval snapshot stored at `--output` and mirrored to history if enabled.
  - Automatic mixed precision (AMP) activates on CUDA devices by default; disable with `--disable-amp` for full-precision training.
  - Logs append to `runs/train_log_<timestamp>.jsonl` for later analysis.
- **中文：**
  - `--network-version`：选择增强残差 CNN（值为 `2`）或传统 CNN（值为 `1`）。
  - `--train-best-*`：根据训练指标保存最佳模型，文件名形如 `*_best_reward.pt` 或 `*_best_score.pt`。
  - 每隔 `--eval-interval` 个 episode 进行评估，最佳评估模型保存在 `--output` 指定路径，并可复制到历史目录。
  - 检测到 CUDA 时默认开启混合精度 (AMP)，可通过 `--disable-amp` 关闭以使用全精度训练。
  - 训练指标实时写入 `runs/train_log_<timestamp>.jsonl` 便于后续分析。

### Resuming / 恢复训练
```
python -m snake_game.train_dqn --output models/dqn_snake.pt --episodes 500 --resume-best-on-decline
```
- **EN:** Metadata (`.meta.json`) restores epsilon, replay progress, best metrics, and train-best checkpoints automatically.
- **中文：** 元数据文件（`.meta.json`）会自动恢复 epsilon、回放进度、最佳评估以及训练最佳模型信息。

## Evaluation & Monitoring / 评估与监控
- **EN:** Check `runs/train_log_*.jsonl` for per-episode rewards, shaped rewards, scores, and evaluation summaries.
- **中文：** 通过 `runs/train_log_*.jsonl` 查看每个 episode 的奖励、塑形奖励、得分以及评估结果。
- **EN:** Optional history snapshots live in `{output}_history/` when `--best-history-limit` > 0.
- **中文：** 若 `--best-history-limit` > 0，可在 `{output}_history/` 目录中找到历史最佳模型。

## Inference / 推理
```
python -m snake_game.play_dqn --model models/dqn_snake.pt --episodes 5
```
- **EN:**
  - GUI mode is default; add `--console --render` for ASCII playback.
  - Safety fallback avoids immediate collisions; disable via `--disable-safety-check` to evaluate raw policy behaviour.
  - Use `--device cuda` to run on GPU, or set `--seed` for reproducible runs.
- **中文：**
  - 默认启动 GUI，使用 `--console --render` 切换为文本模式。
  - 安全回退机制可避免立即撞墙或自撞，可通过 `--disable-safety-check` 关闭以观察原始策略。
  - 通过 `--device cuda` 使用 GPU，`--seed` 保证推理可复现。

## Development Notes / 开发说明
- **EN:**
  - Observation builder now augments raw grids with geometry, direction, distance, and danger maps.
  - Training loop clips gradients, supports soft/hard target updates, and logs per-segment metadata.
  - Checkpoints store network version ensuring legacy compatibility.
- **中文：**
  - 观测编码在原始网格基础上增加了坐标、方向、距离与危险区域特征。
  - 训练循环包含梯度裁剪，支持软/硬目标网络更新，并在每个阶段写入元数据。
  - 模型检查点记录网络版本，保证旧模型可继续使用。

## File Tree / 文件结构
```
├── dqn_agent.py       # DQN agent & observation utilities / DQN 智能体与观测工具
├── train_dqn.py       # Training entry / 训练入口
├── play_dqn.py        # Inference runner / 推理脚本
├── env.py             # Snake environment / 贪吃蛇环境
├── runs/              # Training logs & metadata / 训练日志与元数据
├── models/            # Saved checkpoints / 模型保存目录
└── README.md          # Documentation / 文档
```

## License / 许可
- **EN:** Specify the intended license here (e.g., MIT).
- **中文：** 在此处填写项目的许可协议（例如 MIT）。
