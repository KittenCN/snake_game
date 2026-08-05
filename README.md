# Snake DQN Toolkit / 贪吃蛇 DQN 训练工具集

这是一个基于 PyTorch 的贪吃蛇强化学习项目，包含游戏环境、DQN 训练、固定种子评估、
断点恢复、GUI/控制台推理和 JSONL 训练诊断工具。

当前训练主线是 checkpoint `network_version=3`。旧版 v1/v2 权重仍可加载和播放，但新训练
不会再把旧的 best 权重当成 latest 状态静默恢复。

## 本次平台期诊断

仓库中现有 61,508 条训练记录表明，最近 5,000 回合平均分约为 19.44，最近 1,000 回合
相对前一窗口没有提升；epsilon 在第 272 回合已经到达 0.01。长期继续原循环无法可靠改善。

定位到的根因不是单纯“网络不够大”，而是多项问题叠加：

- 头进入即将腾空的尾格时，旧环境会把新头从 `_occupied` 删除，污染长蛇状态；
- 反向绝对动作实际执行为直行，却以反向动作标签写入 replay；
- epsilon 约 1,500 次更新后就降至 0.01；
- v2 行为和评估期间 Dropout/BatchNorm 仍处于训练模式；
- soft target update 不同步 BatchNorm buffers；
- `resume_best_on_decline` 形成周期性旧模型回滚；
- best/latest checkpoint 混用，外部 metadata 与实际权重版本已经不一致；
- replay 由大量常驻 CUDA tensor 组成，v2 默认容量的裸数据需要数 GiB；
- 观测缺少尾部和身体顺序，不同长蛇状态可能得到同一输入；
- 固定 idle 上限和 500 步硬截断会切断表现最好的长蛇轨迹。

详细证据和验收标准见 [docs/TRAINING_OPTIMIZATION.md](docs/TRAINING_OPTIMIZATION.md)。

## v3 改进

- 三个相对动作：直行、左转、右转，消除反向动作别名；
- 观测增加 tail mask、从头到尾的 body-order channel 和有限时域进度；
- 使用 GroupNorm residual CNN，无 BatchNorm running stats 和 Dropout；
- 3×3 spatial head 保留空间布局，不再只做全局平均池化；
- CPU float16 预分配 ring replay，采样时才传入训练设备；
- 基于 sum-tree 的 O(log N) Prioritized Experience Replay、importance weights 和 3-step return；
- epsilon 按真实行为步数线性衰减，默认 250,000 步从 1.0 到 0.05；
- resume 因 replay 不落盘而显式将 epsilon 回热到默认 0.25；
- potential-based shaping 同时考虑食物距离、可达空间和尾部连通；
- idle 预算随已吃食物数量增长，默认 `90 + 2 * score`；
- 训练截断会以 terminal transition 写入 replay，不再跨 reset bootstrap；
- 固定评估 seed suite，记录均值、标准差、中位数、P10/P90 与终止原因；
- 默认使用确定性后端算法；可用 `--allow-nondeterministic` 显式换取吞吐；
- `latest` 与 `best_eval` 分离、原子保存，并用 SHA-256 将 latest、best 及 sidecar 串成可验证身份；
- 保存 optimizer、AMP scaler、Python/NumPy/Torch RNG 和探索进度；
- replay 不写入 checkpoint，metadata 会明确标记 `replay_restored=false`。

## 安装

推荐 Python 3.10–3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

只运行程序可安装 `requirements.txt`；开发和测试使用 `requirements-dev.txt`。

## 开始新的 v3 训练

```powershell
python train_dqn.py `
  --episodes 10000 `
  --width 12 --height 12 `
  --device cuda
```

关键默认输出：

- `models/dqn_snake_v3_latest.pt`：每次评估/检查点更新，可用于恢复；
- `models/dqn_snake_v3_best.pt`：只在固定评估集的原始策略平均分提升时更新（不使用安全回退）；
- 对应 `.meta.json`：记录 checkpoint SHA-256、episode、架构和训练配置；
- `runs/train_log_<timestamp>.jsonl`：逐回合指标与评估分布。

若机器没有 CUDA，省略 `--device cuda`，程序会自动使用 CPU。

## 恢复训练

默认情况下，只要 `models/dqn_snake_v3_latest.pt` 存在，同一命令会从 latest 恢复：

```powershell
python train_dqn.py --episodes 5000
```

也可显式指定：

```powershell
python train_dqn.py `
  --resume-from models/dqn_snake_v3_latest.pt `
  --episodes 5000
```

使用 `--fresh` 强制开始新模型。若目标路径已有文件，程序会拒绝混写；确认要替换这些精确
输出时需同时使用 `--overwrite-fresh-output`。恢复时会校验 sidecar SHA-256、完整环境配置、
固定评估种子/规模以及 best 文件身份，避免再次出现“旧 best 权重 + 新 episode metadata”的
伪恢复。要有意改变评估基线时使用 `--reset-best-evaluation`。

旧 v1/v2 checkpoint 仅建议用于推理。如果确实需要作为 warm start，必须显式传入
`--resume-from`、`--ignore-resume-metadata`，并为 `--latest-output`、`--output` 指定不含
`v3` 的独立文件名。若奖励、wrap、idle 或时域等环境配置也变化，还必须显式使用
`--allow-environment-change`；这样仍会沿用旧观测和四动作网络，不能获得 v3 的观测与架构优势。

## 推理

GUI：

```powershell
python play_dqn.py --model models/dqn_snake_v3_best.pt --episodes 5
```

控制台：

```powershell
python play_dqn.py `
  --model models/dqn_snake_v3_best.pt `
  --console --render --episodes 5 --seed 42
```

默认安全回退会避开立即碰撞；`--disable-safety-check` 用于评估原始策略。固定 seed 现在会
产生可复现但连续不同的多局序列，不再每局重播完全相同的食物顺序。

## 分析训练日志

```powershell
python analyze_training.py "runs/train_log_*.jsonl"
python analyze_training.py "runs/train_log_*.jsonl" --json
```

工具会报告全局及最近 100/1,000/5,000 回合指标、best/last evaluation、epsilon floor、
loss、终止事件和分数/蛇长分桶。平台期判断是诊断启发式，不是统计学证明。

## 测试

```powershell
python -m pytest -q
python -m ruff check .
```

测试覆盖环境状态不变量、尾格移动、随机多 seed、相对动作、动态 idle、seed 序列、PER、
n-step、epsilon、action mask、target buffer 同步、旧 checkpoint、tail/body-order 观测、
固定评估、有限时域、checkpoint 身份校验、防止 fresh 混写和短训练闭环。GitHub Actions
会运行同样的 pytest 门槛。

## 文件结构

```text
env.py                         游戏环境与相对动作
dqn_agent.py                   v1/v2/v3 网络、replay 与 DQN agent
train_dqn.py                   训练、评估、latest/best checkpoint
play_dqn.py                    GUI/控制台模型推理
analyze_training.py            JSONL 日志诊断
gui.py / cli.py / __main__.py  人工与随机策略入口
tests/                         回归和训练管线测试
docs/TRAINING_OPTIMIZATION.md  平台期证据与架构方案
HISTORY.md                     变更历史
```

`models/`、`runs/` 与 Python 缓存不会提交到 Git。默认不会清理或覆盖已有本地模型和日志；
只有显式 `--overwrite-fresh-output` 会删除两个指定输出及其 sidecar。v3 使用新的默认文件名。

## License

仓库目前尚未声明开源许可证；在许可证明确前，请勿假定拥有再分发授权。
