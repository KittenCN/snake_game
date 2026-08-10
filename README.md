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
- 支持多环境批量采集、批量观测编码/动作前向，以及 CUDA/ROCm pinned staging 与 non-blocking H2D；
  具体的吞吐分解和 Ubuntu 运行方式见下文及 [docs/TRAINING_OPTIMIZATION.md](docs/TRAINING_OPTIMIZATION.md)。

## 安装

推荐 Python 3.10–3.13：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

只运行程序可安装 `requirements.txt`；开发和测试使用 `requirements-dev.txt`。

### Windows 11 原生 ROCm 7.14

ROCm 版 PyTorch 继续使用 `torch.cuda` API，所以训练参数仍是 `--device cuda`，不能写成
`rocm` 或 `hip`。AMD 官方要求 Windows 11 25H2、匹配的 AMD 驱动和 Python 3.11–3.14；
以下示例针对 RDNA3 `gfx1102`，并使用独立环境保护已有 CPU/CUDA 安装：

```powershell
python -m venv .venv-rocm714
.\.venv-rocm714\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

python -m pip install `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "rocm[libraries,device-gfx1102]==7.14.0"

python -m pip install `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1102]==2.12.0+rocm7.14.0"

python -m pip install -r requirements-dev.txt
rocm-sdk targets
rocm-sdk test
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

验收输出必须同时包含 `+rocm7.14.0`、非空 HIP 版本、`True` 和真实 AMD GPU 名称。
显式传入 `--device cuda` 而后端不可用时，训练会直接失败，不会静默退回 CPU。2026-08-07
已在 RX 7600M XT (`gfx1102`, 8 GiB) 上验证 SDK 19 项自检、GPU FP32/AMP 反向传播、
115 项项目测试和可保存/恢复 checkpoint 的多环境短训练；该移动 SKU 未被 AMD 7.14 硬件表单独点名，
其他驱动组合仍应重复上述真机验收。ROCm 当前对 `adaptive_avg_pool2d_backward` 缺少确定性实现，
默认 `warn_only=True` 会给出警告并继续，跨后端逐位复现不应据此承诺。
0.5.0 的 demonstration/imitation 训练链另在服务器 RTX 3060、PyTorch `2.5.1+cu124` 上通过
158 项测试及 CUDA AMP 真机反向 smoke：elite 轨迹完整晋升、终止动作 imitation mask、分层 batch、
TD loss 与 large-margin imitation loss 均在 GPU 上实际执行。

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
- 对应 `.meta.json`：记录 checkpoint SHA-256、episode、架构、基础/当前学习率及完整收敛控制器状态；
- `runs/train_log_<timestamp>.jsonl`：逐回合、collection、独立 warm-start 基线及评估决策。

若机器没有 CUDA/ROCm，省略 `--device cuda` 可自动使用 CPU；显式指定不可用的加速器会报错。

## CUDA/ROCm 批量训练与 Ubuntu 运行

训练默认保持单环境兼容行为：`--num-envs 1`、`--rollout-steps 1`、
`--updates-per-collection 0`。在 CUDA 或 ROCm 设备上，观测编码和 replay 采样会使用可用的 pinned CPU
staging，并以 non-blocking H2D 传输完整 batch；策略网络会对活动环境的状态执行一次批量动作前向。
这旨在减少小批量传输和逐状态前向的开销，但实际吞吐与 GPU 利用率取决于显卡、CPU、棋盘大小、
batch 以及环境负载，应以 JSONL collection 指标实测为准。

Ubuntu 前台起步配置（在项目根目录执行）：

```bash
mkdir -p runs
PYTHONUNBUFFERED=1 python3 train_dqn.py \
  --episodes 100000 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 32 \
  --batch-size 256 --min-replay 10000 --replay-capacity 100000 \
  --device cuda --allow-nondeterministic
```

Ubuntu 脱离 SSH 会话运行时，保留实时日志并由 `nohup` 接管：

```bash
mkdir -p runs
nohup env PYTHONUNBUFFERED=1 python3 train_dqn.py \
  --episodes 100000 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 32 \
  --batch-size 256 --min-replay 10000 --replay-capacity 100000 \
  --device cuda --allow-nondeterministic \
  > runs/train_cuda_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

`--num-envs` 是持久训练环境数，`--rollout-steps` 是每轮更新前每个活动环境收集的步数；一次
collection 的 transition 数约为 `active_envs * rollout_steps`（尾段活动环境减少时会更少）。
`--updates-per-collection` 显式指定每个 collection 的梯度更新次数，从而直接控制更新/数据比。
值为 `0` 的自动模式保留旧的 `--train-frequency` / `--gradient-steps` 比例；在很大的环境数下它可能
变得计算密集，应先观察指标再调整。更完整的架构、调优和字段说明见
[训练优化说明](docs/TRAINING_OPTIMIZATION.md)。

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

`--episodes` 表示本次调用要完成的回合数，而非要启动的环境数。多环境运行若被中断，尚未完成的
in-flight seed 不会在恢复时重放；恢复会从已记录的 `episodes_started` 之后继续分配 seed。可以在
恢复时调整运行时调度参数（例如 `--num-envs`、`--rollout-steps`、`--updates-per-collection`），但
网络与 replay 相关的 checkpoint 绑定超参数仍遵循既有恢复校验：显式传入且与 checkpoint 不一致时
会拒绝静默变更，需开始新训练。

## 固定评估、降学习率与早停

默认 best 选择仍使用固定 seed 套件的原始 `eval_score_mean`，保持旧训练兼容。成熟策略 warm start
应启用 `--require-paired-promotion`：每次评估保存相同 seed 的逐局分数，并按
`meaningful_delta=max(early_stop_delta, paired_promotion_min_delta)` 做三态判断。Bonferroni 校正后的
配对 CI 下界严格高于 meaningful delta 才是 `confirmed_improvement`；上界严格低于它才是
`confirmed_plateau`；其余为 `inconclusive`，不消耗 plateau/min-LR/early-stop 耐心，也不降 LR 或停训。
`clear_regression` 仍按负向阈值独立判断。
`--early-stop-delta` 定义“显著改善”的绝对分数门槛。`--lr-plateau-patience 0`（默认）完全禁用调度；启用后，连续达到指定
次数的非显著评估会把所有 optimizer 参数组的当前 LR 乘以 `--lr-plateau-factor`，并钳制到
`--lr-plateau-min`。每次降 LR 都清零平台期耐心，不恢复权重、不重建 optimizer，也不清空 replay。

调度启用时，早停不会在降 LR 阶段触发。只有所有参数组已经到达最小 LR 后，再经历
`--early-stop-patience` 次非显著评估才会停止；显著改善会更新参考分数并清零两类耐心。latest 与 best
sidecar 都保存控制器配置/计数以及当前 LR。普通 resume 在这些选项未显式提供时自动恢复；显式冲突
会拒绝启动，sidecar 当前 LR 与 checkpoint optimizer 状态不一致也会失败。完整恢复只接受 `latest`
角色；要以 `best_eval` 为源必须使用 warm start，避免覆盖不可变的 best。

`--regression-stop-patience N` 提供独立的硬退化门槛：若配对差值置信区间上界仍低于负的
`--regression-stop-delta`（即可以排除“只是评估噪声”），连续 N 次后直接停止本次实验。它只停止，
不会恢复历史权重、optimizer 或 replay；immutable best 始终保持不变。

paired 模式可用 `--adaptive-eval-max-episodes` 与 `--adaptive-eval-growth-factor` 从
`--eval-episodes` 基数逐级扩容。每一级只评估新增 seed，不重复前缀；inconclusive 继续扩容，前缀
confirmed improvement 也必须评满 max 后才能晋升，从而保存完整 reference。confirmed plateau 或
clear regression 可提前结束。多次 look 使用固定计划数的 `paired_normal_bonferroni_v1` 校正，避免
把普通 95% CI 用于 optional stopping。warm-start baseline 一次评满 max seed，后续较短 candidate
只与完整 reference 的同长度前缀比较。非 paired 或 max=0 时仍保持单次固定评估行为。

## 仅迁移网络权重

当需要保留已学策略、同时更换地图尺寸、batch、学习率或 replay 配置时，使用
`--warm-start-from`，不要使用普通 `--resume-from`。训练器通过
`DQNAgent.from_policy_checkpoint(...)` 从一次性读取的 checkpoint 字节快照构造目标地图 agent；
网络结构继承 source，目标 `obs_shape`/`GameConfig` 由新地图决定，不恢复 optimizer、replay、AMP、
计数或 RNG。跨图只接受具备明确 spatial-transfer 契约的 v3 relative-action/20-channel checkpoint；
未知 network version、action/observation schema 冲突或 sidecar 身份冲突均会在创建输出前失败。

从 immutable `stable_v5_best` 迁移到 10x10 的推荐命令（使用独立 stable_v6 输出，不覆盖 v4/v5）：

```bash
mkdir -p runs/stable_v6_transfer_10x10
nohup env PYTHONUNBUFFERED=1 /root/miniconda3/bin/python train_dqn.py \
  --episodes 50000 --seed 20260809 \
  --warm-start-from models/dqn_snake_8x8_stable_v5_finetune_best.pt \
  --width 10 --height 10 --initial-length 3 --max-steps 0 \
  --reward-step -0.003 --reward-food 5 --reward-death -5 \
  --reward-shaping-scale 1 --max-idle-steps 90 \
  --idle-growth-per-food 2 --idle-penalty -5 \
  --network-version 3 --hidden 256 256 \
  --gamma 0.99 --n-step 3 --per-alpha 0.6 \
  --per-beta-start 0.4 --per-beta-frames 500000 \
  --target-update 5000 --target-update-tau 0.005 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 8 \
  --batch-size 512 --min-replay 50000 --replay-capacity 100000 \
  --policy-anchor-weight 0.20 --teacher-replay-steps 50000 \
  --demonstration-capacity 20000 --demonstration-batch-fraction 0.20 \
  --elite-demonstration-batch-fraction 0.05 \
  --demonstration-min-score 4 --demonstration-min-return 10 \
  --demonstration-elite-score 7 --demonstration-elite-return 25 \
  --imitation-loss-weight 0.15 --imitation-margin 0.8 \
  --lr 0.0000015625 --lr-plateau-patience 4 \
  --lr-plateau-factor 0.5 --lr-plateau-min 0.00000009765625 \
  --early-stop-patience 8 --early-stop-delta 0.10 \
  --require-paired-promotion --paired-promotion-min-delta 0.10 \
  --regression-stop-patience 3 --regression-stop-delta 0.20 \
  --epsilon-start 0.05 --epsilon-final 0.01 --epsilon-decay-steps 1000000 \
  --eval-interval 400 --eval-episodes 100 \
  --adaptive-eval-max-episodes 600 --adaptive-eval-growth-factor 2 \
  --eval-seed-base 900000 --checkpoint-interval 400 \
  --device cuda --allow-nondeterministic \
  --output models/dqn_snake_10x10_stable_v6_transfer_best.pt \
  --latest-output models/dqn_snake_10x10_stable_v6_transfer_latest.pt \
  --log-dir runs/stable_v6_transfer_10x10 \
  > runs/stable_v6_transfer_10x10/console_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 固定 12x12 的 v8 提分阶段

`stable_v7_transfer_12x12` 的 immutable best 为 `9.432 @ episode 11200`，而训练在 episode 25200
于最小学习率平台期停止。后续阶段不再扩大地图，而是从该 best 做新的 policy-only warm start；不能用
v7 latest 普通 resume，因为新的动作生存掩码和 idle 下限改变了训练/评估身份。推荐完整命令：

```bash
mkdir -p runs/stable_v8_score_12x12
nohup env PYTHONUNBUFFERED=1 /root/miniconda3/bin/python train_dqn.py \
  --episodes 100000 --seed 20260811 \
  --warm-start-from models/dqn_snake_12x12_stable_v7_transfer_best.pt \
  --width 12 --height 12 --initial-length 3 --max-steps 0 \
  --reward-step -0.003 --reward-food 5 --reward-death -5 \
  --reward-shaping-scale 1 --max-idle-steps 90 \
  --idle-growth-per-food 2 --idle-limit-floor-steps 144 --idle-penalty -5 \
  --network-version 3 --hidden 256 256 \
  --action-mask-mode one_step_survival_v1 \
  --gamma 0.99 --n-step 3 --per-alpha 0.6 \
  --per-beta-start 0.4 --per-beta-frames 800000 \
  --target-update 5000 --target-update-tau 0.005 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 10 \
  --batch-size 512 --min-replay 50000 --replay-capacity 100000 \
  --policy-anchor-weight 0.20 --policy-anchor-final-weight 0.03 \
  --policy-anchor-decay-steps 800000 --teacher-replay-steps 50000 \
  --demonstration-capacity 20000 --demonstration-batch-fraction 0.25 \
  --elite-demonstration-batch-fraction 0.10 \
  --demonstration-min-score 8 --demonstration-min-return 30 \
  --demonstration-elite-score 14 --demonstration-elite-return 60 \
  --demonstration-terminal-exclusion-steps 3 \
  --imitation-loss-weight 0.15 --imitation-margin 0.8 \
  --lr 0.00000078125 --lr-plateau-patience 5 \
  --lr-plateau-factor 0.5 --lr-plateau-min 0.000000048828125 \
  --early-stop-patience 10 --early-stop-delta 0.10 \
  --require-paired-promotion --paired-promotion-min-delta 0.10 \
  --regression-stop-patience 3 --regression-stop-delta 0.20 \
  --epsilon-start 0.025 --epsilon-final 0.005 --epsilon-decay-steps 1000000 \
  --eval-interval 400 --eval-episodes 100 \
  --adaptive-eval-max-episodes 600 --adaptive-eval-growth-factor 2 \
  --eval-seed-base 1700000 --checkpoint-interval 400 \
  --device cuda --allow-nondeterministic \
  --output models/dqn_snake_12x12_stable_v8_score_best.pt \
  --latest-output models/dqn_snake_12x12_stable_v8_score_latest.pt \
  --log-dir runs/stable_v8_score_12x12 \
  > runs/stable_v8_score_12x12/console_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

v8 的 episode-0 分数是在新约束身份下重新评估 v7 权重，不应直接与 v7 的 `9.432` 横向比较；之后
best 晋升、学习率和停止决策只比较 v8 内相同 seed、相同掩码和相同 idle 规则的 paired evaluation。
训练器会用 source sidecar 自动认证 checkpoint；启动前可另行核对 v7 best SHA-256 为
`1cf8438c41a0a4b99e424a470779fc99daf9aca1888eef4d9a4f383bf6ed0027`。

### 固定 12x12 的 v9 拓扑提分阶段

服务器 v8 在长局阶段明显变慢：每个训练 episode 的平均行为步数约为 v7 的 2.95 倍，且旧评估路径
每 400 个训练 episode 最多重复执行 600 个固定 seed，评估成本可能超过训练本身。v9 从 v8 的
immutable best 做新的 policy-only warm start，并同时解决后期动作陷阱、评估选择偏差、模糊证据无限
等待和 JSONL 重复膨胀。不要从 v8 latest 做 full resume，也不要与仍在运行的 v8 共用一块 GPU。

先进入新的 tmux 会话，再在仓库根目录执行：

```bash
tmux new -s snake_v9_12x12
cd /root/autodl-tmp/snake_game
mkdir -p runs/stable_v9_topology_12x12
env PYTHONUNBUFFERED=1 /root/miniconda3/bin/python train_dqn.py \
  --episodes 100000 --seed 20260812 \
  --warm-start-from models/dqn_snake_12x12_stable_v8_score_best.pt \
  --width 12 --height 12 --initial-length 3 --max-steps 0 \
  --reward-step -0.003 --reward-food 5 --reward-death -5 \
  --reward-shaping-scale 1 --max-idle-steps 90 \
  --idle-growth-per-food 2 --idle-limit-floor-steps 144 --idle-penalty -5 \
  --network-version 3 --hidden 256 256 \
  --action-mask-mode topology_survival_v1 \
  --gamma 0.99 --n-step 3 --per-alpha 0.6 \
  --per-beta-start 0.4 --per-beta-frames 800000 \
  --target-update 5000 --target-update-tau 0.005 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 8 \
  --batch-size 512 --min-replay 50000 --replay-capacity 100000 \
  --policy-anchor-weight 0.15 --policy-anchor-final-weight 0 \
  --policy-anchor-decay-steps 400000 --teacher-replay-steps 50000 \
  --demonstration-capacity 20000 --demonstration-batch-fraction 0.25 \
  --elite-demonstration-batch-fraction 0.10 \
  --demonstration-min-score 8 --demonstration-min-return 30 \
  --demonstration-elite-score 14 --demonstration-elite-return 60 \
  --demonstration-terminal-exclusion-steps 3 \
  --imitation-loss-weight 0.15 --imitation-margin 0.8 \
  --lr 0.00000078125 --lr-plateau-patience 5 \
  --lr-plateau-factor 0.5 --lr-plateau-min 0.000000048828125 \
  --early-stop-patience 10 --early-stop-delta 0.10 \
  --require-paired-promotion --paired-promotion-min-delta 0.10 \
  --regression-stop-patience 3 --regression-stop-delta 0.20 \
  --epsilon-start 0.015 --epsilon-final 0.003 --epsilon-decay-steps 1000000 \
  --eval-interval 400 --eval-episodes 64 \
  --adaptive-eval-max-episodes 600 --adaptive-eval-growth-factor 2 \
  --eval-seed-base 1700000 \
  --inconclusive-scheduler-mode bounded_probe_v1 \
  --bounded-inconclusive-patience 3 \
  --full-eval-confirmation-interval 8 \
  --full-eval-seed-base 3000000 --full-eval-max-attempts 64 \
  --collection-log-interval 10 --checkpoint-interval 400 \
  --device cuda --allow-nondeterministic \
  --output models/dqn_snake_12x12_stable_v9_topology_best.pt \
  --latest-output models/dqn_snake_12x12_stable_v9_topology_latest.pt \
  --log-dir runs/stable_v9_topology_12x12 \
  2>&1 | tee runs/stable_v9_topology_12x12/console_$(date +%Y%m%d_%H%M%S).log
```

按 `Ctrl-B`、再按 `D` 可安全离开 tmux；`tmux attach -t snake_v9_12x12` 可重新查看。固定的 64-seed
probe 仅用于调度；每 8 次 probe 的 full confirmation 会在全新的 600-seed block 上，用完全相同的 seeds
重新评估 candidate 与 immutable best。每个 full block 会在观察结果前持久预留，并在最多 64 次的预注册
family-wise alpha 预算内判定；崩溃恢复不会复用已经查看过的 holdout。连续 3 次 probe inconclusive 只会在
学习率高于最小值时兑换一个 plateau tick，绝不会在最小学习率上误触发早停。

`topology_survival_v1` 在一步安全之外验证下一步逃生动作和头尾连通性；过严时依次退回两步安全、一步安全，
只有所有动作都会立即死亡时才开放完整动作集合。anchor 从 0.15 完全退火到 0，降低 v8 后期 anchor loss
长期压过 TD loss 的约束；每轮 collection 的梯度更新也从 10 降到 8。普通 JSONL collection 每 10 轮采样
一次，evaluation 和最终状态仍强制记录，完整控制器状态只保存在 checkpoint/sidecar 中。

历史 8x8 adaptive 示例：

```bash
mkdir -p runs/stable_v4_adaptive_8x8
nohup env PYTHONUNBUFFERED=1 /root/miniconda3/bin/python train_dqn.py \
  --episodes 30000 \
  --warm-start-from models/dqn_snake_8x8_stable_v3_latest.pt \
  --width 8 --height 8 \
  --max-idle-steps 70 --idle-growth-per-food 2 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 8 \
  --batch-size 512 --min-replay 50000 --replay-capacity 100000 \
  --policy-anchor-weight 0.25 --teacher-replay-steps 50000 \
  --demonstration-capacity 20000 \
  --demonstration-batch-fraction 0.25 \
  --elite-demonstration-batch-fraction 0.0625 \
  --demonstration-min-score 4 --demonstration-min-return 5 \
  --demonstration-elite-score 6 --demonstration-elite-return 20 \
  --imitation-loss-weight 0.25 --imitation-margin 0.8 \
  --lr 0.000003125 --lr-plateau-patience 4 \
  --lr-plateau-factor 0.5 --lr-plateau-min 0.0000001953125 \
  --early-stop-patience 8 --early-stop-delta 0.10 \
  --require-paired-promotion --paired-promotion-min-delta 0.10 \
  --regression-stop-patience 3 --regression-stop-delta 0.10 \
  --epsilon-start 0.02 --epsilon-final 0.01 --epsilon-decay-steps 600000 \
  --eval-interval 200 --eval-episodes 50 \
  --adaptive-eval-max-episodes 300 --adaptive-eval-growth-factor 2 \
  --eval-seed-base 300000 --checkpoint-interval 200 \
  --device cuda --allow-nondeterministic \
  --output models/dqn_snake_8x8_stable_v4_adaptive_best.pt \
  --latest-output models/dqn_snake_8x8_stable_v4_adaptive_latest.pt \
  --log-dir runs/stable_v4_adaptive_8x8 \
  > runs/stable_v4_adaptive_8x8/console_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

这里从 `stable_v3_latest` 而不是旧 `stable_v3_best` 启动是一次有证据的例外：独立 500-seed
配对留出评估显示 latest 相对旧 best/source 的平均 score 差为 `+0.506`，95% CI 为
`[+0.307, +0.706]`。没有这种独立证据时，仍应从 SHA-256 已验证的 immutable best warm start。

warm start 只迁移 `policy_net` 权重，并用其重新同步 target；启用保守参数后还会冻结同一 policy 作为
teacher。前 `--teacher-replay-steps` 个 transition 使用 teacher 的贪心动作收集 replay，期间不执行任何
梯度更新；之后 TD loss 叠加有效 anchor 权重。`--policy-anchor-final-weight` 与
`--policy-anchor-decay-steps` 可在 teacher 预热后把 anchor 从初值线性退火到终值，前期限制灾难性遗忘、
后期允许策略超过 source；decay 为 0 时保持旧版常量语义。
每个完整 episode 结束后，训练器才按原始游戏 `score` 与未 shaping 的环境 return 联合判定轨迹质量；
同时达到 success 或 elite 两组门槛的完整轨迹会原子复制进独立 demonstration replay。该 replay 不会
被普通经验环形覆盖，batch 按 `--demonstration-batch-fraction` 固定混入成功轨迹，并以
`--elite-demonstration-batch-fraction` 为高分/高回报层保留配额。demo 行为动作通过 DQfD 风格的大间隔
`imitation_loss` 直接约束 Q 排名，最终目标为 `TD + anchor + imitation`，从而让固定评估 score 的成功
行为不再只能经 shaped reward 间接反传。最新策略后续产生的合格完整轨迹也可晋升，形成自举式成功
回放；达到容量后，只有质量元组 `(tier, score, return)` 严格更强的完整轨迹才能原子替换更弱样本，
success 不会覆盖 elite，拒绝的轨迹也不会部分写入。造成终止/截断的最后
`--demonstration-terminal-exclusion-steps` 个动作只保留 TD 监督，不进入 imitation。demo 配额随当前 unique demo 数量
逐步升高且每批无放回采样，不会用一个新 transition 复制填满 batch；低分、低回报、超过 demo 容量
或源 replay 已发生覆盖的轨迹都不会部分写入 demo。
完整 resume 因 replay 不持久化，会自动重新完成这一 teacher 预热，而不是在空 replay 上恢复更新。
预热期固定评估仅记录，不累计学习率、早停或退化耐心。
optimizer、AMP scaler、replay、
n-step 队列、epsilon、行为/学习计数、训练 seed 流以及 best 评估身份都从新配置重新开始。源
checkpoint 默认必须带有 SHA-256 匹配的 sidecar，且源文件不能与新 best/latest 输出同名；
`--ignore-warm-start-metadata` 只用于经过人工确认的旧 checkpoint。网络版本、动作空间、观测
通道和隐藏层必须兼容，但只有具备明确 schema 的 v3 网络允许地图高宽变化。新 checkpoint 与 sidecar
会持续记录 source/target map、`cross_map`、SHA-256、sidecar role、源回合及源/目标观测形状；普通
resume 会保留这些 provenance。每次 warm start 都会在任何训练环境 step 或梯度更新之前运行固定评估套件，
写入 episode 0 的独立 JSONL evaluation，并原子保存全新的 episode-0 best/latest；源 checkpoint 与
sidecar 保持字节不变，因此务必像上例一样使用唯一的新输出名。

旧 v1/v2 checkpoint 仅建议用于推理。如果确实需要作为完整状态的 legacy resume，必须显式传入
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

推理也始终使用 policy-only factory，因此不会恢复训练 RNG/optimizer。显式跨到 10x10：

```powershell
python play_dqn.py `
  --model models/dqn_snake_8x8_stable_v5_finetune_best.pt `
  --width 10 --height 10 --console --episodes 5 --seed 42
```

跨图默认要求 SHA-256 匹配且 `checkpoint_role=best_eval` 的 sidecar；只有命令行 SHA、但没有可证明
best 身份的 sidecar 仍会拒绝。仅诊断 latest/legacy source 可显式加 `--allow-non-best-transfer`。
训练跨图遵循相同 best_eval 默认门槛，只有明确的 `--ignore-warm-start-metadata` 才允许 legacy/non-best
例外并打印警告。未传 `--max-steps` 时，新地图时域为 `width*height*20`，不会继承
源地图较短的截断上限。启动输出会显示 target map、step limit 与 `cross_map`。

默认安全回退会避开立即碰撞；`--disable-safety-check` 用于评估原始策略。固定 seed 现在会
产生可复现但连续不同的多局序列，不再每局重播完全相同的食物顺序。

## 分析训练日志

```powershell
python analyze_training.py "runs/train_log_*.jsonl"
python analyze_training.py "runs/train_log_*.jsonl" --json
```

工具会报告全局及最近 100/1,000/5,000 回合指标、best/last evaluation、epsilon floor、
TD/anchor/imitation loss、demo batch 占比与 success/elite replay 规模、终止事件和分数/蛇长分桶。
平台期判断是诊断启发式，不是统计学证明。
episode-0 独立基线会参与 evaluation 汇总；配对模式还会记录逐 seed 样本、差值置信区间、晋升资格和
明确退化计数。

## 测试

```powershell
python -m pytest -q
python -m ruff check .
```

测试覆盖环境状态不变量、尾格移动、随机多 seed、相对动作、动态 idle/下限、seed 序列、PER、
n-step、epsilon、合法/一步生存 action mask、target buffer 同步、anchor 退火、demo 质量替换、冻结 teacher、旧 checkpoint、tail/body-order 观测、
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
