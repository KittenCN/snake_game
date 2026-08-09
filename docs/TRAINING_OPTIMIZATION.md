# 训练平台期修复与 CUDA 批量训练说明

## 既有训练的证据

最新连续日志覆盖第 17,901 至 28,681 回合；最近 5,000 回合的平均分约为 19.4，而外部
metadata 指向第 12,400 回合的最佳评估。继续原循环不能可靠地改善策略。

平台期来自互相影响的正确性与训练系统问题，而不是单一的网络容量不足：

1. 进入即将腾空的尾格会污染环境占用集合。
2. 反向绝对动作被静默执行为直行，却以请求动作写入 replay，污染动作标签。
3. Epsilon 约 1,500 次梯度更新后到达 0.01 并保持不变。
4. 恢复时 replay 为空，而 epsilon 已耗尽。
5. 评估使用五个不断变化的 seed，且策略仍处于训练模式；BatchNorm 与 Dropout 会使结果含噪并改变模型状态。
6. soft target update 未同步 BatchNorm buffer。
7. `resume_best_on_decline` 反复恢复旧 best，阻断持续进展；best checkpoint 还曾被错误用作 latest。
8. 观测缺少尾部和身体顺序，使不同的后期状态可能无法区分。
9. 固定 90 步食物期限及 500 步回合上限会截断最强的长蛇轨迹。
10. Replay tensor 曾由大量 CUDA allocation 构成；配置的 v2 replay 在 allocator 开销之前已需数 GiB。

## v3 架构方向

版本 3 保留旧 checkpoint 的推理能力，但以三个相对动作（直行、左转、右转）、信息更完整的
空间观测和保留空间布局的 GroupNorm 残差网络训练新策略。有限回合时域属于观测的一部分，故截断
状态相对配置时间上限仍满足 Markov 性。训练使用 CPU ring replay、O(log N) sum-tree 优先采样、
n-step return、动作合法性约束的 Double DQN target，以及按环境行为步数推进的探索调度。

Checkpoint 按用途分离：

- `latest`：每次评估和正常退出写入的可恢复训练状态；
- `best`：只由固定评估套件选出的部署候选模型；
- metadata：从其描述的 checkpoint 导出，若与实际 checkpoint 冲突便会被拒绝；latest metadata 还以
  路径和 SHA-256 关联精确的 best artifact 与评估身份。

评估使用固定 seed 套件，记录分布统计和终止原因；训练与评估共享同一套回合时域规则。

## CUDA 批量采集架构

训练循环维护 `--num-envs` 个持久环境。每一轮 collection 会让每个活动环境运行
`--rollout-steps` 步，故 transition 数近似 `active_envs * rollout_steps`；当最后一轮有环境完成且
不再补充时，实际数会较小。默认值为 `--num-envs 1`、`--rollout-steps 1`、
`--updates-per-collection 0`，因此单环境调用保持兼容。

在 CUDA 上，`BatchObservationEncoder` 为当前与下一状态重用 pinned CPU 观测缓冲。批量状态编码、
批量可达性（将尾部视为可通过的 flood fill）和批量 potential 都在同一批环境上计算；potential 将
食物接近度、可达空间比例和尾部可达性组合为 shaping 信号。动作选择以一个 `select_actions` 调用对
整批状态执行一次网络前向，再逐环境应用 epsilon 探索和合法动作掩码。

Replay 保持 CPU 存储。CUDA 训练采样时会先把已存储字段收集到可复用的 pinned staging tensor，再以
non-blocking H2D 传送 state、next_state、action、reward、done、discount 与 next-action mask；PER
weights 在采样时新计算后直接传输。观测批次同样通过 non-blocking H2D 进入设备。pinned allocation
不可用时实现会回退到 pageable CPU 内存；因此这些路径是减少传输等待的机制，不承诺任何固定速度提升
或 GPU 利用率。

## Ubuntu 用法与调优

在项目根目录运行。前台任务使用未缓冲输出，便于立即观察训练日志：

```bash
mkdir -p runs
PYTHONUNBUFFERED=1 python3 train_dqn.py \
  --episodes 100000 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 32 \
  --batch-size 256 --min-replay 10000 --replay-capacity 100000 \
  --device cuda --allow-nondeterministic
```

需要脱离 SSH 运行时：

```bash
mkdir -p runs
nohup env PYTHONUNBUFFERED=1 python3 train_dqn.py \
  --episodes 100000 \
  --num-envs 32 --rollout-steps 4 --updates-per-collection 32 \
  --batch-size 256 --min-replay 10000 --replay-capacity 100000 \
  --device cuda --allow-nondeterministic \
  > runs/train_cuda_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

`--updates-per-collection` 大于 0 时是每个 collection 明确执行的梯度更新数，可直接控制更新/数据比。
值为 `0` 时进入自动模式，保留既有 `--train-frequency` 与 `--gradient-steps` 的更新/transition 比；
在大 `N` 环境数下，这个旧比例可能让一次 collection 触发大量计算。应从上例的 32/4/32 起步，依据
设备与负载逐项调节，而不是假设并行一定更快。

collection JSONL 记录可用于定位瓶颈：

- `env_steps_per_second`：collection transition 数除以 collection 用时；
- `updates_per_second`：实际成功的更新数除以更新阶段用时；
- `sampling_seconds`：本 collection 所有成功更新的 replay 采样时间之和；
- `gpu_wait_seconds`：为把 TD error、loss、梯度范数和 Q 均值取回主机所经过的既有 CUDA 同步等待之和；
- `encoding_seconds`：当前和下一状态的批量观测编码时间之和；
- `action_selection_seconds`：批量动作选择（含网络前向、掩码和 epsilon 处理）时间之和。

上述指标是工作负载与设备相关的测量值，不是吞吐或 GPU 利用率的保证。

## Windows 原生 ROCm 7.14 验证

PyTorch ROCm 复用 CUDA 设备接口，因此本项目无需维护另一套 `hip` device 分支，启动仍使用
`--device cuda`。训练启动记录现在包含实际 device、后端（`cuda`/`rocm`）、设备名称、Torch、
HIP 和 CUDA build 版本；显式请求不可用的加速器会失败，避免性能验收被静默 CPU 回退污染。

2026-08-07 在 Windows 11 25H2、RX 7600M XT (`gfx1102`, 8 GiB)、ROCm 7.14.0、
PyTorch 2.12.0+rocm7.14.0 和 Python 3.13 上完成：

- `rocm-sdk test`：19/19；
- GPU FP32 matmul backward 与 AMP Conv2d/GroupNorm/Adam backward；
- 项目 pytest：115/115，CPU 与 ROCm 环境均通过；
- 8 个并行环境的 40 回合吞吐 smoke：约 55–62 updates/s；修复后最终 smoke 可生成、加载
  latest/best checkpoint 并干净退出；
- 正常训练结束前显式 GC、设备同步和 allocator cache 释放，避免原生 Windows ROCm 在解释器
  析构阶段长期占用 CPU。

ROCm 会提示 `adaptive_avg_pool2d_backward_cuda` 尚无确定性实现。当前训练使用
`torch.use_deterministic_algorithms(..., warn_only=True)`，因此警告不会中断训练，但 ROCm 与其他
后端之间不具备逐位一致保证。`xnack 'Off'` 提示来自不支持 XNACK 的 `gfx1102` 设备库，本次验证中
未影响前向、反向、保存或恢复。

## 并行恢复语义

`--episodes` 表示本次 invocation 要完成的回合数。`episodes_started` 单独记录已分配 seed 的回合：若
进程在多环境 collection 中断，尚未完成的 in-flight seed 不会在恢复时重放，恢复会从其后的 seed
继续分配。恢复时可更改运行时调度参数，例如 `--num-envs`、`--rollout-steps` 和
`--updates-per-collection`；但网络与 replay 的 checkpoint 绑定超参数仍使用既有校验规则，包括网络、
隐藏层、学习率、折扣、batch、replay 容量/门槛、n-step、PER、target 与 epsilon 配置。若显式传入的
值与 checkpoint 不一致，训练会拒绝静默变更，需改为 fresh run。

## 可恢复的收敛控制器

训练循环使用显式、可序列化的 `EvaluationConvergenceController`。默认模式仍以固定 seed 套件的原始
`eval_score_mean` 保持旧行为；保守 warm start 使用 `--require-paired-promotion`，保存每个 seed 的
分数并计算候选与当前 best 的配对差值。有效改善门槛为
`max(early_stop_delta, paired_promotion_min_delta)`：校正 CI 下界严格高于门槛才是
`confirmed_improvement`，上界严格低于门槛才是 `confirmed_plateau`，相等或区间跨越门槛均为
`inconclusive`。inconclusive 不增加 plateau/min-LR/early-stop 计数，不降 LR，也不停训；
`clear_regression` 仍是独立的负向硬保护。

paired 模式可把 `--eval-episodes` 作为 base，并用 `--adaptive-eval-max-episodes` 与
`--adaptive-eval-growth-factor` 建立固定 look 计划。每一 look 只运行新增 seed chunk，再合并前缀样本；
inconclusive 会继续扩，前缀 confirmed improvement 也会继续到 max，只有 confirmed plateau 或 clear
regression 可以提前结束。为控制 optional stopping 的 family-wise error，每次比较使用
`paired_normal_bonferroni_v1`：固定 `alpha=0.05`，每 look 使用 `alpha/num_looks`，临界值由标准库
`NormalDist.inv_cdf(1-alpha_each/2)` 计算。日志保存 method、family/look confidence、planned looks、
实际/计划/max episode、扩容 stage、三态和 `patience_deferred`。

`--lr-plateau-patience 0` 禁用调度并保持旧早停行为。大于零时，每累计相应次数的非显著评估，所有
optimizer 参数组当前 LR 都乘以 `--lr-plateau-factor`，再钳制到 `--lr-plateau-min`；降 LR 后平台期
计数清零。这个动作只改 optimizer LR，不恢复历史权重、不重置 optimizer/AMP、不清 replay/n-step。
到达最小 LR 的那次降幅不计入早停耐心；必须在最小 LR 上再累计 `--early-stop-patience` 次非显著
评估才停止。`--regression-stop-patience` 是更快的独立保护：配对差值 CI 上界低于负的
`--regression-stop-delta` 才计一次明确退化，连续达到耐心后停止，不执行任何模型/训练状态回滚。

latest 与 best sidecar 同时保存控制器版本、配置、均值/逐 seed reference、三类耐心计数、降幅/评估次数、基础
学习率及 optimizer 各参数组当前 LR。resume 先校验 sidecar 当前 LR 与 checkpoint optimizer state，
再恢复控制器；CLI 未显式给出的调度/早停选项继承 sidecar，显式冲突则安全失败。旧 sidecar 没有
控制器时会明确警告，以已保存 best 为 reference、所有计数为零初始化。controller schema v3 可读
v1/v2；由于旧 paired CI 可能已让 inconclusive 错误消耗耐心，迁移会保留实际 LR/reduction 历史但
清零 plateau、min-LR 和 regression patience，并记录 migration note。基础 `--lr` 只与 checkpoint
的初始 `agent.lr` 比较，不能把平台期降低后的当前 LR 错当成基础配置冲突。完整 `--resume-from` 只
接受 `latest` 角色；`best_eval` 应作为 `--warm-start-from` 的不可变策略源。

## 权重迁移与课程阶段

普通 `--resume-from` 用于延续同一训练身份，因此必须保留 checkpoint 绑定的网络、optimizer、
batch、replay 和环境契约。课程训练或吞吐基准需要改变地图、batch、学习率或 replay 时，应改用
`--warm-start-from SOURCE.pt`：`DQNAgent.from_policy_checkpoint(...)` 从一次性读取并可选 SHA-256
认证的 source 字节快照直接构造目标地图 agent；网络结构继承 source，目标 `GameConfig`/`obs_shape`
来自新地图，并只加载 policy 权重、重新同步 target。optimizer、AMP scaler、replay、n-step、epsilon、计数、seed 流与 best 阈值均从
零开始，旧训练的退化状态不会冒充新阶段的可恢复进度。

迁移前默认校验 source sidecar 与 checkpoint SHA-256；新输出不得覆盖 source。v3 的 3x3 adaptive
pool 允许观测高宽变化，但跨图会严格校验已知 network version、relative-action schema、20-channel
observation schema、hidden sizes 以及每个 state-dict tensor 的 key/shape/dtype；未知/未来 version
fail closed。same-map policy-only factory 仍按 checkpoint 实际 action/obs 形状兼容 v1/v2。
新 checkpoint 与 latest/best sidecar 保存 source/target map、`cross_map`、source sidecar role/SHA 及 warm-start provenance。
跨图训练默认要求 SHA 匹配且 role 为 `best_eval` 的 source sidecar；只有显式
`--ignore-warm-start-metadata` 才允许 intentional legacy/non-best 例外并输出警告。
后续普通 resume 继续携带该来源。`--ignore-warm-start-metadata` 只为人工确认过的 legacy source
提供显式逃生口，不放宽权重结构检查。

通过验证的 warm start 在启动训练环境与任何梯度更新之前，先运行完整固定评估套件。该结果以
`record_type=evaluation`、`evaluation_kind=warm_start_baseline`、`episode=0` 独立写入 JSONL，初始化
控制器完整 reference，并分别原子保存新运行的 episode-0 best 与 latest。启用 adaptive paired 时，
baseline 一次运行完整 max seed；candidate 可以用较短前缀比较，但晋升时 reference 必须仍为完整 max。
源 checkpoint/sidecar 不会被
修改；即使跨地图迁移，也必须为新阶段指定两个互不相同且不与源重叠的输出路径。

`--teacher-replay-steps` 会冻结 warm-start policy 作为 teacher，先用其贪心动作采集指定数量的
transition，且在完成前强制 `collection_update_attempts=0`。之后标准 TD loss 叠加冻结 teacher 的
全动作 Q 值 Smooth-L1 anchor loss；teacher 权重、权重系数和预热步数进入 checkpoint，完整 resume
可审计恢复。由于 replay 本来就不进入 checkpoint，每次完整 resume 都会从空 replay 重新执行 teacher
预热，而不是直接用刚恢复的网络开始更新。预热期评估仍写入日志，但标记为
`teacher_replay_warmup`，不消耗 LR/早停/退化耐心。它不会把历史 best 回滚进 policy，也不会改变
immutable source。

### 固定评估目标对齐：demonstration replay 与 imitation loss

potential shaping 只保证在理想无限数据/精确求解条件下不改变最优策略；有限容量网络、n-step、PER、
截断和小学习率共同存在时，TD loss 优化的是 shaped return，并不直接保证固定评估的原始 score 上升。
因此训练器在完整 episode 终止后，用原始 `score` 与未 shaping 的环境 return 联合分层：同时超过
success 门槛的轨迹晋升为 tier 1，同时超过更高 score/return 门槛的晋升为 elite tier 2。未完整结束、
只满足一个门槛或 replay slot 已被覆盖的轨迹不会部分晋升。

晋升轨迹原子复制到独立的 `demonstration_replay`，因此普通 replay 环形覆盖不会清除成功监督。
每个 learner batch 保留固定 demo 配额，并在 demo 内按 success/elite 配额分层、在每层内部均匀
无放回采样；普通 replay 继续使用 PER。demo action 使用 DQfD large-margin objective：正确行为的 Q 必须至少比其他动作高
`--imitation-margin`；elite 样本权重为普通 success 的 1.5 倍。总损失为：

`TD loss + policy_anchor_weight * anchor_loss + imitation_loss_weight * imitation_loss`

这不是把固定评估 score 当作逐 transition reward，也不修改环境 MDP；它把已经由完整原始 score/return
验证的成功行为作为额外监督，直接弥合 TD shaped-return 与固定评估 score 的有限样本偏差。teacher
预热首先提供稳定 demo，之后当前策略自己的合格高分完整轨迹也能晋升并替换 demo 环中的较旧样本。
终止/截断动作保留在完整 demo 轨迹中用于 TD，但通过独立 eligibility mask 排除在 imitation 外，避免
负终局 TD 与成功动作 margin 发生梯度冲突。demo 采用无放回采样；unique transition 不足时只使用当前
可用数量，其余 batch 回填普通 PER，随 demo 积累逐步达到目标配额。demo 不套用普通 PER importance
公式，避免在优先级无放回抽样中使用错误的 inclusion probability；每个入选 demo 的 TD 权重均为 1。
demo replay 到达容量后会先对完整轨迹做准入预检：只有质量元组 `(tier, score, return)` 严格高于足够
数量的现有样本时才整条替换，因此 success 不能覆盖 elite，失败准入也不会造成部分写入。
`--demonstration-terminal-exclusion-steps` 可把终点前多个高风险动作排除在 imitation 外，而不是只排除
最后一步。checkpoint 不保存两个大 replay；完整 resume 会从冻结 teacher 重新采集并重建 demo，所有门槛、batch
占比、imitation 参数及累计晋升计数仍进入 checkpoint/sidecar 供身份校验与审计。

### 固定 12x12 的收敛后提分

服务器 `stable_v7_transfer_12x12` 在 episode 11200 得到 fixed-suite best `9.432`，随后在 episode
25200 以 `min_lr_early_stop_patience` 停止。独立 1000-seed 留出评估中，v7 best 平均分为 `9.191`
（95% CI `[8.899, 9.483]`），明显优于 v6 source 的 `5.336`，也优于 v7 latest 的 `8.523`；因此
下阶段必须从 immutable v7 best warm start，而不是恢复退化的 latest。

后期继续提高分数的四个约束修复为：

- `idle_limit_floor_steps=144`：12x12 至少允许一个棋盘面积量级的无食物移动窗口；旧 checkpoint 缺省
  为 0，且禁用 idle 时不能单独设置 floor。
- `one_step_survival_v1`：在合法相对动作中优先保留不会下一步立即撞墙/撞身的动作；若所有动作都会
  立即死亡则回退完整合法集合，避免全 false mask 产生无效 TD target。行为采集、teacher、固定评估、
  推理和非终止 next-state target 共用同一 helper，终止 transition 的 next mask 规范化为全 true。
- anchor 从 `0.20` 在 teacher replay 完成后按 behavior steps 线性退火到 `0.03`，避免固定 anchor 在
  后期永久把策略束缚在 source Q 值附近。有效权重及 schedule 状态进入 checkpoint/sidecar 和日志。
- demo 保留高质量整轨迹、排除终点前 3 个动作的 imitation，降低自撞/撞墙前连续危险动作与负 TD
  target 的梯度冲突。

这些参数改变了动作约束与 idle MDP 身份，因此应新建 `stable_v8_score_12x12`，不得 full resume v7。
README 中的完整命令使用新 seed base 1700000 和全新输出。episode-0 baseline 是 v7 权重在 v8 身份下
的重新评估；v8 只在自身相同评估身份内做 paired promotion、降学习率和停止决策。

从 immutable `stable_v5_best` 迁移到 10x10 的新推荐命令：

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

这条命令使用 adaptive paired base=100、max=600、growth=2 与未被 v5 使用的新 seed base=900000，
并使用全新的 stable_v6 目标，不会覆盖 v4/v5 artifact。相对 v5 的 8x8 微调，它提高探索率和
update ratio、降低 anchor 强度，并下调 demonstration 门槛以适应 10x10 初期较低的 score 分布；
源应为 SHA-256 sidecar 已验证且角色为 `best_eval` 的 immutable v5 best。

历史 `stable_v3_latest` 8x8 adaptive 命令：

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
  --lr 0.000003125 \
  --lr-plateau-patience 4 --lr-plateau-factor 0.5 \
  --lr-plateau-min 0.0000001953125 --early-stop-patience 8 \
  --early-stop-delta 0.10 \
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

这组值直接针对服务器 `stable_v2` 的证据：episode-0 固定评估为 3.54，学习期间虽比 v1 稳定，
但 active evaluation 最高仅 3.38，说明 anchor 成功防止了大幅遗忘，却没有提供超越 immutable best 的
直接监督。v3 保留低 LR、低 update/data ratio、teacher 和配对保护，同时用 25% demo batch（其中
6.25% 为 elite）把原始 score/return 验证过的动作加入优化目标；LR 平台期与最小 LR 后耐心也适度拉长。
本次新阶段明确以 `dqn_snake_8x8_stable_v3_latest.pt` 作 policy-only warm start，并使用独立 stable_v4
输出；不会覆盖 stable_v3 源 checkpoint/sidecar，也不会把旧 optimizer/replay/耐心带入新运行。
这是一次有证据的例外：独立 500-seed 配对留出评估显示 latest 相对旧 best/source 的平均 score 差为
`+0.506`，95% CI 为 `[+0.307, +0.706]`。没有这种独立证据时，仍应从 SHA-256 已验证的
immutable best warm start。

推荐每个课程阶段使用独立输出，例如 `6x6 -> 8x8 -> 10x10 -> 12x12`，并以上一阶段固定评估的
best checkpoint 作为下一阶段 source；不要从未经独立配对证据验证的 latest 迁移，也不要跨尺寸携带 replay。

## 验收门槛

- 环境不变量通过针对性的尾随测试和随机多 seed 运行。
- 同一输入的重复贪心推理保持确定性。
- 非法/反向动作标签不会进入新的 replay 数据。
- Epsilon 按配置的 frame 调度，并在无 replay 的恢复时明确回热。
- Latest 与 best checkpoint 保持分离，metadata 与 checkpoint 架构、步数、环境、评估套件和 best artifact 身份一致。
- 单元测试覆盖 replay、n-step target、动作掩码、target 同步、截断和固定 seed 评估。
- 保守 warm start 覆盖冻结 teacher、零更新 replay 预热、anchor checkpoint 恢复、配对晋升与退化停止。
- demonstration 回归覆盖完整轨迹原子晋升、覆盖版本保护、success/elite 分层采样、large-margin imitation 和 resume 参数身份。
- 12x12 提分回归覆盖 idle floor、一步生存动作掩码/all-fatal 回退、terminal next mask、anchor 退火恢复、
  demo 严格质量替换与多步 terminal imitation 排除。
- 确定性短训练 smoke test 无 NaN/Inf，能生成可恢复 latest checkpoint 与独立选出的 best checkpoint。
