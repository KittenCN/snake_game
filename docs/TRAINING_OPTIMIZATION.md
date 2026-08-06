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

## 并行恢复语义

`--episodes` 表示本次 invocation 要完成的回合数。`episodes_started` 单独记录已分配 seed 的回合：若
进程在多环境 collection 中断，尚未完成的 in-flight seed 不会在恢复时重放，恢复会从其后的 seed
继续分配。恢复时可更改运行时调度参数，例如 `--num-envs`、`--rollout-steps` 和
`--updates-per-collection`；但网络与 replay 的 checkpoint 绑定超参数仍使用既有校验规则，包括网络、
隐藏层、学习率、折扣、batch、replay 容量/门槛、n-step、PER、target 与 epsilon 配置。若显式传入的
值与 checkpoint 不一致，训练会拒绝静默变更，需改为 fresh run。

## 权重迁移与课程阶段

普通 `--resume-from` 用于延续同一训练身份，因此必须保留 checkpoint 绑定的网络、optimizer、
batch、replay 和环境契约。课程训练或吞吐基准需要改变地图、batch、学习率或 replay 时，应改用
`--warm-start-from SOURCE.pt`：新 agent 先按当前 CLI 完整创建，再只加载 source 的 policy 权重并
重新同步 target。optimizer、AMP scaler、replay、n-step、epsilon、计数、seed 流与 best 阈值均从
零开始，旧训练的退化状态不会冒充新阶段的可恢复进度。

迁移前默认校验 source sidecar 与 checkpoint SHA-256；新输出不得覆盖 source。v3 的 3x3 adaptive
pool 允许观测高宽变化，但 network version、action dimension、观测通道、hidden sizes 以及每个
state-dict tensor 的 key/shape/dtype 必须完全兼容。新 latest/best sidecar 保存 warm-start provenance，
后续普通 resume 继续携带该来源。`--ignore-warm-start-metadata` 只为人工确认过的 legacy source
提供显式逃生口，不放宽权重结构检查。

推荐每个课程阶段使用独立输出，例如 `6x6 -> 8x8 -> 10x10 -> 12x12`，并以上一阶段固定评估的
best checkpoint 作为下一阶段 source；不要从 latest 迁移，也不要跨尺寸携带 replay。

## 验收门槛

- 环境不变量通过针对性的尾随测试和随机多 seed 运行。
- 同一输入的重复贪心推理保持确定性。
- 非法/反向动作标签不会进入新的 replay 数据。
- Epsilon 按配置的 frame 调度，并在无 replay 的恢复时明确回热。
- Latest 与 best checkpoint 保持分离，metadata 与 checkpoint 架构、步数、环境、评估套件和 best artifact 身份一致。
- 单元测试覆盖 replay、n-step target、动作掩码、target 同步、截断和固定 seed 评估。
- 确定性短训练 smoke test 无 NaN/Inf，能生成可恢复 latest checkpoint 与独立选出的 best checkpoint。
