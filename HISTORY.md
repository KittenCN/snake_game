# History

## 0.6.0 - 2026-08-09

- Added a configurable idle-limit floor so large boards do not terminate otherwise safe long-path play before one board-area traversal.
- Added a versioned one-step-survival action mask shared by collection, teacher, evaluation, inference and Double-DQN bootstrap, with a safe all-fatal fallback and canonical terminal masks.
- Added behavior-step policy-anchor annealing from an initial to a final weight after teacher replay warmup, with checkpoint, sidecar, resume and logging identity.
- Made demonstration admission atomic and quality-monotonic at capacity, preventing success trajectories from overwriting elite samples.
- Added configurable multi-step terminal imitation exclusion to keep pre-crash actions out of the successful-action margin objective.
- Added the evidence-backed fixed-12x12 v8 score-improvement profile and regressions for all new MDP, replay and loss contracts.

## 0.5.2 - 2026-08-08

- Added a single-read, optionally SHA-256-authenticated policy checkpoint factory that constructs a fresh target-map agent without restoring optimizer, replay, counters, scaler or RNG state.
- Made cross-map transfer fail closed on unknown network versions and incompatible action/observation schemas, while preserving same-map policy-only compatibility for legacy v1/v2 checkpoints.
- Added auditable source/target map provenance to checkpoint payloads, sidecars and resumed runs, including source sidecar role and digest identity checks.
- Switched inference to policy-only construction for both same-map and cross-map play; target overrides now receive a target-sized episode horizon and report map, horizon and cross-map status.
- Added explicit full-state `restore_training_checkpoint` and retained `load` as its compatibility alias.
- Added 8x8-to-10x10 inference/training, 10x10 resume, invalid target, schema/version conflict and tampered-digest regressions.

## 0.5.1 - 2026-08-07

- Added paired evaluation states for confirmed improvement, confirmed plateau, and inconclusive evidence; inconclusive results now defer every plateau/min-LR/early-stop action.
- Added incremental paired evaluation expansion with disjoint seed chunks, complete maximum-sized warm-start references, and full-suite promotion.
- Applied fixed-look Bonferroni confidence intervals to adaptive evaluation so optional stopping does not reuse an unadjusted 95% interval.
- Upgraded convergence-controller sidecars to schema v3 with v1/v2 migration, adaptive configuration resume gates, and conservative patience-counter cleanup.
- Logged adaptive episode counts, expansion stages, confidence method, statistical state, and patience deferral decisions.
- Kept clear-regression patience independent from LR/plateau patience and extended the log analyzer with adaptive sample/state summaries.

## 0.5.0 - 2026-08-07

- Added a dedicated demonstration replay that atomically promotes complete trajectories only after their raw score and environment return meet auditable success or elite thresholds.
- Added fixed success/elite batch strata with uniform sampling without replacement inside each stratum, preventing high-score experience from being diluted, duplicated or overwritten by the ordinary replay ring.
- Added DQfD-style large-margin successful-action imitation loss alongside TD and immutable policy-anchor losses.
- Persisted all demonstration/imitation hyperparameters and lifetime counters in checkpoints/sidecars, validated them on resume, and exposed replay composition and imitation metrics in JSONL analysis.
- Added overwrite-safe trajectory tokens, atomic capacity rejection, terminal-action imitation masks, unique-sample quota ramping, and unit/end-to-end regressions for promotion, stratified sampling, imitation loss and checkpoint metadata.

## 0.4.4 - 2026-08-07

- Added a frozen warm-start policy teacher and configurable Q-value anchor loss to prevent destructive fine-tuning.
- Added teacher-driven replay bootstrap that defers all gradient updates until the configured transition budget is collected.
- Persisted policy-anchor weights and conservative replay configuration in resumable checkpoints and sidecars.
- Added per-seed evaluation samples, paired confidence-interval best promotion, and clear-regression early stopping without rollback.
- Reduced the recommended mature-policy update/data ratio and documented a distinct immutable-best `stable_v2` server run.

## 0.4.3 - 2026-08-07

- Added a serializable fixed-suite convergence controller with optional plateau LR reductions and minimum-LR-gated early stopping.
- Kept raw mean score as the sole best-checkpoint selector while using the absolute early-stop delta only for significant-improvement patience resets.
- Persisted and verified controller state, base/current learning rates, and safe resume option inheritance/conflict rejection.
- Added mandatory episode-0 fixed-suite baselines and distinct atomic best/latest outputs for authenticated warm starts without modifying their sources.
- Made JSONL diagnostics include standalone baselines and select best evaluations by raw score consistently with training.

## 0.4.2 - 2026-08-07

- Validated native Windows ROCm 7.14 training on an AMD Radeon RX 7600M XT (`gfx1102`).
- Added runtime backend/device/version evidence to training logs and console startup output.
- Made explicit unavailable accelerator requests fail instead of silently falling back to CPU.
- Made checkpoint loading choose the current runtime device rather than inheriting a stale saved-device string.
- Added explicit accelerator cleanup so native Windows ROCm training exits cleanly after saving checkpoints.
- Documented isolated ROCm installation, verification, determinism boundaries, and the tested smoke profile.
- Pinned Ruff to the validated 0.9 minor line after newer releases changed the effective default lint set.
- Removed random food respawn from the idle-limit test so accelerator test runs are deterministic.

## 0.4.1 - 2026-08-06

- Added SHA-validated policy-only warm starts for changing board and training configuration without restoring optimizer, replay, counters or best identity.
- Added strict source/output separation, architecture compatibility checks and persistent warm-start provenance.
- Documented curriculum-stage weight transfer and the difference between resume and warm start.

## 0.4.0 - 2026-08-06

- Added batched persistent-environment collection with configurable rollout and update scheduling.
- Added reusable CUDA pinned staging, non-blocking host-to-device transfers, batched observation/topology processing, and batched action selection.
- Added collection-level throughput and timing diagnostics for environment steps, updates, sampling, GPU host waits, encoding, and action selection.
- Documented Ubuntu foreground/nohup CUDA operation, workload-dependent tuning, and parallel resume semantics.

## 0.3.0 - 2026-08-05

- Fixed the tail-cell occupancy corruption and added environment invariant tests.
- Added relative actions, dynamic idle budgets and reproducible multi-episode seeds.
- Added v3 tail/body-order/horizon observations and a spatial GroupNorm residual network.
- Replaced CUDA tensor replay with a CPU float16 sum-tree prioritized ring buffer and n-step returns.
- Fixed sum-tree rounding drift and upper-bound sampling into padded replay leaves.
- Made NumPy RNG checkpoint state portable to PyTorch versions without uint32 storage support.
- Replaced per-update multiplicative exploration with a behavior-step linear schedule.
- Split latest and best checkpoints, made writes atomic and added linked SHA-256 sidecar validation.
- Added strict resume gates for MDP, seed, v3 state/action, best identity and effective agent settings.
- Removed the best-on-decline rollback loop and added explicit resume exploration reheating.
- Added fixed-suite distributional evaluation and potential-based topology shaping.
- Added a tolerant JSONL training diagnostics CLI.
- Added dependency metadata, 71 tests, Ruff validation and GitHub Actions CI.
- Removed tracked Python bytecode and expanded generated-file ignores.
- Documented the observed plateau evidence, migration path and acceptance gates.
