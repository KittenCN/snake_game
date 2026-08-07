# History

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
