# History

## 0.3.0 - 2026-08-05

- Fixed the tail-cell occupancy corruption and added environment invariant tests.
- Added relative actions, dynamic idle budgets and reproducible multi-episode seeds.
- Added v3 tail/body-order/horizon observations and a spatial GroupNorm residual network.
- Replaced CUDA tensor replay with a CPU float16 sum-tree prioritized ring buffer and n-step returns.
- Replaced per-update multiplicative exploration with a behavior-step linear schedule.
- Split latest and best checkpoints, made writes atomic and added linked SHA-256 sidecar validation.
- Added strict resume gates for MDP, seed, v3 state/action, best identity and effective agent settings.
- Removed the best-on-decline rollback loop and added explicit resume exploration reheating.
- Added fixed-suite distributional evaluation and potential-based topology shaping.
- Added a tolerant JSONL training diagnostics CLI.
- Added dependency metadata, 66 tests, Ruff validation and GitHub Actions CI.
- Removed tracked Python bytecode and expanded generated-file ignores.
- Documented the observed plateau evidence, migration path and acceptance gates.
