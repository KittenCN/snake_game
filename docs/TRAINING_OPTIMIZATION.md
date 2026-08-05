# Training plateau remediation plan

## Evidence from the existing run

The latest continuous log covers episodes 17,901 through 28,681. Its recent
5,000-episode mean score remains around 19.4, while the external metadata says
the best evaluation was reached at episode 12,400. Continuing the same loop is
therefore not expected to improve the policy reliably.

The plateau is caused by interacting correctness and training-system issues,
not by a single undersized neural network:

1. Moving into the departing tail cell corrupts the environment occupancy set.
2. Opposite absolute actions are silently executed as straight moves but stored
   under the requested action, corrupting replay labels.
3. Epsilon reaches 0.01 after roughly 1,500 gradient updates and stays there.
4. Resume starts with an empty replay buffer and the already exhausted epsilon.
5. Evaluation uses five changing seeds while the policy remains in training
   mode; BatchNorm and Dropout make the result noisy and mutate model state.
6. Soft target updates omit BatchNorm buffers.
7. `resume_best_on_decline` repeatedly restores an old best model, preventing
   sustained progress; the best checkpoint is also incorrectly used as latest.
8. The observation omits the tail and body order, so distinct late-game states
   can be indistinguishable.
9. A fixed 90-step food deadline and a 500-step episode cap cut off the strongest
   long-snake trajectories.
10. Replay tensors are stored as many CUDA allocations; the configured v2 replay
    would require several GiB before allocator overhead.

## Architecture direction

Version 3 keeps legacy checkpoint inference but trains a new policy with three
relative actions (straight, left, right), a Markov-richer spatial observation,
and a GroupNorm residual network that preserves spatial layout. The finite
episode horizon is part of the observation, so truncated states remain Markov
with respect to the configured time limit. Training uses a
CPU ring replay buffer, O(log N) sum-tree prioritized sampling, n-step returns, legal-action-aware
Double DQN targets, and an environment-step exploration schedule.

Checkpoints are split by purpose:

- `latest`: complete resumable training state written at every evaluation and
  normal shutdown;
- `best`: deployment candidate selected only by a fixed evaluation suite;
- metadata: derived from the checkpoint being described and rejected when it
  conflicts with the actual checkpoint; latest metadata also links the exact
  best artifact and evaluation identity by path and SHA-256.

Evaluation uses a fixed seed suite and records distribution statistics and
terminal causes. Training and evaluation share the same episode horizon rules.

## Acceptance gates

- Environment invariants hold under targeted tail-following tests and randomized
  multi-seed runs.
- Repeated greedy inference on the same input is deterministic.
- Illegal/opposite action labels never enter new replay data.
- Epsilon follows the configured frame schedule and reheats explicitly on a
  replay-less resume.
- Latest and best checkpoints remain distinct and metadata matches checkpoint
  architecture, step, environment, evaluation-suite and best-artifact identity.
- Unit tests cover replay, n-step targets, action masking, target synchronization,
  truncation and fixed-seed evaluation.
- A deterministic short training smoke test learns without NaN/Inf values and
  produces a resumable latest checkpoint plus an independently selected best one.
