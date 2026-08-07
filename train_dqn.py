"""Train a reproducible DQN agent to play Snake.

Version 3 deliberately separates resumable ``latest`` checkpoints from evaluated
``best`` checkpoints.  Legacy v1/v2 checkpoints remain loadable when supplied via
``--resume-from``, but new training uses a three-action relative control space.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .batch_processing import BatchObservationEncoder, batch_state_potentials
    from .dqn_agent import DQNAgent, flatten_observation
    from .env import Action, GameConfig, RelativeAction, SnakeGameEnv
except ImportError:
    from batch_processing import BatchObservationEncoder, batch_state_potentials
    from dqn_agent import DQNAgent, flatten_observation
    from env import Action, GameConfig, RelativeAction, SnakeGameEnv


CHECKPOINT_FORMAT = 3
V3_OBSERVATION_CHANNELS = 20
CONVERGENCE_CONTROLLER_VERSION = 2


@dataclass
class EvaluationConvergenceController:
    """Serializable fixed-suite convergence policy.

    Legacy runs use raw fixed-suite mean score. Conservative warm starts can also
    require a positive paired-seed confidence interval and stop on clear regression.
    """

    lr_plateau_patience: int
    lr_plateau_factor: float
    lr_plateau_min: float
    early_stop_patience: int
    early_stop_delta: float
    require_paired_promotion: bool = False
    paired_promotion_min_delta: float = 0.0
    regression_stop_patience: int = 0
    regression_stop_delta: float = 0.0
    reference_score: float | None = None
    reference_scores: list[float] | None = None
    plateau_evaluations: int = 0
    min_lr_evaluations: int = 0
    regression_evaluations: int = 0
    reductions: int = 0
    evaluations: int = 0

    @property
    def scheduler_enabled(self) -> bool:
        return self.lr_plateau_patience > 0

    def reset(self, reference_score: float | None = None) -> None:
        self.reference_score = reference_score
        self.reference_scores = None
        self.plateau_evaluations = 0
        self.min_lr_evaluations = 0
        self.regression_evaluations = 0
        self.reductions = 0
        self.evaluations = 0

    def set_paired_reference(self, scores: Sequence[float | int]) -> None:
        parsed = [float(value) for value in scores]
        if not parsed or not all(math.isfinite(value) for value in parsed):
            raise ValueError("Paired reference scores must be non-empty and finite")
        self.reference_scores = parsed
        self.regression_evaluations = 0

    def paired_comparison(
        self, scores: Sequence[float | int] | None
    ) -> dict[str, Any] | None:
        if scores is None or self.reference_scores is None:
            return None
        candidate = [float(value) for value in scores]
        if len(candidate) != len(self.reference_scores):
            raise ValueError(
                "Paired evaluation sample count changed: "
                f"reference={len(self.reference_scores)}, candidate={len(candidate)}"
            )
        if not candidate or not all(math.isfinite(value) for value in candidate):
            raise ValueError("Paired candidate scores must be non-empty and finite")
        differences = [
            current - reference
            for current, reference in zip(candidate, self.reference_scores, strict=True)
        ]
        mean = float(statistics.mean(differences))
        std = float(statistics.stdev(differences)) if len(differences) > 1 else 0.0
        margin = 1.96 * std / math.sqrt(len(differences))
        ci95_low = mean - margin
        ci95_high = mean + margin
        return {
            "count": len(differences),
            "mean_delta": mean,
            "std_delta": std,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "min_delta": min(differences),
            "max_delta": max(differences),
            "promotion_eligible": (
                mean >= self.paired_promotion_min_delta and ci95_low > 0.0
            ),
            "clear_regression": ci95_high < -self.regression_stop_delta,
        }

    @staticmethod
    def learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
        return [float(group["lr"]) for group in optimizer.param_groups]

    def at_min_lr(self, optimizer: torch.optim.Optimizer) -> bool:
        tolerance = max(1e-15, abs(self.lr_plateau_min) * 1e-12)
        return all(
            lr <= self.lr_plateau_min + tolerance
            for lr in self.learning_rates(optimizer)
        )

    def observe(
        self,
        score: float,
        optimizer: torch.optim.Optimizer,
        *,
        sample_scores: Sequence[float | int] | None = None,
        defer_reason: str | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(score):
            raise ValueError("Evaluation score must be finite")
        before_reference = self.reference_score
        before_lrs = self.learning_rates(optimizer)
        paired = self.paired_comparison(sample_scores)
        aggregate_significant = (
            before_reference is None
            or score >= before_reference + self.early_stop_delta
        )
        if defer_reason is not None:
            return {
                "score": score,
                "decision": defer_reason,
                "observation_deferred": True,
                "significant_improvement": False,
                "aggregate_significant_improvement": aggregate_significant,
                "paired_comparison": paired,
                "paired_promotion_eligible": False,
                "clear_regression": False,
                "regression_evaluations": self.regression_evaluations,
                "lr_reduced": False,
                "should_stop": False,
                "reference_score_before": before_reference,
                "reference_score": self.reference_score,
                "learning_rates_before": before_lrs,
                "learning_rates": before_lrs,
                "at_min_lr": self.at_min_lr(optimizer),
                "plateau_evaluations": self.plateau_evaluations,
                "min_lr_evaluations": self.min_lr_evaluations,
                "reductions": self.reductions,
                "evaluations": self.evaluations,
            }
        self.evaluations += 1
        significant = aggregate_significant and (
            before_reference is None
            or not self.require_paired_promotion
            or bool(paired and paired["promotion_eligible"])
        )
        clear_regression = bool(paired and paired["clear_regression"])
        if clear_regression:
            self.regression_evaluations += 1
        else:
            self.regression_evaluations = 0
        regression_stop = (
            self.regression_stop_patience > 0
            and self.regression_evaluations >= self.regression_stop_patience
        )
        reduced = False
        stop = False
        decision = "significant_improvement"

        if significant:
            self.reference_score = (
                score if before_reference is None else max(before_reference, score)
            )
            self.plateau_evaluations = 0
            self.min_lr_evaluations = 0
        elif regression_stop:
            decision = "paired_regression_patience"
            stop = True
        elif not self.scheduler_enabled:
            self.plateau_evaluations += 1
            decision = "early_stop_patience"
            stop = (
                self.early_stop_patience > 0
                and self.plateau_evaluations >= self.early_stop_patience
            )
        elif self.at_min_lr(optimizer):
            self.min_lr_evaluations += 1
            decision = "min_lr_early_stop_patience"
            stop = (
                self.early_stop_patience > 0
                and self.min_lr_evaluations >= self.early_stop_patience
            )
        else:
            self.plateau_evaluations += 1
            decision = "lr_plateau_patience"
            if self.plateau_evaluations >= self.lr_plateau_patience:
                for group in optimizer.param_groups:
                    group["lr"] = max(
                        self.lr_plateau_min,
                        float(group["lr"]) * self.lr_plateau_factor,
                    )
                self.reductions += 1
                self.plateau_evaluations = 0
                self.min_lr_evaluations = 0
                reduced = True
                decision = "lr_reduced"

        after_lrs = self.learning_rates(optimizer)
        return {
            "score": score,
            "decision": decision,
            "observation_deferred": False,
            "significant_improvement": significant,
            "aggregate_significant_improvement": aggregate_significant,
            "paired_comparison": paired,
            "paired_promotion_eligible": bool(
                paired and paired["promotion_eligible"]
            ),
            "clear_regression": clear_regression,
            "regression_evaluations": self.regression_evaluations,
            "lr_reduced": reduced,
            "should_stop": stop,
            "reference_score_before": before_reference,
            "reference_score": self.reference_score,
            "learning_rates_before": before_lrs,
            "learning_rates": after_lrs,
            "at_min_lr": self.at_min_lr(optimizer),
            "plateau_evaluations": self.plateau_evaluations,
            "min_lr_evaluations": self.min_lr_evaluations,
            "reductions": self.reductions,
            "evaluations": self.evaluations,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CONVERGENCE_CONTROLLER_VERSION,
            "config": {
                "lr_plateau_patience": self.lr_plateau_patience,
                "lr_plateau_factor": self.lr_plateau_factor,
                "lr_plateau_min": self.lr_plateau_min,
                "early_stop_patience": self.early_stop_patience,
                "early_stop_delta": self.early_stop_delta,
                "require_paired_promotion": self.require_paired_promotion,
                "paired_promotion_min_delta": self.paired_promotion_min_delta,
                "regression_stop_patience": self.regression_stop_patience,
                "regression_stop_delta": self.regression_stop_delta,
            },
            "state": {
                "reference_score": self.reference_score,
                "reference_scores": self.reference_scores,
                "plateau_evaluations": self.plateau_evaluations,
                "min_lr_evaluations": self.min_lr_evaluations,
                "regression_evaluations": self.regression_evaluations,
                "reductions": self.reductions,
                "evaluations": self.evaluations,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationConvergenceController:
        version = payload.get("version")
        if version not in {1, CONVERGENCE_CONTROLLER_VERSION}:
            raise RuntimeError(
                "Unsupported convergence controller sidecar version: "
                f"{payload.get('version')!r}"
            )
        config = payload.get("config")
        state = payload.get("state")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise RuntimeError("Convergence controller sidecar is missing config/state")
        controller = cls(
            lr_plateau_patience=int(config["lr_plateau_patience"]),
            lr_plateau_factor=float(config["lr_plateau_factor"]),
            lr_plateau_min=float(config["lr_plateau_min"]),
            early_stop_patience=int(config["early_stop_patience"]),
            early_stop_delta=float(config["early_stop_delta"]),
            require_paired_promotion=bool(
                config.get("require_paired_promotion", False)
            ),
            paired_promotion_min_delta=float(
                config.get("paired_promotion_min_delta", 0.0)
            ),
            regression_stop_patience=int(
                config.get("regression_stop_patience", 0)
            ),
            regression_stop_delta=float(config.get("regression_stop_delta", 0.0)),
            reference_score=(
                None
                if state.get("reference_score") is None
                else float(state["reference_score"])
            ),
            reference_scores=(
                None
                if state.get("reference_scores") is None
                else [float(value) for value in state["reference_scores"]]
            ),
            plateau_evaluations=int(state.get("plateau_evaluations", 0)),
            min_lr_evaluations=int(state.get("min_lr_evaluations", 0)),
            regression_evaluations=int(state.get("regression_evaluations", 0)),
            reductions=int(state.get("reductions", 0)),
            evaluations=int(state.get("evaluations", 0)),
        )
        controller.validate()
        return controller

    def validate(self) -> None:
        if self.lr_plateau_patience < 0 or self.early_stop_patience < 0:
            raise RuntimeError("Controller patience values must be non-negative")
        if not 0.0 < self.lr_plateau_factor < 1.0:
            raise RuntimeError("Controller LR plateau factor must be between 0 and 1")
        if not math.isfinite(self.lr_plateau_min) or self.lr_plateau_min <= 0:
            raise RuntimeError("Controller minimum LR must be finite and positive")
        if not math.isfinite(self.early_stop_delta) or self.early_stop_delta < 0:
            raise RuntimeError("Controller early-stop delta must be finite and non-negative")
        if (
            not math.isfinite(self.paired_promotion_min_delta)
            or self.paired_promotion_min_delta < 0
        ):
            raise RuntimeError(
                "Controller paired-promotion delta must be finite and non-negative"
            )
        if self.regression_stop_patience < 0:
            raise RuntimeError("Controller regression patience must be non-negative")
        if not math.isfinite(self.regression_stop_delta) or self.regression_stop_delta < 0:
            raise RuntimeError(
                "Controller regression-stop delta must be finite and non-negative"
            )
        if self.reference_score is not None and not math.isfinite(self.reference_score):
            raise RuntimeError("Controller reference score must be finite or null")
        if self.reference_scores is not None and (
            not self.reference_scores
            or not all(math.isfinite(value) for value in self.reference_scores)
        ):
            raise RuntimeError("Controller paired reference scores must be finite")
        if min(
            self.plateau_evaluations,
            self.min_lr_evaluations,
            self.regression_evaluations,
            self.reductions,
            self.evaluations,
        ) < 0:
            raise RuntimeError("Controller counters must be non-negative")


def _same_artifact(first: Path, second: Path) -> bool:
    """Compare future paths and existing hard links as the same artifact."""
    if first.resolve() == second.resolve():
        return True
    if first.exists() and second.exists():
        try:
            return first.samefile(second)
        except OSError:
            return False
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train a DQN Snake agent using PyTorch"
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Episode cap; 0 uses 20x board area and relies primarily on the idle limit",
    )
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--initial-length", type=int, default=3)
    parser.add_argument("--allow-wrap", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--min-replay", type=int, default=2_000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=500_000)
    parser.add_argument("--target-update", type=int, default=5_000)
    parser.add_argument("--target-update-tau", type=float, default=0.005)
    parser.add_argument("--hard-update-interval", type=int, default=0)
    parser.add_argument("--disable-double-dqn", action="store_true")
    parser.add_argument("--disable-dueling", action="store_true")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000)
    parser.add_argument(
        "--policy-anchor-weight",
        type=float,
        default=0.0,
        help="Smooth-L1 weight that anchors Q values to the frozen warm-start policy",
    )
    parser.add_argument(
        "--teacher-replay-steps",
        type=int,
        default=0,
        help="Initial environment transitions collected greedily from the frozen teacher",
    )
    parser.add_argument(
        "--resume-epsilon",
        type=float,
        default=0.25,
        help="Exploration floor after resume because replay is intentionally not checkpointed",
    )
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of persistent training environments collected in parallel",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="Environment steps collected per active environment before each update phase",
    )
    parser.add_argument(
        "--updates-per-collection",
        type=int,
        default=0,
        help=(
            "Gradient updates after each collection; 0 preserves the configured "
            "train-frequency/gradient-steps update-to-transition ratio"
        ),
    )
    parser.add_argument("--reward-step", type=float, default=-0.003)
    parser.add_argument("--reward-food", type=float, default=5.0)
    parser.add_argument("--reward-death", type=float, default=-5.0)
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=1.0,
        help="Scale for potential-based food/topology shaping; 0 disables",
    )
    parser.add_argument("--max-idle-steps", type=int, default=90)
    parser.add_argument("--idle-growth-per-food", type=int, default=2)
    parser.add_argument("--idle-penalty", type=float, default=-5.0)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Allow faster but potentially non-reproducible backend algorithms",
    )
    parser.add_argument("--network-version", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--output", default="models/dqn_snake_v3_best.pt")
    parser.add_argument("--latest-output", default="models/dqn_snake_v3_latest.pt")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume-from", default=None)
    initialization.add_argument(
        "--warm-start-from",
        default=None,
        help="Initialize a fresh run from policy weights in an existing checkpoint",
    )
    initialization.add_argument(
        "--fresh", action="store_true", help="Ignore an existing latest checkpoint"
    )
    parser.add_argument(
        "--overwrite-fresh-output",
        action="store_true",
        help="Delete exact existing latest/best outputs before an intentional fresh run",
    )
    parser.add_argument(
        "--reset-best-evaluation",
        action="store_true",
        help="Discard the resumed best threshold when changing evaluation identity/output",
    )
    parser.add_argument(
        "--ignore-resume-metadata",
        action="store_true",
        help="Explicitly allow a missing or mismatched resume sidecar",
    )
    parser.add_argument(
        "--ignore-warm-start-metadata",
        action="store_true",
        help=(
            "Explicitly allow a missing, invalid, or mismatched warm-start source "
            "sidecar (for intentional legacy sources only)"
        ),
    )
    parser.add_argument(
        "--allow-environment-change",
        action="store_true",
        help="Allow an intentional MDP/config change while resuming full state",
    )
    parser.add_argument(
        "--allow-seed-change",
        action="store_true",
        help="Allow an intentional training episode-seed stream change on resume",
    )
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--render-frequency", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-delta", type=float, default=0.25)
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=0,
        help="Non-significant fixed-suite evaluations before reducing LR; 0 disables",
    )
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min", type=float, default=1e-6)
    parser.add_argument(
        "--require-paired-promotion",
        action="store_true",
        help="Promote best only when paired fixed-seed score CI is strictly positive",
    )
    parser.add_argument("--paired-promotion-min-delta", type=float, default=0.0)
    parser.add_argument("--regression-stop-patience", type=int, default=0)
    parser.add_argument("--regression-stop-delta", type=float, default=0.0)
    # Accepted only to make old commands fail safe instead of re-enabling the rollback loop.
    parser.add_argument(
        "--resume-best-on-decline", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume-decline-threshold", type=float, default=0.0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--resume-decline-cooldown", type=int, default=0, help=argparse.SUPPRESS
    )
    args = parser.parse_args(raw_argv)
    args._provided_options = sorted(
        {token.split("=", 1)[0] for token in raw_argv if token.startswith("--")}
    )
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error("episodes and eval-episodes must be positive")
    if args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("eval-interval and checkpoint-interval must be positive")
    if args.train_frequency <= 0 or args.gradient_steps <= 0:
        parser.error("train-frequency and gradient-steps must be positive")
    if args.num_envs <= 0 or args.rollout_steps <= 0:
        parser.error("num-envs and rollout-steps must be positive")
    if args.updates_per_collection < 0:
        parser.error("updates-per-collection must be non-negative")
    if args.teacher_replay_steps < 0:
        parser.error("teacher-replay-steps must be non-negative")
    if args.teacher_replay_steps > args.replay_capacity:
        parser.error("teacher-replay-steps must not exceed replay-capacity")
    if not math.isfinite(args.policy_anchor_weight) or args.policy_anchor_weight < 0:
        parser.error("policy-anchor-weight must be finite and non-negative")
    if args.early_stop_patience < 0 or args.lr_plateau_patience < 0:
        parser.error("early-stop-patience and lr-plateau-patience must be non-negative")
    if not math.isfinite(args.early_stop_delta) or args.early_stop_delta < 0:
        parser.error("early-stop-delta must be finite and non-negative")
    if args.regression_stop_patience < 0:
        parser.error("regression-stop-patience must be non-negative")
    if (
        not math.isfinite(args.regression_stop_delta)
        or args.regression_stop_delta < 0
    ):
        parser.error("regression-stop-delta must be finite and non-negative")
    if (
        not math.isfinite(args.paired_promotion_min_delta)
        or args.paired_promotion_min_delta < 0
    ):
        parser.error("paired-promotion-min-delta must be finite and non-negative")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        parser.error("lr-plateau-factor must be greater than 0 and less than 1")
    if not math.isfinite(args.lr_plateau_min) or args.lr_plateau_min <= 0:
        parser.error("lr-plateau-min must be finite and positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if args.lr_plateau_patience > 0 and args.lr_plateau_min > args.lr:
        parser.error("lr-plateau-min must not exceed the initial lr")
    output_artifacts = (
        ("--output", Path(args.output)),
        ("--output sidecar", sidecar_path(Path(args.output))),
        ("--latest-output", Path(args.latest_output)),
        ("--latest-output sidecar", sidecar_path(Path(args.latest_output))),
    )
    for index, (first_label, first_path) in enumerate(output_artifacts):
        for second_label, second_path in output_artifacts[index + 1 :]:
            if _same_artifact(first_path, second_path):
                parser.error(
                    f"{first_label} and {second_label} must be distinct artifacts"
                )
    explicit_source = args.warm_start_from or args.resume_from
    if explicit_source is not None and not Path(explicit_source).is_file():
        parser.error(f"Checkpoint not found: {explicit_source}")
    if args.resume_from is not None and _same_artifact(
        Path(args.resume_from), Path(args.output)
    ):
        parser.error(
            "--resume-from must differ from --output; resume a latest checkpoint or "
            "use --warm-start-from so an immutable best cannot be overwritten"
        )
    if args.warm_start_from is not None:
        source = Path(args.warm_start_from).resolve()
        source_artifacts = (
            ("--warm-start-from", source),
            ("warm-start sidecar", sidecar_path(source)),
        )
        for source_label, source_path in source_artifacts:
            for output_label, output_path in output_artifacts:
                if _same_artifact(source_path, output_path):
                    parser.error(
                        f"{source_label} must differ from {output_label} to preserve "
                        "the source checkpoint and metadata"
                    )
        resume_only = {
            "--reset-best-evaluation",
            "--ignore-resume-metadata",
            "--allow-environment-change",
            "--allow-seed-change",
            "--resume-epsilon",
        }
        invalid_options = sorted(resume_only.intersection(args._provided_options))
        if invalid_options:
            parser.error(
                "Warm start does not accept resume-only options: "
                + ", ".join(invalid_options)
            )
    elif "--ignore-warm-start-metadata" in args._provided_options:
        parser.error("--ignore-warm-start-metadata requires --warm-start-from")
    if args.resume_best_on_decline:
        print(
            "Warning: --resume-best-on-decline is retired; no model rollback will occur."
        )
    return args


def set_global_seed(seed: int) -> None:
    seed32 = int(seed) % (2**32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)


def accelerator_runtime_info(device: torch.device) -> dict[str, Any]:
    """Describe the actual PyTorch runtime selected for this training process."""
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    if device.type == "cuda":
        backend = "rocm" if hip_version else "cuda"
        device_name = torch.cuda.get_device_name(device)
    else:
        backend = device.type
        device_name = None
    return {
        "device": str(device),
        "backend": backend,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "hip_version": hip_version,
        "cuda_version": cuda_version,
    }


def _release_accelerator_resources(device: torch.device) -> None:
    """Release accelerator state before Windows tears the Python runtime down."""
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def deterministic_episode_seed(base_seed: int, episode: int, stream: int = 0) -> int:
    """Derive an episode seed without coupling training seeds to evaluation calls."""
    sequence = np.random.SeedSequence([int(base_seed), int(episode), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def episode_step_limit(config: GameConfig, configured_limit: int) -> int:
    if configured_limit > 0:
        return configured_limit
    if config.max_episode_steps > 0:
        return config.max_episode_steps
    return config.width * config.height * 20


def _neighbours(
    env: SnakeGameEnv, position: tuple[int, int]
) -> Iterable[tuple[int, int]]:
    for action in Action:
        dx, dy = action.vector
        x, y = position[0] + dx, position[1] + dy
        if env.config.allow_wrap:
            yield x % env.config.width, y % env.config.height
        elif 0 <= x < env.config.width and 0 <= y < env.config.height:
            yield x, y


def _food_distance(env: SnakeGameEnv) -> int:
    if env.food is None:
        return 0
    hx, hy = env.snake[0]
    fx, fy = env.food
    dx, dy = abs(hx - fx), abs(hy - fy)
    if env.config.allow_wrap:
        dx = min(dx, env.config.width - dx)
        dy = min(dy, env.config.height - dy)
    return dx + dy


def state_potential(env: SnakeGameEnv) -> float:
    """Bounded potential combining food progress and late-game topology.

    The tail is considered vacating for reachability.  This is deliberately a
    potential, not an immediate 'move toward food' reward, so safe detours do not
    accumulate a permanent penalty.
    """
    if env.done or env.food is None:
        return 0.0
    snake = env.snake
    head, tail = snake[0], snake[-1]
    blocked = set(snake[1:-1])
    queue = deque([head])
    reachable = {head}
    while queue:
        current = queue.popleft()
        for candidate in _neighbours(env, current):
            if candidate in blocked or candidate in reachable:
                continue
            reachable.add(candidate)
            queue.append(candidate)
    max_distance = max(1, env.config.width + env.config.height - 2)
    food_closeness = 1.0 - min(1.0, _food_distance(env) / max_distance)
    traversable = max(1, env.config.width * env.config.height - len(blocked))
    space_ratio = min(1.0, len(reachable) / traversable)
    tail_reachable = 1.0 if tail in reachable else 0.0
    return 0.55 * food_closeness + 0.30 * space_ratio + 0.15 * tail_reachable


def potential_shaping(
    previous_potential: float,
    env: SnakeGameEnv,
    *,
    gamma: float,
    scale: float,
    terminal: bool | None = None,
) -> float:
    if scale <= 0:
        return 0.0
    next_potential = (
        0.0 if (env.done if terminal is None else terminal) else state_potential(env)
    )
    return float(scale * (gamma * next_potential - previous_potential))


def action_mask(agent: DQNAgent, env: SnakeGameEnv) -> list[bool]:
    if agent.action_dim == len(RelativeAction):
        return [True] * agent.action_dim
    if agent.action_dim == len(Action):
        legal = set(env.legal_actions())
        return [action in legal for action in Action]
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def step_agent_action(
    agent: DQNAgent, env: SnakeGameEnv, action_index: int
) -> tuple[dict[str, object], float, bool, dict[str, object]]:
    if agent.action_dim == len(RelativeAction):
        return env.step_relative(RelativeAction(action_index))
    if agent.action_dim == len(Action):
        return env.step(Action(action_index))
    raise ValueError(f"Unsupported checkpoint action dimension: {agent.action_dim}")


def evaluate_agent(
    agent: DQNAgent,
    game_config: GameConfig,
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, Any]:
    environments = [SnakeGameEnv(game_config) for _ in seeds]
    for env, seed in zip(environments, seeds, strict=True):
        env.reset(seed=int(seed))
    encoder = BatchObservationEncoder(
        game_config.width,
        game_config.height,
        max_batch_size=len(environments),
        channels=agent.obs_shape[0],
        pin_memory=agent.device.type == "cuda",
    )
    rewards = [0.0] * len(environments)
    events: Counter[str] = Counter()
    truncated_count = 0
    active = list(range(len(environments)))
    while active:
        active_envs = [environments[index] for index in active]
        states = encoder.encode(active_envs)
        chosen_actions = agent.select_actions(
            states,
            epsilon_override=0.0,
            action_masks=[action_mask(agent, env) for env in active_envs],
        )
        survivors: list[int] = []
        for index, env, chosen in zip(active, active_envs, chosen_actions, strict=True):
            _, reward, done, info = step_agent_action(agent, env, chosen)
            rewards[index] += reward
            external_truncation = not done and env.steps >= max_steps
            if done or external_truncation:
                terminal_event = (
                    str(info.get("event", "terminated")) if done else "truncated"
                )
                if external_truncation or bool(info.get("truncated")):
                    truncated_count += 1
                events[terminal_event] += 1
            else:
                survivors.append(index)
        active = survivors

    scores = [env.score for env in environments]
    steps_taken = [env.steps for env in environments]

    def distribution(values: Sequence[float | int]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        margin = 1.96 * std / math.sqrt(len(array))
        return {
            "mean": mean,
            "std": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "reward": distribution(rewards),
        "score": distribution(scores),
        "steps": distribution(steps_taken),
        "seeds": [int(seed) for seed in seeds],
        "reward_samples": [float(value) for value in rewards],
        "score_samples": [float(value) for value in scores],
        "step_samples": [int(value) for value in steps_taken],
        "terminal_events": dict(sorted(events.items())),
        "truncated_count": truncated_count,
        "episodes": len(seeds),
    }


def evaluation_samples(
    evaluation: dict[str, Any], group: str
) -> list[float]:
    """Read per-seed samples while remaining compatible with legacy test/eval payloads."""
    raw = evaluation.get(f"{group}_samples")
    if isinstance(raw, list) and raw:
        return [float(value) for value in raw]
    episodes = int(evaluation.get("episodes", 1))
    distribution_group = "steps" if group == "step" else group
    mean = float(evaluation[distribution_group]["mean"])
    return [mean] * max(1, episodes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".meta.json")


def evaluation_identity(
    train_args: argparse.Namespace, game_config: GameConfig
) -> dict[str, Any]:
    identity = {
        "selection_metric": (
            "paired_score_ci95" if train_args.require_paired_promotion else "raw_score_mean"
        ),
        "safety_fallback": False,
        "run_seed": train_args.seed,
        "eval_seed_base": train_args.eval_seed_base,
        "eval_episodes": train_args.eval_episodes,
        "game_config": asdict(game_config),
    }
    if train_args.require_paired_promotion:
        identity["paired_promotion_min_delta"] = train_args.paired_promotion_min_delta
    return identity


def effective_agent_config(agent: DQNAgent) -> dict[str, Any]:
    learning_rates = [float(group["lr"]) for group in agent.optimizer.param_groups]
    return {
        "network_version": agent.network_version,
        "hidden_sizes": list(agent.hidden_sizes),
        "base_lr": float(agent.lr),
        # Kept for sidecar compatibility; this field has historically held the
        # optimizer's current per-group values rather than the base LR.
        "lr": learning_rates,
        "current_learning_rates": learning_rates,
        "gamma": agent.gamma,
        "batch_size": agent.batch_size,
        "replay_capacity": agent.replay_capacity,
        "min_replay_size": agent.min_replay_size,
        "n_step": agent.n_step,
        "per_alpha": agent.per_alpha,
        "per_beta_start": agent.per_beta_start,
        "per_beta_frames": agent.per_beta_frames,
        "target_update_interval": agent.target_update_interval,
        "target_update_tau": agent.target_update_tau,
        "hard_update_interval": agent.hard_update_interval,
        "use_double_dqn": agent.use_double_dqn,
        "use_dueling": agent.use_dueling,
        "epsilon_start": agent.epsilon_start,
        "epsilon_final": agent.epsilon_final,
        "epsilon_decay_steps": agent.epsilon_decay_steps,
        "policy_anchor_weight": agent.policy_anchor_weight,
        "teacher_replay_steps": agent.teacher_replay_steps,
        "policy_anchor_enabled": agent.policy_anchor_enabled,
        "amp_enabled": agent.amp_enabled,
    }


def save_checkpoint(
    agent: DQNAgent,
    path: Path,
    *,
    episode: int,
    run_seed: int,
    best_eval_score: float | None,
    best_eval_episode: int | None,
    train_args: argparse.Namespace,
    checkpoint_role: str,
    best_checkpoint_path: Path | None = None,
    episodes_started: int | None = None,
    warm_start_provenance: dict[str, Any] | None = None,
    convergence_controller: EvaluationConvergenceController | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    checkpoint_sha256 = _sha256(path)
    referenced_best_path: str | None = None
    referenced_best_sha256: str | None = None
    if best_eval_score is not None:
        candidate = path if checkpoint_role == "best_eval" else best_checkpoint_path
        if candidate is not None and candidate.exists():
            referenced_best_path = str(candidate.resolve())
            referenced_best_sha256 = (
                checkpoint_sha256
                if candidate.resolve() == path.resolve()
                else _sha256(candidate)
            )
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "episodes_completed": episode,
        "episodes_started": episode if episodes_started is None else episodes_started,
        "run_seed": run_seed,
        "best_eval_score": best_eval_score,
        "best_eval_episode": best_eval_episode,
        "best_checkpoint_path": referenced_best_path,
        "best_checkpoint_sha256": referenced_best_sha256,
        "evaluation_identity": evaluation_identity(
            train_args, agent.game_config or GameConfig()
        ),
        "network_version": agent.network_version,
        "action_dim": agent.action_dim,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
        "epsilon": agent.epsilon,
        "replay_size_at_save": len(agent.replay_buffer),
        "replay_restored": False,
        "game_config": asdict(agent.game_config) if agent.game_config else None,
        "effective_agent_config": effective_agent_config(agent),
        "base_learning_rate": float(agent.lr),
        "current_learning_rates": EvaluationConvergenceController.learning_rates(
            agent.optimizer
        ),
        "convergence_controller": (
            convergence_controller.to_dict()
            if convergence_controller is not None
            else None
        ),
        "train_args": vars(train_args),
        "warm_start_provenance": warm_start_provenance,
    }
    _atomic_json(sidecar_path(path), payload)


def load_resume_metadata(path: Path, *, ignore_mismatch: bool) -> dict[str, Any]:
    meta_path = sidecar_path(path)
    if not meta_path.exists():
        if ignore_mismatch:
            return {}
        raise RuntimeError(
            f"Resume metadata is missing: {meta_path}. Use --ignore-resume-metadata "
            "only for an intentional legacy resume."
        )
    with meta_path.open("r", encoding="utf-8-sig") as stream:
        metadata = json.load(stream)
    expected = metadata.get("checkpoint_sha256")
    actual = _sha256(path)
    if expected != actual:
        message = f"Checkpoint/metadata mismatch for {path}: expected SHA-256 {expected}, got {actual}."
        if not ignore_mismatch:
            raise RuntimeError(message + " Refusing a silent stale-best resume.")
        print("Warning:", message, "Treating it as a legacy resume.")
        return {}
    return metadata


def load_warm_start_metadata(
    path: Path, *, ignore_mismatch: bool
) -> tuple[dict[str, Any], bool, str]:
    """Load and authenticate source metadata without applying resume semantics."""
    actual = _sha256(path)
    meta_path = sidecar_path(path)
    if not meta_path.exists():
        if ignore_mismatch:
            print(
                "Warning: warm-start source metadata is missing:",
                meta_path,
                "Continuing only because --ignore-warm-start-metadata was supplied.",
            )
            return {}, False, actual
        raise RuntimeError(
            f"Warm-start source metadata is missing: {meta_path}. "
            "Use --ignore-warm-start-metadata only for an intentional legacy source."
        )
    try:
        with meta_path.open("r", encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        if ignore_mismatch:
            print(
                f"Warning: warm-start source metadata is unreadable ({meta_path}: {exc}). "
                "Continuing only because --ignore-warm-start-metadata was supplied."
            )
            return {}, False, actual
        raise RuntimeError(
            f"Warm-start source metadata is unreadable: {meta_path}: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        message = (
            f"Warm-start source metadata must be a JSON object: {meta_path}; "
            f"got {type(metadata).__name__}."
        )
        if not ignore_mismatch:
            raise RuntimeError(message)
        print(
            "Warning:",
            message,
            "Continuing only because --ignore-warm-start-metadata was supplied.",
        )
        return {}, False, actual

    expected = metadata.get("checkpoint_sha256")
    if expected != actual:
        message = (
            f"Warm-start checkpoint/metadata mismatch for {path}: expected SHA-256 "
            f"{expected}, got {actual}."
        )
        if not ignore_mismatch:
            raise RuntimeError(
                message + " Refusing an unauthenticated source; use "
                "--ignore-warm-start-metadata only when this is intentional."
            )
        print(
            "Warning:",
            message,
            "Continuing only because --ignore-warm-start-metadata was supplied.",
        )
        return metadata, False, actual
    return metadata, True, actual


def validate_resume_identity(metadata: dict[str, Any], agent: DQNAgent) -> None:
    """Ensure sidecar architecture claims describe the checkpoint actually loaded."""
    if not metadata:
        return
    expected = {
        "network_version": agent.network_version,
        "action_dim": agent.action_dim,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Resume sidecar fields conflict with the checkpoint payload: "
            + ", ".join(
                f"{key}=sidecar:{pair[0]!r}/checkpoint:{pair[1]!r}"
                for key, pair in mismatches.items()
            )
        )

    role = metadata.get("checkpoint_role")
    if role != "latest":
        raise RuntimeError(
            "Full resume requires a latest checkpoint sidecar; "
            f"got checkpoint_role={role!r}. Use --warm-start-from for a best_eval "
            "checkpoint so the immutable source cannot be overwritten."
        )


def validate_v3_contract(agent: DQNAgent) -> None:
    if agent.network_version < 3:
        return
    if agent.action_dim != len(RelativeAction):
        raise RuntimeError(
            "A v3 Snake checkpoint must use exactly three relative actions."
        )
    if agent.obs_shape[0] != V3_OBSERVATION_CHANNELS:
        raise RuntimeError(
            f"A v3 Snake checkpoint must use {V3_OBSERVATION_CHANNELS} observation channels; "
            f"got {agent.obs_shape[0]}. Start a fresh v3 run instead of silently dropping state."
        )


def validate_resume_agent_options(args: argparse.Namespace, agent: DQNAgent) -> None:
    """Reject explicitly requested hyperparameters that resume cannot silently apply."""
    provided = set(getattr(args, "_provided_options", ()))
    checks: dict[str, tuple[Any, Any]] = {
        "--network-version": (args.network_version, agent.network_version),
        "--hidden": (tuple(args.hidden), tuple(agent.hidden_sizes)),
        "--lr": (float(args.lr), float(agent.lr)),
        "--gamma": (args.gamma, agent.gamma),
        "--batch-size": (args.batch_size, agent.batch_size),
        "--replay-capacity": (args.replay_capacity, agent.replay_capacity),
        "--min-replay": (args.min_replay, agent.min_replay_size),
        "--n-step": (args.n_step, agent.n_step),
        "--per-alpha": (args.per_alpha, agent.per_alpha),
        "--per-beta-start": (args.per_beta_start, agent.per_beta_start),
        "--per-beta-frames": (args.per_beta_frames, agent.per_beta_frames),
        "--target-update": (args.target_update, agent.target_update_interval),
        "--target-update-tau": (args.target_update_tau, agent.target_update_tau),
        "--hard-update-interval": (
            args.hard_update_interval,
            agent.hard_update_interval,
        ),
        "--disable-double-dqn": (not args.disable_double_dqn, agent.use_double_dqn),
        "--disable-dueling": (not args.disable_dueling, agent.use_dueling),
        "--epsilon-start": (args.epsilon_start, agent.epsilon_start),
        "--epsilon-final": (args.epsilon_final, agent.epsilon_final),
        "--epsilon-decay-steps": (args.epsilon_decay_steps, agent.epsilon_decay_steps),
        "--policy-anchor-weight": (
            args.policy_anchor_weight,
            agent.policy_anchor_weight,
        ),
        "--teacher-replay-steps": (
            args.teacher_replay_steps,
            agent.teacher_replay_steps,
        ),
    }
    conflicts = {
        option: values
        for option, values in checks.items()
        if option in provided and values[0] != values[1]
    }
    if conflicts:
        detail = ", ".join(
            f"{option}=requested:{values[0]!r}/checkpoint:{values[1]!r}"
            for option, values in conflicts.items()
        )
        raise RuntimeError(
            "Explicit agent hyperparameters conflict with the resumed checkpoint: "
            + detail
            + ". Start a fresh run for a new optimizer/agent configuration."
        )


_CONTROLLER_OPTIONS = {
    "--lr-plateau-patience": "lr_plateau_patience",
    "--lr-plateau-factor": "lr_plateau_factor",
    "--lr-plateau-min": "lr_plateau_min",
    "--early-stop-patience": "early_stop_patience",
    "--early-stop-delta": "early_stop_delta",
    "--require-paired-promotion": "require_paired_promotion",
    "--paired-promotion-min-delta": "paired_promotion_min_delta",
    "--regression-stop-patience": "regression_stop_patience",
    "--regression-stop-delta": "regression_stop_delta",
}


def _new_convergence_controller(
    args: argparse.Namespace, *, reference_score: float | None = None
) -> EvaluationConvergenceController:
    controller = EvaluationConvergenceController(
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_min=args.lr_plateau_min,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        require_paired_promotion=args.require_paired_promotion,
        paired_promotion_min_delta=args.paired_promotion_min_delta,
        regression_stop_patience=args.regression_stop_patience,
        regression_stop_delta=args.regression_stop_delta,
        reference_score=reference_score,
    )
    controller.validate()
    return controller


def restore_convergence_controller(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    agent: DQNAgent,
    *,
    legacy_reference_score: float | None,
) -> EvaluationConvergenceController:
    """Restore controller policy/state and verify its optimizer linkage."""
    optimizer_lrs = EvaluationConvergenceController.learning_rates(agent.optimizer)
    stored_lrs = metadata.get("current_learning_rates") if metadata else None
    if stored_lrs is not None:
        if not isinstance(stored_lrs, list):
            raise RuntimeError("Resume current_learning_rates must be a list")
        parsed_lrs = [float(value) for value in stored_lrs]
        matches = len(parsed_lrs) == len(optimizer_lrs) and all(
            math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-15)
            for expected, actual in zip(parsed_lrs, optimizer_lrs, strict=True)
        )
        if not matches:
            raise RuntimeError(
                "Resume sidecar current learning rates conflict with checkpoint "
                f"optimizer state: sidecar={parsed_lrs}, optimizer={optimizer_lrs}."
            )

    payload = metadata.get("convergence_controller") if metadata else None
    if payload is None:
        if metadata:
            print(
                "Warning: resume sidecar has no convergence controller state; "
                "initializing from the saved best score with zero patience counters."
            )
        return _new_convergence_controller(
            args, reference_score=legacy_reference_score
        )
    if not isinstance(payload, dict):
        raise RuntimeError("Resume convergence_controller must be a JSON object")
    restored = EvaluationConvergenceController.from_dict(payload)
    if restored.scheduler_enabled and restored.lr_plateau_min > float(agent.lr):
        raise RuntimeError(
            "Resume convergence controller minimum LR exceeds the checkpoint base LR: "
            f"minimum={restored.lr_plateau_min}, base={agent.lr}."
        )
    provided = set(getattr(args, "_provided_options", ()))
    conflicts: list[str] = []
    for option, attribute in _CONTROLLER_OPTIONS.items():
        stored = getattr(restored, attribute)
        requested = getattr(args, attribute)
        if option in provided and requested != stored:
            conflicts.append(
                f"{option}=requested:{requested!r}/checkpoint:{stored!r}"
            )
        elif option not in provided:
            setattr(args, attribute, stored)
    if conflicts:
        raise RuntimeError(
            "Explicit scheduler/early-stop options conflict with the resumed "
            "convergence controller: " + ", ".join(conflicts)
        )
    return restored


def validate_resume_seed(metadata: dict[str, Any], args: argparse.Namespace) -> None:
    if not metadata or metadata.get("run_seed") is None:
        return
    stored_seed = int(metadata["run_seed"])
    if stored_seed != args.seed and not args.allow_seed_change:
        raise RuntimeError(
            f"Resume run seed changed from {stored_seed} to {args.seed}. "
            "Use --allow-seed-change for an intentional new episode stream."
        )


def validate_resume_environment(
    agent: DQNAgent,
    requested: GameConfig,
    *,
    allow_change: bool,
) -> None:
    if agent.game_config is None:
        return
    stored = asdict(agent.game_config)
    current = asdict(requested)
    mismatches = {
        key: (stored.get(key), current.get(key))
        for key in current
        if stored.get(key) != current.get(key)
    }
    if not mismatches:
        return
    detail = ", ".join(
        f"{key}=checkpoint:{pair[0]!r}/requested:{pair[1]!r}"
        for key, pair in mismatches.items()
    )
    if not allow_change:
        raise RuntimeError(
            "Resume environment/MDP differs from the checkpoint: "
            + detail
            + ". Use --allow-environment-change only for an intentional legacy resume."
        )
    print("Warning: intentionally changing resume environment:", detail)


def validate_resume_best(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    game_config: GameConfig,
    output_path: Path,
) -> tuple[float, int | None]:
    """Validate that a resumed best threshold names a real, comparable artifact."""
    if args.reset_best_evaluation:
        for artifact in (output_path, sidecar_path(output_path)):
            if artifact.exists():
                artifact.unlink()
        return -math.inf, None
    if not metadata or metadata.get("best_eval_score") is None:
        return -math.inf, None

    stored_identity = metadata.get("evaluation_identity")
    current_identity = evaluation_identity(args, game_config)
    if stored_identity != current_identity:
        raise RuntimeError(
            "Resume best evaluation is not comparable with the requested fixed suite/MDP. "
            "Use --reset-best-evaluation for an intentional new selection baseline."
        )

    referenced_path = metadata.get("best_checkpoint_path")
    referenced_sha = metadata.get("best_checkpoint_sha256")
    if not referenced_path or Path(referenced_path).resolve() != output_path.resolve():
        raise RuntimeError(
            "Resume metadata does not link its best score to the requested --output artifact. "
            "Use the original best path or --reset-best-evaluation."
        )
    if (
        not output_path.exists()
        or not referenced_sha
        or _sha256(output_path) != referenced_sha
    ):
        raise RuntimeError(
            "The best checkpoint linked by the resume metadata is missing or stale."
        )

    best_meta_path = sidecar_path(output_path)
    if not best_meta_path.exists():
        raise RuntimeError(f"Best checkpoint metadata is missing: {best_meta_path}")
    with best_meta_path.open("r", encoding="utf-8-sig") as stream:
        best_metadata = json.load(stream)
    score = float(metadata["best_eval_score"])
    episode_value = metadata.get("best_eval_episode")
    episode = int(episode_value) if episode_value is not None else None
    if (
        best_metadata.get("checkpoint_role") != "best_eval"
        or best_metadata.get("checkpoint_sha256") != referenced_sha
        or best_metadata.get("evaluation_identity") != current_identity
        or best_metadata.get("best_eval_score") != score
        or best_metadata.get("best_eval_episode") != episode
    ):
        raise RuntimeError(
            "Best checkpoint sidecar conflicts with the resumed best identity."
        )
    return score, episode


def _resume_path(args: argparse.Namespace) -> Path | None:
    if args.fresh or args.warm_start_from:
        return None
    if args.resume_from:
        path = Path(args.resume_from)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    latest = Path(args.latest_output)
    return latest if latest.exists() else None


def _validate_output_network_identity(
    network_version: int, output_path: Path, latest_path: Path
) -> None:
    if network_version < 3 and any(
        "v3" in path.name.lower() for path in (output_path, latest_path)
    ):
        raise RuntimeError(
            "A legacy v1/v2 run must use explicit non-v3 --latest-output and "
            "--output paths so it cannot masquerade as a v3 checkpoint."
        )


def _new_agent(
    args: argparse.Namespace,
    game_config: GameConfig,
    train_env: SnakeGameEnv,
) -> DQNAgent:
    initial_channels = 3 if args.network_version == 1 else None
    initial_state = flatten_observation(
        train_env, device="cpu", expected_channels=initial_channels
    )
    obs_shape = tuple(int(value) for value in initial_state.shape)
    return DQNAgent(
        state_dim=int(np.prod(obs_shape)),
        action_dim=len(RelativeAction) if args.network_version >= 3 else len(Action),
        hidden_sizes=tuple(args.hidden),
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay,
        target_update_interval=args.target_update,
        target_update_tau=args.target_update_tau,
        hard_update_interval=args.hard_update_interval,
        use_double_dqn=not args.disable_double_dqn,
        use_dueling=not args.disable_dueling,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        n_step=args.n_step,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_frames=args.per_beta_frames,
        device=args.device,
        game_config=game_config,
        obs_shape=obs_shape,
        network_version=args.network_version,
        amp_enabled=False if args.disable_amp else None,
        policy_anchor_weight=args.policy_anchor_weight,
        teacher_replay_steps=args.teacher_replay_steps,
    )


def _prepare_fresh_outputs(args: argparse.Namespace, resume_path: Path | None) -> None:
    if resume_path is not None:
        return
    outputs = (Path(args.output), Path(args.latest_output))
    artifacts = [
        item for path in outputs for item in (path, sidecar_path(path)) if item.exists()
    ]
    if not artifacts:
        return
    if not args.overwrite_fresh_output:
        rendered = ", ".join(str(path) for path in artifacts)
        raise RuntimeError(
            "Fresh training would mix with existing output artifacts: "
            f"{rendered}. Use distinct paths or --overwrite-fresh-output."
        )
    for artifact in artifacts:
        artifact.unlink()
    print(
        "Removed exact stale outputs for intentional fresh training:",
        ", ".join(map(str, artifacts)),
    )


def _reheat_exploration(agent: DQNAgent, minimum: float) -> None:
    target = min(agent.epsilon_start, max(agent.epsilon_final, float(minimum)))
    if agent.epsilon >= target:
        return
    span = agent.epsilon_start - agent.epsilon_final
    progress = 0.0 if span <= 0 else (agent.epsilon_start - target) / span
    agent.behavior_steps = int(max(0.0, min(1.0, progress)) * agent.epsilon_decay_steps)
    agent.epsilon = target


@dataclass
class _TrainingSlot:
    stream_id: int
    env: SnakeGameEnv
    seed_index: int = 0
    total_env_reward: float = 0.0
    total_shaped_reward: float = 0.0
    started_at: float = 0.0
    active: bool = False

    def reset(self, *, run_seed: int, seed_index: int) -> None:
        self.seed_index = seed_index
        self.env.reset(seed=deterministic_episode_seed(run_seed, seed_index))
        self.total_env_reward = 0.0
        self.total_shaped_reward = 0.0
        self.started_at = time.perf_counter()
        self.active = True


def _train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    torch.use_deterministic_algorithms(not args.allow_nondeterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.allow_nondeterministic
        torch.backends.cudnn.deterministic = not args.allow_nondeterministic

    configured_step_limit = (
        args.max_steps if args.max_steps > 0 else args.width * args.height * 20
    )
    game_config = GameConfig(
        width=args.width,
        height=args.height,
        initial_length=args.initial_length,
        reward_step=args.reward_step,
        reward_food=args.reward_food,
        reward_death=args.reward_death,
        allow_wrap=args.allow_wrap,
        seed=None,
        max_idle_steps=args.max_idle_steps,
        idle_penalty=args.idle_penalty,
        idle_growth_per_food=args.idle_growth_per_food,
        max_episode_steps=configured_step_limit,
    )
    train_env = SnakeGameEnv(game_config)
    train_env.reset(seed=deterministic_episode_seed(args.seed, 0))
    step_limit = episode_step_limit(game_config, args.max_steps)

    resume_path = _resume_path(args)
    output_path = Path(args.output)
    latest_path = Path(args.latest_output)
    if resume_path is None:
        _validate_output_network_identity(
            args.network_version, output_path, latest_path
        )
    conservative_options_enabled = any(
        (
            args.policy_anchor_weight > 0,
            args.teacher_replay_steps > 0,
            args.require_paired_promotion,
            args.regression_stop_patience > 0,
        )
    )
    if resume_path is None and not args.warm_start_from and conservative_options_enabled:
        raise RuntimeError(
            "Policy anchoring, teacher replay, and paired evaluation guards require "
            "--warm-start-from (or a latest checkpoint that already contains them)."
        )
    if not args.warm_start_from:
        _prepare_fresh_outputs(args, resume_path)
    start_episode = 1
    episodes_started = 0
    best_eval_score = -math.inf
    best_eval_episode: int | None = None
    warm_start_provenance: dict[str, Any] | None = None
    controller: EvaluationConvergenceController
    if resume_path is not None:
        metadata = load_resume_metadata(
            resume_path, ignore_mismatch=args.ignore_resume_metadata
        )
        agent = DQNAgent.load(str(resume_path), device=args.device)
        validate_resume_identity(metadata, agent)
        validate_resume_seed(metadata, args)
        validate_resume_agent_options(args, agent)
        validate_resume_environment(
            agent, game_config, allow_change=args.allow_environment_change
        )
        validate_v3_contract(agent)
        _validate_output_network_identity(
            agent.network_version, output_path, latest_path
        )
        agent.game_config = game_config
        agent.configure_amp(False if args.disable_amp else None)
        agent.configure_replay_pin_memory()
        if (
            (agent.policy_anchor_weight > 0 or agent.teacher_replay_steps > 0)
            and not agent.policy_anchor_enabled
        ):
            raise RuntimeError(
                "Resume checkpoint requires a frozen policy anchor, but none was restored."
            )
        resume_state = flatten_observation(
            train_env, agent.device, expected_channels=agent.obs_shape[0]
        )
        if tuple(resume_state.shape) != tuple(agent.obs_shape):
            raise RuntimeError(
                f"Resume board/observation shape {tuple(resume_state.shape)} does not match "
                f"checkpoint shape {tuple(agent.obs_shape)}. Start a fresh v3 run for a new board size."
            )
        if metadata:
            warm_start_provenance = metadata.get("warm_start_provenance")
            episodes_completed = int(metadata.get("episodes_completed", 0))
            start_episode = episodes_completed + 1
            episodes_started = int(metadata.get("episodes_started", episodes_completed))
        stored_best = metadata.get("best_eval_score") if metadata else None
        controller = restore_convergence_controller(
            metadata,
            args,
            agent,
            legacy_reference_score=(
                None if stored_best is None else float(stored_best)
            ),
        )
        if metadata:
            best_eval_score, best_eval_episode = validate_resume_best(
                metadata,
                args,
                game_config,
                output_path,
            )
        if args.reset_best_evaluation:
            controller.reset()
        _reheat_exploration(agent, args.resume_epsilon)
        print(
            f"Resuming {resume_path} at episode {start_episode}; replay starts empty and "
            f"epsilon is {agent.epsilon:.3f}."
        )
    else:
        agent = _new_agent(args, game_config, train_env)
        controller = _new_convergence_controller(args)
        if args.warm_start_from:
            source = Path(args.warm_start_from)
            (
                source_metadata,
                source_sidecar_verified,
                source_checkpoint_sha256,
            ) = load_warm_start_metadata(
                source, ignore_mismatch=args.ignore_warm_start_metadata
            )
            embedded_metadata = agent.load_policy_weights(
                str(source), expected_sha256=source_checkpoint_sha256
            )
            if agent.policy_anchor_weight > 0 or agent.teacher_replay_steps > 0:
                agent.snapshot_policy_anchor()
            embedded_obs_shape = embedded_metadata.get("obs_shape")
            warm_start_provenance = {
                "source_path": str(source.resolve()),
                "checkpoint_sha256": source_checkpoint_sha256,
                "sidecar_role": source_metadata.get("checkpoint_role"),
                "sidecar_episode": source_metadata.get("episodes_completed"),
                "embedded_network_version": embedded_metadata.get("network_version"),
                "embedded_obs_shape": (
                    list(embedded_obs_shape) if embedded_obs_shape is not None else None
                ),
                "source_sidecar_verified": source_sidecar_verified,
            }
            # Preserve existing destinations until source authentication and policy
            # compatibility have both succeeded.
            _prepare_fresh_outputs(args, None)
            print(
                f"Warm-started policy weights from {source.resolve()}; target is "
                "synchronized from that policy, while optimizer, scaler, replay, "
                "epsilon, counters, seeds, best identity, and outputs are fresh."
            )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_log_{int(time.time())}.jsonl"
    eval_seeds = [args.eval_seed_base + index for index in range(args.eval_episodes)]
    rolling_scores: deque[int] = deque(maxlen=100)
    episodes_completed = start_episode - 1
    final_episode = episodes_completed
    initial_completed = episodes_completed
    episodes_launched_this_run = 0
    next_seed_index = episodes_started + 1
    next_eval_episode = (
        (episodes_completed // args.eval_interval) + 1
    ) * args.eval_interval
    next_checkpoint_episode = (
        (episodes_completed // args.checkpoint_interval) + 1
    ) * args.checkpoint_interval

    slots = [
        _TrainingSlot(stream_id=index, env=SnakeGameEnv(game_config))
        for index in range(min(args.num_envs, args.episodes))
    ]
    current_encoder = BatchObservationEncoder(
        game_config.width,
        game_config.height,
        max_batch_size=len(slots),
        channels=agent.obs_shape[0],
        pin_memory=agent.device.type == "cuda",
    )
    next_encoder = BatchObservationEncoder(
        game_config.width,
        game_config.height,
        max_batch_size=len(slots),
        channels=agent.obs_shape[0],
        pin_memory=agent.device.type == "cuda",
    )
    runtime_info = accelerator_runtime_info(agent.device)
    print(
        "Training device: "
        f"{runtime_info['device']} ({runtime_info['backend']}; "
        f"{runtime_info['device_name'] or 'host CPU'}; torch {runtime_info['torch_version']}; "
        f"HIP {runtime_info['hip_version'] or 'n/a'})."
    )

    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "record_type": "run_start",
                    "seed": args.seed,
                    "start_episode": start_episode,
                    "episodes_started": episodes_started,
                    "resume_path": str(resume_path) if resume_path else None,
                    "warm_start_provenance": warm_start_provenance,
                    "network_version": agent.network_version,
                    "action_dim": agent.action_dim,
                    "obs_shape": list(agent.obs_shape),
                    "step_limit": step_limit,
                    "observation_pinned": current_encoder.is_pinned,
                    "runtime": runtime_info,
                    "eval_seeds": eval_seeds,
                    "current_learning_rates": controller.learning_rates(
                        agent.optimizer
                    ),
                    "convergence_controller": controller.to_dict(),
                    "args": vars(args),
                }
            )
            + "\n"
        )

    if args.warm_start_from:
        baseline = evaluate_agent(agent, game_config, eval_seeds, step_limit)
        baseline_score = float(baseline["score"]["mean"])
        baseline_score_samples = evaluation_samples(baseline, "score")
        controller_decision = controller.observe(
            baseline_score,
            agent.optimizer,
            sample_scores=baseline_score_samples,
        )
        controller.set_paired_reference(baseline_score_samples)
        best_eval_score = baseline_score
        best_eval_episode = 0
        save_checkpoint(
            agent,
            output_path,
            episode=0,
            run_seed=args.seed,
            best_eval_score=best_eval_score,
            best_eval_episode=best_eval_episode,
            train_args=args,
            checkpoint_role="best_eval",
            best_checkpoint_path=output_path,
            episodes_started=0,
            warm_start_provenance=warm_start_provenance,
            convergence_controller=controller,
        )
        save_checkpoint(
            agent,
            latest_path,
            episode=0,
            run_seed=args.seed,
            best_eval_score=best_eval_score,
            best_eval_episode=best_eval_episode,
            train_args=args,
            checkpoint_role="latest",
            best_checkpoint_path=output_path,
            episodes_started=0,
            warm_start_provenance=warm_start_provenance,
            convergence_controller=controller,
        )
        baseline_record: dict[str, Any] = {
            "record_type": "evaluation",
            "evaluation_kind": "warm_start_baseline",
            "episode": 0,
            "best_eval_score": best_eval_score,
            "best_eval_episode": best_eval_episode,
            "current_learning_rates": controller.learning_rates(agent.optimizer),
            "convergence_decision": controller_decision,
            "convergence_controller": controller.to_dict(),
        }
        for group in ("reward", "score", "steps"):
            for name, value in baseline[group].items():
                baseline_record[f"eval_{group}_{name}"] = value
        baseline_record["eval_terminal_events"] = baseline["terminal_events"]
        baseline_record["eval_truncated_count"] = baseline["truncated_count"]
        baseline_record["eval_seeds"] = baseline.get("seeds", eval_seeds)
        baseline_record["eval_reward_samples"] = evaluation_samples(
            baseline, "reward"
        )
        baseline_record["eval_score_samples"] = baseline_score_samples
        baseline_record["eval_step_samples"] = evaluation_samples(baseline, "step")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")
        print(
            f"Warm-start baseline fixed-suite score {baseline_score:.3f}; "
            f"saved episode-0 best {output_path} and latest {latest_path}."
        )

    # Training environments are not launched until an authenticated warm-start
    # baseline and its two new-run checkpoints have been durably recorded.
    for slot in slots:
        slot.reset(run_seed=args.seed, seed_index=next_seed_index)
        next_seed_index += 1
        episodes_started += 1
        episodes_launched_this_run += 1

    stop_requested = False
    collection_index = 0
    while episodes_completed < initial_completed + args.episodes and not stop_requested:
        collection_index += 1
        collection_started = time.perf_counter()
        behavior_steps_before = agent.behavior_steps
        collection_transitions = 0
        encoding_seconds = 0.0
        action_selection_seconds = 0.0
        completed_summaries: list[dict[str, Any]] = []

        for _ in range(args.rollout_steps):
            active_slots = [slot for slot in slots if slot.active]
            if not active_slots:
                break
            active_envs = [slot.env for slot in active_slots]
            encode_started = time.perf_counter()
            states = current_encoder.encode(active_envs)
            encoding_seconds += time.perf_counter() - encode_started
            previous_potentials = (
                batch_state_potentials(active_envs)
                if args.reward_shaping_scale > 0
                else np.zeros(len(active_envs), dtype=np.float64)
            )
            selection_started = time.perf_counter()
            masks = [action_mask(agent, env) for env in active_envs]
            if (
                agent.policy_anchor_enabled
                and len(agent.replay_buffer) < agent.teacher_replay_steps
            ):
                chosen_actions = agent.select_anchor_actions(
                    states,
                    action_masks=masks,
                    advance_behavior_steps=True,
                )
            else:
                chosen_actions = agent.select_actions(states, action_masks=masks)
            action_selection_seconds += time.perf_counter() - selection_started

            step_results: list[tuple[float, bool, bool, dict[str, object]]] = []
            for slot, chosen in zip(active_slots, chosen_actions, strict=True):
                _, env_reward, env_done, info = step_agent_action(
                    agent, slot.env, chosen
                )
                truncated = bool(info.get("truncated")) or (
                    not env_done and slot.env.steps >= step_limit
                )
                step_results.append((env_reward, env_done, env_done or truncated, info))

            encode_started = time.perf_counter()
            next_states = next_encoder.encode(active_envs)
            encoding_seconds += time.perf_counter() - encode_started
            next_potentials = (
                batch_state_potentials(active_envs)
                if args.reward_shaping_scale > 0
                else np.zeros(len(active_envs), dtype=np.float64)
            )

            for batch_index, (slot, chosen, result) in enumerate(
                zip(active_slots, chosen_actions, step_results, strict=True)
            ):
                env_reward, env_done, replay_done, info = result
                if replay_done:
                    next_potential = 0.0
                else:
                    next_potential = float(next_potentials[batch_index])
                shaped_reward = env_reward + args.reward_shaping_scale * (
                    agent.gamma * next_potential
                    - float(previous_potentials[batch_index])
                )
                agent.remember(
                    states[batch_index],
                    chosen,
                    shaped_reward,
                    next_states[batch_index],
                    replay_done,
                    next_action_mask=action_mask(agent, slot.env),
                    stream_id=slot.stream_id,
                )
                collection_transitions += 1
                slot.total_env_reward += env_reward
                slot.total_shaped_reward += shaped_reward

                if not replay_done:
                    continue
                truncated = bool(info.get("truncated")) or not env_done
                terminal_event = (
                    str(info.get("event", "terminated")) if env_done else "truncated"
                )
                episodes_completed += 1
                final_episode = episodes_completed
                rolling_scores.append(slot.env.score)
                completed_summaries.append(
                    {
                        "record_type": "episode",
                        "episode": episodes_completed,
                        "seed_index": slot.seed_index,
                        "environment_stream": slot.stream_id,
                        "reward": slot.total_env_reward,
                        "shaped_reward": slot.total_shaped_reward,
                        "score": slot.env.score,
                        "snake_length": len(slot.env.snake),
                        "steps": slot.env.steps,
                        "rolling_score_100": float(statistics.mean(rolling_scores)),
                        "terminal_event": terminal_event,
                        "terminated": env_done and not truncated,
                        "truncated": truncated,
                        "duration_seconds": time.perf_counter() - slot.started_at,
                        "render": (
                            slot.env.render(to_string=True)
                            if args.render_frequency
                            and episodes_completed % args.render_frequency == 0
                            else None
                        ),
                    }
                )
                if episodes_launched_this_run < args.episodes:
                    slot.reset(run_seed=args.seed, seed_index=next_seed_index)
                    next_seed_index += 1
                    episodes_started += 1
                    episodes_launched_this_run += 1
                else:
                    slot.active = False

        collection_seconds = max(time.perf_counter() - collection_started, 1e-12)
        teacher_replay_complete = (
            len(agent.replay_buffer) >= agent.teacher_replay_steps
        )
        if not teacher_replay_complete:
            update_attempts = 0
        elif args.updates_per_collection > 0:
            update_attempts = args.updates_per_collection
        else:
            update_attempts = (
                (agent.behavior_steps // args.train_frequency)
                - (behavior_steps_before // args.train_frequency)
            ) * args.gradient_steps
        learn_metrics: list[dict[str, float]] = []
        updates_started = time.perf_counter()
        for _ in range(update_attempts):
            result = agent.learn()
            if result is not None:
                learn_metrics.append(result)
        update_seconds = max(time.perf_counter() - updates_started, 1e-12)

        def metric_mean(
            name: str, metrics: list[dict[str, float]] = learn_metrics
        ) -> float | None:
            values = [item[name] for item in metrics if name in item]
            return float(statistics.mean(values)) if values else None

        sampling_seconds = sum(
            item.get("sampling_seconds", 0.0) for item in learn_metrics
        )
        gpu_wait_seconds = sum(
            item.get("gpu_wait_seconds", 0.0) for item in learn_metrics
        )
        collection_metrics: dict[str, Any] = {
            "record_type": "collection",
            "collection": collection_index,
            "episodes_completed": episodes_completed,
            "episodes_started": episodes_started,
            "collection_transitions": collection_transitions,
            "collection_update_attempts": update_attempts,
            "collection_updates": len(learn_metrics),
            "env_steps_per_second": collection_transitions / collection_seconds,
            "updates_per_second": len(learn_metrics) / update_seconds,
            "sampling_seconds": sampling_seconds,
            "gpu_wait_seconds": gpu_wait_seconds,
            "encoding_seconds": encoding_seconds,
            "action_selection_seconds": action_selection_seconds,
            "collection_seconds": collection_seconds,
            "update_seconds": update_seconds,
            "epsilon": agent.epsilon,
            "current_learning_rates": controller.learning_rates(agent.optimizer),
            "convergence_controller": controller.to_dict(),
            "behavior_steps": agent.behavior_steps,
            "learn_step_counter": agent.learn_step_counter,
            "replay_size": len(agent.replay_buffer),
            "teacher_replay_steps": agent.teacher_replay_steps,
            "teacher_replay_collected": min(
                len(agent.replay_buffer), agent.teacher_replay_steps
            ),
            "teacher_replay_complete": teacher_replay_complete,
            "avg_loss": metric_mean("loss"),
            "avg_td_loss": metric_mean("td_loss"),
            "avg_anchor_loss": metric_mean("anchor_loss"),
            "policy_anchor_weight": agent.policy_anchor_weight,
            "avg_td_error": metric_mean("td_error"),
            "avg_grad_norm": metric_mean("grad_norm"),
            "avg_q_mean": metric_mean("q_mean"),
        }
        for metrics in completed_summaries:
            metrics.update(
                {
                    key: value
                    for key, value in collection_metrics.items()
                    if key not in {"record_type", "collection"}
                }
            )

        should_evaluate = episodes_completed >= next_eval_episode
        controller_decision: dict[str, Any] | None = None
        if should_evaluate:
            evaluation = evaluate_agent(agent, game_config, eval_seeds, step_limit)
            metrics = completed_summaries[-1]
            for group in ("reward", "score", "steps"):
                for name, value in evaluation[group].items():
                    metrics[f"eval_{group}_{name}"] = value
            metrics["eval_terminal_events"] = evaluation["terminal_events"]
            metrics["eval_truncated_count"] = evaluation["truncated_count"]
            score_samples = evaluation_samples(evaluation, "score")
            metrics["eval_seeds"] = evaluation.get("seeds", eval_seeds)
            metrics["eval_reward_samples"] = evaluation_samples(
                evaluation, "reward"
            )
            metrics["eval_score_samples"] = score_samples
            metrics["eval_step_samples"] = evaluation_samples(evaluation, "step")
            average_score = float(evaluation["score"]["mean"])
            previous_best = best_eval_score
            controller_decision = controller.observe(
                average_score,
                agent.optimizer,
                sample_scores=score_samples,
                defer_reason=(
                    None if teacher_replay_complete else "teacher_replay_warmup"
                ),
            )
            improved = teacher_replay_complete and average_score > previous_best and (
                not controller.require_paired_promotion
                or controller_decision["paired_promotion_eligible"]
            )
            metrics["current_learning_rates"] = controller.learning_rates(
                agent.optimizer
            )
            metrics["convergence_decision"] = controller_decision
            metrics["convergence_controller"] = controller.to_dict()
            collection_metrics["current_learning_rates"] = controller.learning_rates(
                agent.optimizer
            )
            collection_metrics["convergence_decision"] = controller_decision
            collection_metrics["convergence_controller"] = controller.to_dict()
            if improved:
                best_eval_score = average_score
                best_eval_episode = episodes_completed
                controller.set_paired_reference(score_samples)
                metrics["convergence_controller"] = controller.to_dict()
                collection_metrics["convergence_controller"] = controller.to_dict()
                save_checkpoint(
                    agent,
                    output_path,
                    episode=episodes_completed,
                    run_seed=args.seed,
                    best_eval_score=best_eval_score,
                    best_eval_episode=best_eval_episode,
                    train_args=args,
                    checkpoint_role="best_eval",
                    best_checkpoint_path=output_path,
                    episodes_started=episodes_started,
                    warm_start_provenance=warm_start_provenance,
                    convergence_controller=controller,
                )
                print(
                    f"New best fixed-suite score {best_eval_score:.3f} at episode "
                    f"{episodes_completed}; "
                    f"saved {output_path}."
                )
            if controller_decision["lr_reduced"]:
                print(
                    "Reduced optimizer learning rates after fixed-suite plateau: "
                    f"{controller_decision['learning_rates_before']} -> "
                    f"{controller_decision['learning_rates']}."
                )
            while next_eval_episode <= episodes_completed:
                next_eval_episode += args.eval_interval

        should_checkpoint = episodes_completed >= next_checkpoint_episode
        if should_checkpoint or should_evaluate:
            save_checkpoint(
                agent,
                latest_path,
                episode=episodes_completed,
                run_seed=args.seed,
                best_eval_score=None
                if best_eval_score == -math.inf
                else best_eval_score,
                best_eval_episode=best_eval_episode,
                train_args=args,
                checkpoint_role="latest",
                best_checkpoint_path=output_path,
                episodes_started=episodes_started,
                warm_start_provenance=warm_start_provenance,
                convergence_controller=controller,
            )
            while next_checkpoint_episode <= episodes_completed:
                next_checkpoint_episode += args.checkpoint_interval

        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(collection_metrics, ensure_ascii=False) + "\n")
            for metrics in completed_summaries:
                rendered = metrics.pop("render")
                stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                if rendered is not None:
                    print(rendered)
                episode = int(metrics["episode"])
                if episode % 10 == 0 or episode == start_episode:
                    print(
                        f"Episode {episode:6d} | score={metrics['score']:3d} | "
                        f"rolling100={metrics['rolling_score_100']:.2f} | "
                        f"steps={metrics['steps']:4d} | epsilon={agent.epsilon:.3f} | "
                        f"loss={metrics['avg_loss']} | {metrics['terminal_event']} | "
                        f"env={collection_metrics['env_steps_per_second']:.1f} steps/s | "
                        f"updates={collection_metrics['updates_per_second']:.1f}/s | "
                        f"sample={sampling_seconds:.3f}s | gpu_wait={gpu_wait_seconds:.3f}s"
                    )
        if controller_decision is not None and controller_decision["should_stop"]:
            print(
                f"Early stopping at episode {episodes_completed}; best fixed-suite score "
                f"{best_eval_score:.3f} at {best_eval_episode}; controller decision "
                f"{controller_decision['decision']}."
            )
            stop_requested = True

    save_checkpoint(
        agent,
        latest_path,
        episode=final_episode,
        run_seed=args.seed,
        best_eval_score=None if best_eval_score == -math.inf else best_eval_score,
        best_eval_episode=best_eval_episode,
        train_args=args,
        checkpoint_role="latest",
        best_checkpoint_path=output_path,
        episodes_started=episodes_started,
        warm_start_provenance=warm_start_provenance,
        convergence_controller=controller,
    )
    print(
        f"Training complete. Latest: {latest_path}; "
        f"best: {output_path if best_eval_episode is not None else 'not evaluated in this run'}; "
        f"log: {log_path}."
    )


def train(args: argparse.Namespace | None = None) -> None:
    """Run training and always release resources after its frame is gone."""
    resolved_args = args or parse_args()
    runtime_device = DQNAgent._resolve_device(resolved_args.device)
    try:
        _train(resolved_args)
    finally:
        _release_accelerator_resources(runtime_device)


if __name__ == "__main__":
    train()
