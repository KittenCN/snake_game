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


CHECKPOINT_FORMAT = 4
V3_OBSERVATION_CHANNELS = 20
CONVERGENCE_CONTROLLER_VERSION = 4
INCONCLUSIVE_SCHEDULER_MODES = {"defer_v1", "bounded_probe_v1"}


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
    adaptive_eval_max_episodes: int = 0
    adaptive_eval_growth_factor: float = 2.0
    inconclusive_scheduler_mode: str = "defer_v1"
    bounded_inconclusive_patience: int = 0
    full_eval_confirmation_interval: int = 0
    full_eval_seed_base: int = 1_000_000
    full_eval_max_attempts: int = 0
    reference_score: float | None = None
    reference_scores: list[float] | None = None
    plateau_evaluations: int = 0
    min_lr_evaluations: int = 0
    regression_evaluations: int = 0
    reductions: int = 0
    evaluations: int = 0
    probe_inconclusive_evaluations: int = 0
    evaluation_episodes: int = 0
    evaluation_seconds: float = 0.0
    full_eval_attempts: int = 0
    scheduler_probes: int = 0
    migration_note: str | None = None

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
        self.probe_inconclusive_evaluations = 0

    def set_paired_reference(self, scores: Sequence[float | int]) -> None:
        parsed = [float(value) for value in scores]
        if not parsed or not all(math.isfinite(value) for value in parsed):
            raise ValueError("Paired reference scores must be non-empty and finite")
        self.reference_scores = parsed
        self.regression_evaluations = 0
        self.probe_inconclusive_evaluations = 0

    def record_evaluation_cost(self, episodes: int, seconds: float) -> None:
        if episodes <= 0:
            raise ValueError("Evaluation episode cost must be positive")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Evaluation duration must be finite and non-negative")
        self.evaluation_episodes += int(episodes)
        self.evaluation_seconds += float(seconds)

    def paired_comparison(
        self,
        scores: Sequence[float | int] | None,
        *,
        planned_looks: int = 1,
        reference_scores: Sequence[float | int] | None = None,
    ) -> dict[str, Any] | None:
        resolved_reference = (
            self.reference_scores if reference_scores is None else reference_scores
        )
        if scores is None or resolved_reference is None:
            return None
        candidate = [float(value) for value in scores]
        parsed_reference = [float(value) for value in resolved_reference]
        if len(candidate) > len(parsed_reference):
            raise ValueError(
                "Paired candidate exceeds the complete reference sample count: "
                f"reference={len(parsed_reference)}, candidate={len(candidate)}"
            )
        if not candidate or not all(math.isfinite(value) for value in candidate):
            raise ValueError("Paired candidate scores must be non-empty and finite")
        if planned_looks <= 0:
            raise ValueError("Paired comparison planned looks must be positive")
        if not all(math.isfinite(value) for value in parsed_reference):
            raise ValueError("Paired reference scores must be finite")
        reference_prefix = parsed_reference[: len(candidate)]
        differences = [
            current - reference
            for current, reference in zip(candidate, reference_prefix, strict=True)
        ]
        mean = float(statistics.mean(differences))
        std = float(statistics.stdev(differences)) if len(differences) > 1 else 0.0
        alpha = 0.05
        alpha_each = alpha / planned_looks
        critical_value = statistics.NormalDist().inv_cdf(1.0 - alpha_each / 2.0)
        standard_critical_value = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
        standard_margin = standard_critical_value * std / math.sqrt(len(differences))
        adjusted_margin = critical_value * std / math.sqrt(len(differences))
        ci95_low = mean - standard_margin
        ci95_high = mean + standard_margin
        adjusted_ci_low = mean - adjusted_margin
        adjusted_ci_high = mean + adjusted_margin
        meaningful_delta = max(self.early_stop_delta, self.paired_promotion_min_delta)
        confirmed_improvement = adjusted_ci_low > meaningful_delta
        confirmed_plateau = adjusted_ci_high < meaningful_delta
        statistical_state = (
            "confirmed_improvement"
            if confirmed_improvement
            else "confirmed_plateau"
            if confirmed_plateau
            else "inconclusive"
        )
        return {
            "count": len(differences),
            "mean_delta": mean,
            "std_delta": std,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "adjusted_ci_low": adjusted_ci_low,
            "adjusted_ci_high": adjusted_ci_high,
            "method": "paired_normal_bonferroni_v1",
            "family_confidence": 1.0 - alpha,
            "look_confidence": 1.0 - alpha_each,
            "planned_looks": planned_looks,
            "critical_value": critical_value,
            "min_delta": min(differences),
            "max_delta": max(differences),
            "meaningful_delta": meaningful_delta,
            "statistical_state": statistical_state,
            "promotion_eligible": confirmed_improvement,
            "confirmed_plateau": confirmed_plateau,
            "clear_regression": adjusted_ci_high < -self.regression_stop_delta,
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
        planned_looks: int = 1,
        evaluation_scope: str = "full_evaluation",
        is_max_sample: bool = False,
        comparison_reference_scores: Sequence[float | int] | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(score):
            raise ValueError("Evaluation score must be finite")
        if evaluation_scope not in {
            "full_evaluation",
            "full_confirmation",
            "scheduler_probe",
        }:
            raise ValueError("Unknown evaluation scope")
        before_reference = self.reference_score
        before_lrs = self.learning_rates(optimizer)
        paired = self.paired_comparison(
            sample_scores,
            planned_looks=planned_looks,
            reference_scores=comparison_reference_scores,
        )
        aggregate_significant = (
            before_reference is None
            or score >= before_reference + self.early_stop_delta
        )

        def payload(
            *,
            decision: str,
            significant: bool = False,
            clear_regression: bool = False,
            reduced: bool = False,
            stop: bool = False,
            observation_deferred: bool = False,
            patience_deferred: bool = False,
            bounded_triggered: bool = False,
        ) -> dict[str, Any]:
            statistical_state = (
                str(paired["statistical_state"])
                if paired is not None
                else (
                    "confirmed_improvement"
                    if aggregate_significant
                    else "confirmed_plateau"
                )
            )
            return {
                "score": score,
                "decision": decision,
                "evaluation_scope": evaluation_scope,
                "is_max_sample": bool(is_max_sample),
                "observation_deferred": observation_deferred,
                "significant_improvement": significant,
                "aggregate_significant_improvement": aggregate_significant,
                "paired_comparison": paired,
                "paired_promotion_eligible": bool(
                    evaluation_scope in {"full_evaluation", "full_confirmation"}
                    and paired
                    and paired["promotion_eligible"]
                ),
                "clear_regression": clear_regression,
                "statistical_state": statistical_state,
                "patience_deferred": patience_deferred,
                "regression_evaluations": self.regression_evaluations,
                "lr_reduced": reduced,
                "should_stop": stop,
                "reference_score_before": before_reference,
                "reference_score": self.reference_score,
                "learning_rates_before": before_lrs,
                "learning_rates": self.learning_rates(optimizer),
                "at_min_lr": self.at_min_lr(optimizer),
                "plateau_evaluations": self.plateau_evaluations,
                "min_lr_evaluations": self.min_lr_evaluations,
                "probe_inconclusive_evaluations": self.probe_inconclusive_evaluations,
                "bounded_inconclusive_triggered": bounded_triggered,
                "inconclusive_scheduler_mode": self.inconclusive_scheduler_mode,
                "reductions": self.reductions,
                "evaluations": self.evaluations,
                "evaluation_episodes": self.evaluation_episodes,
                "evaluation_seconds": self.evaluation_seconds,
                "full_eval_attempts": self.full_eval_attempts,
                "full_eval_max_attempts": self.full_eval_max_attempts,
                "scheduler_probes": self.scheduler_probes,
                "migration_note": self.migration_note,
            }

        if defer_reason is not None:
            return payload(
                decision=defer_reason,
                observation_deferred=True,
                patience_deferred=True,
            )
        self.evaluations += 1
        statistical_state = (
            str(paired["statistical_state"])
            if paired is not None
            else (
                "confirmed_improvement"
                if aggregate_significant
                else "confirmed_plateau"
            )
        )

        if evaluation_scope == "scheduler_probe":
            if paired is None:
                raise ValueError("Scheduler probes require a paired reference")
            self.scheduler_probes += 1
            reduced = False
            patience_deferred = True
            bounded_triggered = False
            decision = f"scheduler_probe_{statistical_state}"
            if (
                statistical_state == "inconclusive"
                and self.inconclusive_scheduler_mode == "bounded_probe_v1"
                and self.bounded_inconclusive_patience > 0
            ):
                self.probe_inconclusive_evaluations += 1
                if (
                    self.probe_inconclusive_evaluations
                    >= self.bounded_inconclusive_patience
                ):
                    bounded_triggered = True
                    self.probe_inconclusive_evaluations = 0
            else:
                self.probe_inconclusive_evaluations = 0
            spends_plateau_tick = (
                statistical_state == "confirmed_plateau" or bounded_triggered
            )
            if spends_plateau_tick:
                prefix = (
                    "bounded_probe_inconclusive"
                    if bounded_triggered
                    else "scheduler_probe"
                )
                if not self.scheduler_enabled:
                    decision = f"{prefix}_no_scheduler"
                elif self.at_min_lr(optimizer):
                    decision = f"{prefix}_at_min_lr_deferred"
                else:
                    patience_deferred = False
                    self.plateau_evaluations += 1
                    decision = f"{prefix}_plateau_patience"
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
                        decision = f"{prefix}_lr_reduced"
            return payload(
                decision=decision,
                reduced=reduced,
                patience_deferred=patience_deferred,
                bounded_triggered=bounded_triggered,
            )

        significant = (
            before_reference is None
            or statistical_state == "confirmed_improvement"
            if self.require_paired_promotion and paired is not None
            else aggregate_significant
        )
        bounded_triggered = False
        clear_regression = bool(
            self.require_paired_promotion
            and self.regression_stop_patience > 0
            and paired
            and paired["clear_regression"]
        )
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
        patience_deferred = False

        if significant:
            self.reference_score = (
                score
                if before_reference is None or evaluation_scope == "full_confirmation"
                else max(before_reference, score)
            )
            self.plateau_evaluations = 0
            self.min_lr_evaluations = 0
        elif regression_stop:
            decision = "paired_regression_patience"
            stop = True
        elif clear_regression:
            # Regression has its own consecutive-evidence guard.  It must not
            # also spend plateau/min-LR patience, otherwise one evaluation can
            # advance two independent stop mechanisms.
            decision = "paired_clear_regression"
        elif self.require_paired_promotion and statistical_state == "inconclusive":
            patience_deferred = True
            decision = "paired_inconclusive"
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
        elif evaluation_scope == "full_confirmation":
            # Probe decisions own pre-min-LR scheduling. Full confirmations are
            # reserved for promotion, regression, and conservative min-LR stop.
            decision = "full_confirmation_plateau_no_scheduler_tick"
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

        return payload(
            decision=decision,
            significant=significant,
            clear_regression=clear_regression,
            reduced=reduced,
            stop=stop,
            patience_deferred=patience_deferred,
            bounded_triggered=bounded_triggered,
        )

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
                "adaptive_eval_max_episodes": self.adaptive_eval_max_episodes,
                "adaptive_eval_growth_factor": self.adaptive_eval_growth_factor,
                "inconclusive_scheduler_mode": self.inconclusive_scheduler_mode,
                "bounded_inconclusive_patience": self.bounded_inconclusive_patience,
                "full_eval_confirmation_interval": self.full_eval_confirmation_interval,
                "full_eval_seed_base": self.full_eval_seed_base,
                "full_eval_max_attempts": self.full_eval_max_attempts,
            },
            "state": {
                "reference_score": self.reference_score,
                "reference_scores": self.reference_scores,
                "plateau_evaluations": self.plateau_evaluations,
                "min_lr_evaluations": self.min_lr_evaluations,
                "regression_evaluations": self.regression_evaluations,
                "reductions": self.reductions,
                "evaluations": self.evaluations,
                "probe_inconclusive_evaluations": (
                    self.probe_inconclusive_evaluations
                ),
                "evaluation_episodes": self.evaluation_episodes,
                "evaluation_seconds": self.evaluation_seconds,
                "full_eval_attempts": self.full_eval_attempts,
                "scheduler_probes": self.scheduler_probes,
                "migration_note": self.migration_note,
            },
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Return controller state suitable for high-frequency JSONL records."""
        summary = self.to_dict()
        state = summary["state"]
        reference_scores = state.pop("reference_scores")
        state["reference_scores_count"] = (
            0 if reference_scores is None else len(reference_scores)
        )
        return summary

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationConvergenceController:
        version = payload.get("version")
        if version not in {1, 2, 3, CONVERGENCE_CONTROLLER_VERSION}:
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
            regression_stop_patience=int(config.get("regression_stop_patience", 0)),
            regression_stop_delta=float(config.get("regression_stop_delta", 0.0)),
            adaptive_eval_max_episodes=int(config.get("adaptive_eval_max_episodes", 0)),
            adaptive_eval_growth_factor=float(
                config.get("adaptive_eval_growth_factor", 2.0)
            ),
            inconclusive_scheduler_mode=str(
                config.get("inconclusive_scheduler_mode", "defer_v1")
            ),
            bounded_inconclusive_patience=int(
                config.get("bounded_inconclusive_patience", 0)
            ),
            full_eval_confirmation_interval=int(
                config.get("full_eval_confirmation_interval", 0)
            ),
            full_eval_seed_base=int(config.get("full_eval_seed_base", 1_000_000)),
            full_eval_max_attempts=int(config.get("full_eval_max_attempts", 0)),
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
            probe_inconclusive_evaluations=int(
                state.get("probe_inconclusive_evaluations", 0)
            ),
            evaluation_episodes=int(state.get("evaluation_episodes", 0)),
            evaluation_seconds=float(state.get("evaluation_seconds", 0.0)),
            full_eval_attempts=int(state.get("full_eval_attempts", 0)),
            scheduler_probes=int(state.get("scheduler_probes", 0)),
            migration_note=(
                None
                if state.get("migration_note") is None
                else str(state["migration_note"])
            ),
        )
        if version in {1, 2}:
            controller.plateau_evaluations = 0
            controller.min_lr_evaluations = 0
            controller.regression_evaluations = 0
            controller.migration_note = (
                f"v{version}_patience_counters_cleared_for_bonferroni_v3"
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
            raise RuntimeError(
                "Controller early-stop delta must be finite and non-negative"
            )
        if (
            not math.isfinite(self.paired_promotion_min_delta)
            or self.paired_promotion_min_delta < 0
        ):
            raise RuntimeError(
                "Controller paired-promotion delta must be finite and non-negative"
            )
        if self.regression_stop_patience < 0:
            raise RuntimeError("Controller regression patience must be non-negative")
        if (
            not math.isfinite(self.regression_stop_delta)
            or self.regression_stop_delta < 0
        ):
            raise RuntimeError(
                "Controller regression-stop delta must be finite and non-negative"
            )
        if self.adaptive_eval_max_episodes < 0:
            raise RuntimeError(
                "Controller adaptive evaluation maximum must be non-negative"
            )
        if (
            not math.isfinite(self.adaptive_eval_growth_factor)
            or self.adaptive_eval_growth_factor <= 1.0
        ):
            raise RuntimeError(
                "Controller adaptive evaluation growth factor must be finite and greater than 1"
            )
        if self.inconclusive_scheduler_mode not in INCONCLUSIVE_SCHEDULER_MODES:
            raise RuntimeError("Controller inconclusive scheduler mode is unsupported")
        if self.bounded_inconclusive_patience < 0:
            raise RuntimeError(
                "Controller bounded-inconclusive patience must be non-negative"
            )
        if (
            self.inconclusive_scheduler_mode == "defer_v1"
            and self.bounded_inconclusive_patience != 0
        ):
            raise RuntimeError(
                "Controller defer_v1 mode requires zero bounded-inconclusive patience"
            )
        if (
            self.inconclusive_scheduler_mode == "bounded_probe_v1"
            and self.bounded_inconclusive_patience <= 0
        ):
            raise RuntimeError(
                "Controller bounded_probe_v1 mode requires positive patience"
            )
        if self.full_eval_confirmation_interval < 0:
            raise RuntimeError(
                "Controller full evaluation confirmation interval must be non-negative"
            )
        if self.full_eval_seed_base < 0:
            raise RuntimeError("Controller full evaluation seed base must be non-negative")
        if self.full_eval_max_attempts < 0:
            raise RuntimeError("Controller full evaluation max attempts must be non-negative")
        if self.full_eval_confirmation_interval > 0 and (
            not self.require_paired_promotion or self.full_eval_max_attempts <= 0
        ):
            raise RuntimeError(
                "Controller gated full evaluation requires paired promotion and alpha budget"
            )
        if self.full_eval_attempts > self.full_eval_max_attempts:
            raise RuntimeError("Controller full evaluation attempts exceed alpha budget")
        if self.reference_score is not None and not math.isfinite(self.reference_score):
            raise RuntimeError("Controller reference score must be finite or null")
        if self.reference_scores is not None and (
            not self.reference_scores
            or not all(math.isfinite(value) for value in self.reference_scores)
        ):
            raise RuntimeError("Controller paired reference scores must be finite")
        if (
            min(
                self.plateau_evaluations,
                self.min_lr_evaluations,
                self.regression_evaluations,
                self.reductions,
                self.evaluations,
                self.probe_inconclusive_evaluations,
                self.evaluation_episodes,
                self.full_eval_attempts,
                self.scheduler_probes,
            )
            < 0
        ):
            raise RuntimeError("Controller counters must be non-negative")
        if not math.isfinite(self.evaluation_seconds) or self.evaluation_seconds < 0:
            raise RuntimeError(
                "Controller cumulative evaluation duration must be finite and non-negative"
            )


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
        "--policy-anchor-final-weight",
        type=float,
        default=None,
        help=(
            "Final policy-anchor weight after linear behavior-step decay; omitted "
            "keeps the initial weight"
        ),
    )
    parser.add_argument(
        "--policy-anchor-decay-steps",
        type=int,
        default=0,
        help="Behavior steps for linear anchor decay; 0 keeps a constant weight",
    )
    parser.add_argument(
        "--teacher-replay-steps",
        type=int,
        default=0,
        help="Initial environment transitions collected greedily from the frozen teacher",
    )
    parser.add_argument(
        "--demonstration-capacity",
        type=int,
        default=0,
        help="Dedicated persistent replay capacity for completed high-score trajectories",
    )
    parser.add_argument(
        "--demonstration-batch-fraction",
        type=float,
        default=0.0,
        help="Fraction of each learner batch reserved for demonstration replay",
    )
    parser.add_argument(
        "--elite-demonstration-batch-fraction",
        type=float,
        default=0.0,
        help="Fraction of each full batch reserved for the elite score/return stratum",
    )
    parser.add_argument("--demonstration-min-score", type=float, default=4.0)
    parser.add_argument("--demonstration-min-return", type=float, default=0.0)
    parser.add_argument("--demonstration-elite-score", type=float, default=6.0)
    parser.add_argument("--demonstration-elite-return", type=float, default=20.0)
    parser.add_argument(
        "--imitation-loss-weight",
        type=float,
        default=0.0,
        help="Weight for large-margin successful-action imitation loss",
    )
    parser.add_argument("--imitation-margin", type=float, default=0.8)
    parser.add_argument(
        "--demonstration-terminal-exclusion-steps",
        type=int,
        default=1,
        help=(
            "Number of final trajectory actions excluded from imitation; 1 preserves "
            "the historical terminal-action exclusion"
        ),
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
    parser.add_argument(
        "--idle-limit-floor-steps",
        type=int,
        default=0,
        help=(
            "Minimum idle budget independent of score; use width*height (144 on "
            "12x12) to permit a full safe board traversal"
        ),
    )
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
    parser.add_argument(
        "--action-mask-mode",
        choices=["legal_v1", "one_step_survival_v1", "topology_survival_v1"],
        default="topology_survival_v1",
        help="Versioned behavior/target action-mask contract",
    )
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
    parser.add_argument(
        "--collection-log-interval",
        type=int,
        default=10,
        help="Write collection JSONL every N collections; evaluations/final state always log",
    )
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
    parser.add_argument(
        "--adaptive-eval-max-episodes",
        type=int,
        default=0,
        help=(
            "Maximum paired evaluation episodes; 0 disables adaptive expansion "
            "and uses --eval-episodes"
        ),
    )
    parser.add_argument("--adaptive-eval-growth-factor", type=float, default=2.0)
    parser.add_argument(
        "--inconclusive-scheduler-mode",
        choices=sorted(INCONCLUSIVE_SCHEDULER_MODES),
        default="defer_v1",
        help=(
            "defer_v1 never spends patience for paired ambiguity; "
            "bounded_probe_v1 may spend only pre-min-LR plateau patience"
        ),
    )
    parser.add_argument(
        "--bounded-inconclusive-patience",
        type=int,
        default=0,
        help="Consecutive inconclusive scheduler probes per pre-min-LR plateau tick",
    )
    parser.add_argument(
        "--full-eval-confirmation-interval",
        type=int,
        default=0,
        help=(
            "Run the maximum paired suite every N scheduler probes; 0 preserves "
            "the adaptive evaluation pipeline"
        ),
    )
    parser.add_argument(
        "--full-eval-seed-base",
        type=int,
        default=1_000_000,
        help="Disjoint seed namespace reserved for fresh full paired attempts",
    )
    parser.add_argument(
        "--full-eval-max-attempts",
        type=int,
        default=0,
        help="Pre-registered family-wise alpha budget for full paired attempts",
    )
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
    if args.network_version < 3:
        if (
            "--action-mask-mode" in args._provided_options
            and args.action_mask_mode != "legal_v1"
        ):
            parser.error("survival action masks require network-version 3")
        args.action_mask_mode = "legal_v1"
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error("episodes and eval-episodes must be positive")
    if args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("eval-interval and checkpoint-interval must be positive")
    if args.collection_log_interval <= 0:
        parser.error("collection-log-interval must be positive")
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
    if args.idle_limit_floor_steps < 0:
        parser.error("idle-limit-floor-steps must be non-negative")
    if args.max_idle_steps == 0 and args.idle_limit_floor_steps > 0:
        parser.error("idle-limit-floor-steps requires max-idle-steps to be positive")
    if not math.isfinite(args.policy_anchor_weight) or args.policy_anchor_weight < 0:
        parser.error("policy-anchor-weight must be finite and non-negative")
    if args.policy_anchor_final_weight is None:
        args.policy_anchor_final_weight = args.policy_anchor_weight
    if (
        not math.isfinite(args.policy_anchor_final_weight)
        or args.policy_anchor_final_weight < 0
        or args.policy_anchor_final_weight > args.policy_anchor_weight
    ):
        parser.error(
            "policy-anchor-final-weight must be finite and between zero and "
            "policy-anchor-weight"
        )
    if args.policy_anchor_decay_steps < 0:
        parser.error("policy-anchor-decay-steps must be non-negative")
    if args.demonstration_capacity < 0:
        parser.error("demonstration-capacity must be non-negative")
    if args.demonstration_terminal_exclusion_steps < 0:
        parser.error("demonstration-terminal-exclusion-steps must be non-negative")
    if not 0.0 <= args.demonstration_batch_fraction < 1.0:
        parser.error("demonstration-batch-fraction must be in [0, 1)")
    if not (
        0.0
        <= args.elite_demonstration_batch_fraction
        <= args.demonstration_batch_fraction
    ):
        parser.error(
            "elite-demonstration-batch-fraction must be between zero and "
            "demonstration-batch-fraction"
        )
    requested_demo_rows = int(
        round(args.batch_size * args.demonstration_batch_fraction)
    )
    if args.demonstration_batch_fraction > 0 and not (
        1 <= requested_demo_rows < args.batch_size
    ):
        parser.error(
            "demonstration-batch-fraction must reserve at least one demo row and "
            "one regular replay row for the configured batch-size"
        )
    requested_elite_rows = int(
        round(args.batch_size * args.elite_demonstration_batch_fraction)
    )
    if args.elite_demonstration_batch_fraction > 0 and requested_elite_rows < 1:
        parser.error(
            "elite-demonstration-batch-fraction must reserve at least one row for "
            "the configured batch-size"
        )
    if args.demonstration_batch_fraction > 0 and args.demonstration_capacity == 0:
        parser.error(
            "demonstration-capacity must be positive when demonstration sampling is enabled"
        )
    if args.imitation_loss_weight > 0 and args.demonstration_batch_fraction == 0:
        parser.error(
            "demonstration-batch-fraction must be positive when imitation loss is enabled"
        )
    if not math.isfinite(args.imitation_loss_weight) or args.imitation_loss_weight < 0:
        parser.error("imitation-loss-weight must be finite and non-negative")
    if not math.isfinite(args.imitation_margin) or args.imitation_margin <= 0:
        parser.error("imitation-margin must be finite and positive")
    demonstration_thresholds = (
        args.demonstration_min_score,
        args.demonstration_min_return,
        args.demonstration_elite_score,
        args.demonstration_elite_return,
    )
    if not all(math.isfinite(value) for value in demonstration_thresholds):
        parser.error("demonstration score/return thresholds must be finite")
    if args.demonstration_elite_score < args.demonstration_min_score:
        parser.error(
            "demonstration-elite-score must not be below demonstration-min-score"
        )
    if args.demonstration_elite_return < args.demonstration_min_return:
        parser.error(
            "demonstration-elite-return must not be below demonstration-min-return"
        )
    if args.early_stop_patience < 0 or args.lr_plateau_patience < 0:
        parser.error("early-stop-patience and lr-plateau-patience must be non-negative")
    if not math.isfinite(args.early_stop_delta) or args.early_stop_delta < 0:
        parser.error("early-stop-delta must be finite and non-negative")
    if args.regression_stop_patience < 0:
        parser.error("regression-stop-patience must be non-negative")
    if not math.isfinite(args.regression_stop_delta) or args.regression_stop_delta < 0:
        parser.error("regression-stop-delta must be finite and non-negative")
    if (
        not math.isfinite(args.paired_promotion_min_delta)
        or args.paired_promotion_min_delta < 0
    ):
        parser.error("paired-promotion-min-delta must be finite and non-negative")
    if args.adaptive_eval_max_episodes < 0:
        parser.error("adaptive-eval-max-episodes must be non-negative")
    if (
        args.adaptive_eval_max_episodes > 0
        and args.adaptive_eval_max_episodes < args.eval_episodes
    ):
        parser.error("adaptive-eval-max-episodes must be at least eval-episodes")
    if (
        not math.isfinite(args.adaptive_eval_growth_factor)
        or args.adaptive_eval_growth_factor <= 1.0
    ):
        parser.error("adaptive-eval-growth-factor must be finite and greater than 1")
    if (
        args.adaptive_eval_max_episodes > 0
        and not args.require_paired_promotion
        and args.resume_from is None
    ):
        parser.error("adaptive evaluation requires --require-paired-promotion")
    if args.bounded_inconclusive_patience < 0:
        parser.error("bounded-inconclusive-patience must be non-negative")
    if (
        args.inconclusive_scheduler_mode == "defer_v1"
        and args.bounded_inconclusive_patience != 0
        and args.resume_from is None
    ):
        parser.error(
            "bounded-inconclusive-patience requires "
            "--inconclusive-scheduler-mode bounded_probe_v1"
        )
    if (
        args.inconclusive_scheduler_mode == "bounded_probe_v1"
        and args.bounded_inconclusive_patience <= 0
        and args.resume_from is None
    ):
        parser.error(
            "bounded_probe_v1 requires positive --bounded-inconclusive-patience"
        )
    if args.bounded_inconclusive_patience > 0 and (
        not args.require_paired_promotion or args.lr_plateau_patience <= 0
    ) and args.resume_from is None:
        parser.error(
            "bounded inconclusive scheduling requires paired promotion and LR scheduling"
        )
    if args.full_eval_confirmation_interval < 0:
        parser.error("full-eval-confirmation-interval must be non-negative")
    if args.full_eval_confirmation_interval > 0 and (
        not args.require_paired_promotion
        or args.adaptive_eval_max_episodes <= args.eval_episodes
        or args.full_eval_max_attempts <= 0
    ) and args.resume_from is None:
        parser.error(
            "gated full evaluation requires paired promotion, adaptive max > base, "
            "and positive full-eval-max-attempts"
        )
    if args.full_eval_seed_base < 0:
        parser.error("full-eval-seed-base must be non-negative")
    if args.full_eval_max_attempts < 0:
        parser.error("full-eval-max-attempts must be non-negative")
    baseline_seed_stop = args.eval_seed_base + max(
        args.eval_episodes, args.adaptive_eval_max_episodes
    )
    full_seed_stop = (
        args.full_eval_seed_base
        + args.full_eval_max_attempts
        * max(args.eval_episodes, args.adaptive_eval_max_episodes)
    )
    if args.full_eval_confirmation_interval > 0 and not (
        full_seed_stop <= args.eval_seed_base
        or args.full_eval_seed_base >= baseline_seed_stop
    ):
        parser.error("baseline/probe and full evaluation seed namespaces must not overlap")
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
        if agent.action_mask_mode == "topology_survival_v1":
            return list(env.relative_topology_survival_mask())
        if agent.action_mask_mode == "one_step_survival_v1":
            return list(env.relative_survival_mask())
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


def evaluation_samples(evaluation: dict[str, Any], group: str) -> list[float]:
    """Read per-seed samples while remaining compatible with legacy test/eval payloads."""
    raw = evaluation.get(f"{group}_samples")
    if isinstance(raw, list) and raw:
        return [float(value) for value in raw]
    episodes = int(evaluation.get("episodes", 1))
    distribution_group = "steps" if group == "step" else group
    mean = float(evaluation[distribution_group]["mean"])
    return [mean] * max(1, episodes)


def adaptive_evaluation_plan(
    base_episodes: int, max_episodes: int, growth_factor: float
) -> list[int]:
    """Return deterministic cumulative look sizes, including base and maximum."""
    resolved_max = max_episodes if max_episodes > 0 else base_episodes
    if base_episodes <= 0 or resolved_max < base_episodes:
        raise ValueError("Adaptive evaluation requires 0 < base <= max episodes")
    if not math.isfinite(growth_factor) or growth_factor <= 1.0:
        raise ValueError("Adaptive evaluation growth factor must exceed 1")
    plan = [base_episodes]
    while plan[-1] < resolved_max:
        grown = max(plan[-1] + 1, math.ceil(plan[-1] * growth_factor))
        plan.append(min(resolved_max, grown))
    return plan


def _merge_evaluation_chunks(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("At least one evaluation chunk is required")
    seeds: list[int] = []
    reward_samples: list[float] = []
    score_samples: list[float] = []
    step_samples: list[float] = []
    terminal_events: Counter[str] = Counter()
    truncated_count = 0
    for chunk in chunks:
        seeds.extend(int(value) for value in chunk.get("seeds", ()))
        reward_samples.extend(evaluation_samples(chunk, "reward"))
        score_samples.extend(evaluation_samples(chunk, "score"))
        step_samples.extend(evaluation_samples(chunk, "step"))
        terminal_events.update(chunk.get("terminal_events", {}))
        truncated_count += int(chunk.get("truncated_count", 0))

    def distribution(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        margin = 1.96 * std / math.sqrt(len(array))
        return {
            "mean": float(array.mean()),
            "std": std,
            "ci95_low": float(array.mean()) - margin,
            "ci95_high": float(array.mean()) + margin,
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "reward": distribution(reward_samples),
        "score": distribution(score_samples),
        "steps": distribution(step_samples),
        "seeds": seeds,
        "reward_samples": reward_samples,
        "score_samples": score_samples,
        "step_samples": [int(value) for value in step_samples],
        "terminal_events": dict(sorted(terminal_events.items())),
        "truncated_count": truncated_count,
        "episodes": len(seeds),
    }


def evaluate_adaptive_paired(
    agent: DQNAgent,
    game_config: GameConfig,
    seeds: Sequence[int],
    max_steps: int,
    controller: EvaluationConvergenceController,
    *,
    base_episodes: int,
    max_episodes: int,
    growth_factor: float,
    adaptive_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate disjoint seed chunks until paired evidence reaches a terminal state."""
    plan = adaptive_evaluation_plan(base_episodes, max_episodes, growth_factor)
    if not adaptive_enabled:
        plan = plan[:1]
    planned_looks = len(
        adaptive_evaluation_plan(base_episodes, max_episodes, growth_factor)
    )
    chunks: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    evaluated = 0
    evaluation: dict[str, Any] | None = None
    paired: dict[str, Any] | None = None
    for stage, target in enumerate(plan, start=1):
        chunk_seeds = seeds[evaluated:target]
        if len(chunk_seeds) != target - evaluated:
            raise ValueError("Adaptive evaluation seed plan is incomplete")
        chunks.append(evaluate_agent(agent, game_config, chunk_seeds, max_steps))
        evaluated = target
        evaluation = _merge_evaluation_chunks(chunks)
        paired = controller.paired_comparison(
            evaluation_samples(evaluation, "score"), planned_looks=planned_looks
        )
        state = paired["statistical_state"] if paired else "unpaired_reference"
        stage_records.append(
            {
                "stage": stage,
                "actual_episodes": evaluated,
                "planned_episodes": target,
                "statistical_state": state,
                "clear_regression": bool(paired and paired["clear_regression"]),
            }
        )
        if not adaptive_enabled:
            break
        if paired and (
            paired["clear_regression"]
            or paired["statistical_state"] == "confirmed_plateau"
        ):
            break
        # Both inconclusive evidence and a prefix improvement continue.  A
        # promotion is allowed only after the complete reference-sized suite.
    if evaluation is None:
        raise RuntimeError("Adaptive evaluation produced no result")
    metadata = {
        "actual_episodes": int(evaluation["episodes"]),
        "planned_episodes": stage_records[-1]["planned_episodes"],
        "max_episodes": max_episodes if max_episodes > 0 else base_episodes,
        "expansion_stage": stage_records[-1]["stage"],
        "planned_looks": planned_looks,
        "stages": stage_records,
        "statistical_method": (
            paired["method"] if paired else "paired_normal_bonferroni_v1"
        ),
        "family_confidence": paired["family_confidence"] if paired else 0.95,
        "look_confidence": (
            paired["look_confidence"] if paired else 1.0 - 0.05 / planned_looks
        ),
        "statistical_state": (
            paired["statistical_state"] if paired else "unpaired_reference"
        ),
    }
    return evaluation, metadata


def evaluate_scheduler_probe(
    agent: DQNAgent,
    game_config: GameConfig,
    seeds: Sequence[int],
    max_steps: int,
    controller: EvaluationConvergenceController,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a cheap paired probe that can screen scheduling but never promote."""
    evaluation = evaluate_agent(agent, game_config, seeds, max_steps)
    score_samples = evaluation_samples(evaluation, "score")
    paired = controller.paired_comparison(score_samples, planned_looks=1)
    if paired is None:
        raise RuntimeError("Scheduler probe requires an immutable paired reference")
    promotion_candidate = bool(
        paired["mean_delta"] > paired["meaningful_delta"]
        and paired["adjusted_ci_high"] > paired["meaningful_delta"]
    )
    return evaluation, {
        "evaluation_scope": "scheduler_probe",
        "actual_episodes": len(score_samples),
        "planned_episodes": len(seeds),
        "max_episodes": len(seeds),
        "expansion_stage": 1,
        "planned_looks": 1,
        "stages": [
            {
                "stage": 1,
                "actual_episodes": len(score_samples),
                "planned_episodes": len(seeds),
                "statistical_state": paired["statistical_state"],
                "clear_regression": False,
            }
        ],
        "statistical_method": paired["method"],
        "family_confidence": paired["family_confidence"],
        "look_confidence": paired["look_confidence"],
        "statistical_state": paired["statistical_state"],
        "promotion_candidate": promotion_candidate,
        "probe_comparison": paired,
    }


def reserve_full_evaluation_attempt(
    controller: EvaluationConvergenceController,
    *,
    episodes: int,
) -> tuple[int, list[int]] | None:
    """Reserve one pre-registered, non-overlapping full-suite seed block."""
    if episodes <= 0:
        raise ValueError("Full evaluation episodes must be positive")
    if controller.full_eval_attempts >= controller.full_eval_max_attempts:
        return None
    attempt_index = controller.full_eval_attempts
    start = controller.full_eval_seed_base + attempt_index * episodes
    controller.full_eval_attempts += 1
    return attempt_index, list(range(start, start + episodes))


def evaluate_fresh_full_pair(
    candidate: DQNAgent,
    reference: DQNAgent,
    game_config: GameConfig,
    seeds: Sequence[int],
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate candidate and immutable best on the exact same fresh full suite."""
    reference_evaluation = evaluate_agent(reference, game_config, seeds, max_steps)
    candidate_evaluation = evaluate_agent(candidate, game_config, seeds, max_steps)
    if candidate_evaluation.get("seeds") != reference_evaluation.get("seeds"):
        raise RuntimeError("Fresh full paired evaluations used different seed ordering")
    return candidate_evaluation, reference_evaluation


def load_policy_evaluation_reference(
    checkpoint_path: Path,
    candidate: DQNAgent,
    game_config: GameConfig,
) -> DQNAgent:
    """Load an authenticated immutable-best policy without perturbing RNG streams."""
    metadata_path = sidecar_path(checkpoint_path)
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Immutable best checkpoint metadata is missing: {metadata_path}"
        )
    try:
        with metadata_path.open("r", encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Immutable best checkpoint metadata is unreadable: {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("Immutable best checkpoint metadata must be a JSON object.")
    expected_sha256 = metadata.get("checkpoint_sha256")
    referenced_path = metadata.get("best_checkpoint_path")
    if (
        metadata.get("checkpoint_role") != "best_eval"
        or not isinstance(expected_sha256, str)
        or metadata.get("best_checkpoint_sha256") != expected_sha256
        or not referenced_path
        or Path(referenced_path).resolve() != checkpoint_path.resolve()
    ):
        raise RuntimeError(
            "Immutable evaluation reference must have an authenticated self-linked "
            "best_eval sidecar."
        )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        []
        if candidate.device.type != "cuda"
        else [
            candidate.device.index
            if candidate.device.index is not None
            else torch.cuda.current_device()
        ]
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            reference = DQNAgent.from_policy_checkpoint(
                str(checkpoint_path),
                target_game_config=game_config,
                device=str(candidate.device),
                expected_sha256=expected_sha256,
                agent_options={"action_mask_mode": candidate.action_mask_mode},
            )
            DQNAgent.validate_policy_sidecar_identity(
                metadata, reference.policy_transfer_provenance or {}
            )
            return reference
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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
            "paired_score_ci95"
            if train_args.require_paired_promotion
            else "raw_score_mean"
        ),
        "safety_fallback": False,
        "run_seed": train_args.seed,
        "eval_seed_base": train_args.eval_seed_base,
        "eval_episodes": train_args.eval_episodes,
        "adaptive_eval_max_episodes": train_args.adaptive_eval_max_episodes,
        "adaptive_eval_growth_factor": train_args.adaptive_eval_growth_factor,
        "full_eval_confirmation_interval": train_args.full_eval_confirmation_interval,
        "inconclusive_scheduler_mode": train_args.inconclusive_scheduler_mode,
        "bounded_inconclusive_patience": train_args.bounded_inconclusive_patience,
        "full_eval_seed_base": train_args.full_eval_seed_base,
        "full_eval_max_attempts": train_args.full_eval_max_attempts,
        "game_config": asdict(game_config),
        "action_mask_mode": train_args.action_mask_mode,
    }
    if train_args.require_paired_promotion:
        identity["paired_promotion_min_delta"] = train_args.paired_promotion_min_delta
    return identity


def effective_agent_config(agent: DQNAgent) -> dict[str, Any]:
    learning_rates = [float(group["lr"]) for group in agent.optimizer.param_groups]
    return {
        "network_version": agent.network_version,
        "action_mask_mode": agent.action_mask_mode,
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
        "policy_anchor_final_weight": agent.policy_anchor_final_weight,
        "policy_anchor_decay_steps": agent.policy_anchor_decay_steps,
        "effective_policy_anchor_weight": agent.effective_policy_anchor_weight,
        "teacher_replay_steps": agent.teacher_replay_steps,
        "demonstration_capacity": agent.demonstration_capacity,
        "demonstration_batch_fraction": agent.demonstration_batch_fraction,
        "elite_demonstration_batch_fraction": agent.elite_demonstration_batch_fraction,
        "demonstration_min_score": agent.demonstration_min_score,
        "demonstration_min_return": agent.demonstration_min_return,
        "demonstration_elite_score": agent.demonstration_elite_score,
        "demonstration_elite_return": agent.demonstration_elite_return,
        "imitation_loss_weight": agent.imitation_loss_weight,
        "imitation_margin": agent.imitation_margin,
        "demonstration_terminal_exclusion_steps": (
            agent.demonstration_terminal_exclusion_steps
        ),
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
    pending_best_promotion: dict[str, Any] | None = None,
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
        "action_mask_mode": agent.action_mask_mode,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
        "epsilon": agent.epsilon,
        "replay_size_at_save": len(agent.replay_buffer),
        "demonstration_replay_size_at_save": (
            len(agent.demonstration_replay)
            if agent.demonstration_replay is not None
            else 0
        ),
        "demonstration_replay_elite_count_at_save": (
            agent.demonstration_replay.elite_demonstration_count
            if agent.demonstration_replay is not None
            else 0
        ),
        "demonstration_trajectories_seen_lifetime": (
            agent.demonstration_trajectories_seen
        ),
        "demonstration_transitions_promoted_lifetime": (
            agent.demonstration_transitions_promoted
        ),
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
        "policy_transfer_provenance": agent.policy_transfer_provenance,
        "pending_best_promotion": pending_best_promotion,
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
        "action_mask_mode": agent.action_mask_mode,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key, "legal_v1" if key == "action_mask_mode" else None)
        != value
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
        "--action-mask-mode": (args.action_mask_mode, agent.action_mask_mode),
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
        "--policy-anchor-final-weight": (
            args.policy_anchor_final_weight,
            agent.policy_anchor_final_weight,
        ),
        "--policy-anchor-decay-steps": (
            args.policy_anchor_decay_steps,
            agent.policy_anchor_decay_steps,
        ),
        "--teacher-replay-steps": (
            args.teacher_replay_steps,
            agent.teacher_replay_steps,
        ),
        "--demonstration-capacity": (
            args.demonstration_capacity,
            agent.demonstration_capacity,
        ),
        "--demonstration-batch-fraction": (
            args.demonstration_batch_fraction,
            agent.demonstration_batch_fraction,
        ),
        "--elite-demonstration-batch-fraction": (
            args.elite_demonstration_batch_fraction,
            agent.elite_demonstration_batch_fraction,
        ),
        "--demonstration-min-score": (
            args.demonstration_min_score,
            agent.demonstration_min_score,
        ),
        "--demonstration-min-return": (
            args.demonstration_min_return,
            agent.demonstration_min_return,
        ),
        "--demonstration-elite-score": (
            args.demonstration_elite_score,
            agent.demonstration_elite_score,
        ),
        "--demonstration-elite-return": (
            args.demonstration_elite_return,
            agent.demonstration_elite_return,
        ),
        "--imitation-loss-weight": (
            args.imitation_loss_weight,
            agent.imitation_loss_weight,
        ),
        "--imitation-margin": (args.imitation_margin, agent.imitation_margin),
        "--demonstration-terminal-exclusion-steps": (
            args.demonstration_terminal_exclusion_steps,
            agent.demonstration_terminal_exclusion_steps,
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
    if "--policy-anchor-weight" not in provided:
        args.policy_anchor_weight = agent.policy_anchor_weight
    if "--policy-anchor-final-weight" not in provided:
        args.policy_anchor_final_weight = agent.policy_anchor_final_weight
    if "--policy-anchor-decay-steps" not in provided:
        args.policy_anchor_decay_steps = agent.policy_anchor_decay_steps
    if "--demonstration-terminal-exclusion-steps" not in provided:
        args.demonstration_terminal_exclusion_steps = (
            agent.demonstration_terminal_exclusion_steps
        )
    if "--action-mask-mode" not in provided:
        args.action_mask_mode = agent.action_mask_mode


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
    "--adaptive-eval-max-episodes": "adaptive_eval_max_episodes",
    "--adaptive-eval-growth-factor": "adaptive_eval_growth_factor",
    "--inconclusive-scheduler-mode": "inconclusive_scheduler_mode",
    "--bounded-inconclusive-patience": "bounded_inconclusive_patience",
    "--full-eval-confirmation-interval": "full_eval_confirmation_interval",
    "--full-eval-seed-base": "full_eval_seed_base",
    "--full-eval-max-attempts": "full_eval_max_attempts",
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
        adaptive_eval_max_episodes=args.adaptive_eval_max_episodes,
        adaptive_eval_growth_factor=args.adaptive_eval_growth_factor,
        inconclusive_scheduler_mode=args.inconclusive_scheduler_mode,
        bounded_inconclusive_patience=args.bounded_inconclusive_patience,
        full_eval_confirmation_interval=args.full_eval_confirmation_interval,
        full_eval_seed_base=args.full_eval_seed_base,
        full_eval_max_attempts=args.full_eval_max_attempts,
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
        return _new_convergence_controller(args, reference_score=legacy_reference_score)
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
            conflicts.append(f"{option}=requested:{requested!r}/checkpoint:{stored!r}")
        elif option not in provided:
            setattr(args, attribute, stored)
    if conflicts:
        raise RuntimeError(
            "Explicit scheduler/early-stop options conflict with the resumed "
            "convergence controller: " + ", ".join(conflicts)
        )
    if (
        restored.require_paired_promotion
        and legacy_reference_score is not None
        and not args.reset_best_evaluation
    ):
        expected_reference_count = (
            args.eval_episodes
            if restored.full_eval_confirmation_interval > 0
            else restored.adaptive_eval_max_episodes
            if restored.adaptive_eval_max_episodes > 0
            else args.eval_episodes
        )
        actual_reference_count = (
            0 if restored.reference_scores is None else len(restored.reference_scores)
        )
        if actual_reference_count != expected_reference_count:
            raise RuntimeError(
                "Resume paired reference cache length conflicts with the evaluation "
                f"identity: expected={expected_reference_count}, "
                f"actual={actual_reference_count}."
            )
        if restored.reference_score is None or not math.isclose(
            restored.reference_score,
            legacy_reference_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Resume paired reference score conflicts with the immutable best score."
            )
        if restored.full_eval_confirmation_interval == 0:
            reference_mean = float(statistics.mean(restored.reference_scores or ()))
            if not math.isclose(
                reference_mean,
                restored.reference_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Resume paired reference sample mean conflicts with the immutable "
                    "best score."
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


def _best_promotion_marker(
    output_path: Path, *, score: float, episode: int
) -> dict[str, Any]:
    """Describe a durable, already-approved best promotion transaction."""
    previous_sha256 = _sha256(output_path) if output_path.is_file() else None
    return {
        "schema_version": 1,
        "best_checkpoint_path": str(output_path.resolve()),
        "best_eval_score": float(score),
        "best_eval_episode": int(episode),
        "previous_best_checkpoint_sha256": previous_sha256,
    }


def reconcile_pending_best_promotion(
    metadata: dict[str, Any],
    agent: DQNAgent,
    controller: EvaluationConvergenceController,
    args: argparse.Namespace,
    *,
    resume_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Finish an approved best promotion without rolling back latest training state."""
    pending = metadata.get("pending_best_promotion")
    if pending is None:
        return metadata
    if not isinstance(pending, dict) or pending.get("schema_version") != 1:
        raise RuntimeError("Resume metadata contains an unsupported best promotion marker.")
    try:
        score = float(pending["best_eval_score"])
        episode = int(pending["best_eval_episode"])
        pending_path = Path(pending["best_checkpoint_path"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Resume metadata contains an invalid best promotion marker.") from exc
    previous_sha256 = pending.get("previous_best_checkpoint_sha256")
    if (
        not math.isfinite(score)
        or episode < 0
        or pending_path != output_path.resolve()
        or metadata.get("best_eval_score") != score
        or metadata.get("best_eval_episode") != episode
        or metadata.get("best_checkpoint_sha256") != previous_sha256
        or (
            previous_sha256 is not None
            and (
                not isinstance(previous_sha256, str)
                or len(previous_sha256) != 64
                or any(character not in "0123456789abcdef" for character in previous_sha256)
            )
        )
    ):
        raise RuntimeError("Resume best promotion marker conflicts with latest state.")

    best_metadata: dict[str, Any] | None = None
    best_meta_path = sidecar_path(output_path)
    if best_meta_path.is_file():
        try:
            with best_meta_path.open("r", encoding="utf-8-sig") as stream:
                value = json.load(stream)
            if isinstance(value, dict):
                best_metadata = value
        except (OSError, json.JSONDecodeError):
            best_metadata = None
    current_sha256 = _sha256(output_path) if output_path.is_file() else None
    promotion_complete = bool(
        current_sha256
        and best_metadata
        and best_metadata.get("checkpoint_role") == "best_eval"
        and best_metadata.get("checkpoint_sha256") == current_sha256
        and best_metadata.get("best_checkpoint_sha256") == current_sha256
        and best_metadata.get("best_eval_score") == score
        and best_metadata.get("best_eval_episode") == episode
    )
    if not promotion_complete:
        sidecar_sha256 = (
            best_metadata.get("checkpoint_sha256") if best_metadata else None
        )
        known_interrupted_state = (
            previous_sha256 is None
            or current_sha256 == previous_sha256
            or sidecar_sha256 == previous_sha256
        )
        if not known_interrupted_state:
            raise RuntimeError(
                "Pending best promotion found an unknown best artifact; refusing to overwrite it."
            )
        save_checkpoint(
            agent,
            output_path,
            episode=episode,
            run_seed=int(metadata.get("run_seed", args.seed)),
            best_eval_score=score,
            best_eval_episode=episode,
            train_args=args,
            checkpoint_role="best_eval",
            best_checkpoint_path=output_path,
            episodes_started=int(metadata.get("episodes_started", episode)),
            warm_start_provenance=metadata.get("warm_start_provenance"),
            convergence_controller=controller,
        )

    save_checkpoint(
        agent,
        resume_path,
        episode=int(metadata.get("episodes_completed", episode)),
        run_seed=int(metadata.get("run_seed", args.seed)),
        best_eval_score=score,
        best_eval_episode=episode,
        train_args=args,
        checkpoint_role="latest",
        best_checkpoint_path=output_path,
        episodes_started=int(metadata.get("episodes_started", episode)),
        warm_start_provenance=metadata.get("warm_start_provenance"),
        convergence_controller=controller,
    )
    print(
        f"Reconciled interrupted best promotion at episode {episode}; latest training "
        "state was preserved."
    )
    return load_resume_metadata(resume_path, ignore_mismatch=False)


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

    def compatible_identity(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        # v0.5.0 and earlier predate adaptive looks; missing fields mean the
        # unchanged single-look defaults, not an unknown evaluation identity.
        normalized.setdefault("adaptive_eval_max_episodes", 0)
        normalized.setdefault("adaptive_eval_growth_factor", 2.0)
        normalized.setdefault("full_eval_confirmation_interval", 0)
        normalized.setdefault("inconclusive_scheduler_mode", "defer_v1")
        normalized.setdefault("bounded_inconclusive_patience", 0)
        normalized.setdefault("full_eval_seed_base", 1_000_000)
        normalized.setdefault("full_eval_max_attempts", 0)
        normalized.setdefault("action_mask_mode", "legal_v1")
        game_identity = normalized.get("game_config")
        if isinstance(game_identity, dict):
            game_identity = dict(game_identity)
            game_identity.setdefault("idle_limit_floor_steps", 0)
            normalized["game_config"] = game_identity
        return normalized

    stored_identity = compatible_identity(metadata.get("evaluation_identity"))
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
        or compatible_identity(best_metadata.get("evaluation_identity"))
        != current_identity
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
    initial_channels = {1: 3, 2: 17, 3: V3_OBSERVATION_CHANNELS}[args.network_version]
    initial_state = flatten_observation(
        train_env, device="cpu", expected_channels=initial_channels
    )
    obs_shape = tuple(int(value) for value in initial_state.shape)
    return DQNAgent(
        state_dim=int(np.prod(obs_shape)),
        action_dim=len(RelativeAction) if args.network_version >= 3 else len(Action),
        hidden_sizes=tuple(args.hidden),
        use_dueling=not args.disable_dueling,
        device=args.device,
        game_config=game_config,
        obs_shape=obs_shape,
        network_version=args.network_version,
        **_agent_training_options(args),
    )


def _agent_training_options(args: argparse.Namespace) -> dict[str, Any]:
    """Return fresh-run controls that policy transfer may safely override."""
    return {
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "min_replay_size": args.min_replay,
        "target_update_interval": args.target_update,
        "target_update_tau": args.target_update_tau,
        "hard_update_interval": args.hard_update_interval,
        "use_double_dqn": not args.disable_double_dqn,
        "epsilon_start": args.epsilon_start,
        "epsilon_final": args.epsilon_final,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "n_step": args.n_step,
        "per_alpha": args.per_alpha,
        "per_beta_start": args.per_beta_start,
        "per_beta_frames": args.per_beta_frames,
        "amp_enabled": False if args.disable_amp else None,
        "policy_anchor_weight": args.policy_anchor_weight,
        "policy_anchor_final_weight": args.policy_anchor_final_weight,
        "policy_anchor_decay_steps": args.policy_anchor_decay_steps,
        "teacher_replay_steps": args.teacher_replay_steps,
        "demonstration_capacity": args.demonstration_capacity,
        "demonstration_batch_fraction": args.demonstration_batch_fraction,
        "elite_demonstration_batch_fraction": args.elite_demonstration_batch_fraction,
        "demonstration_min_score": args.demonstration_min_score,
        "demonstration_min_return": args.demonstration_min_return,
        "demonstration_elite_score": args.demonstration_elite_score,
        "demonstration_elite_return": args.demonstration_elite_return,
        "imitation_loss_weight": args.imitation_loss_weight,
        "imitation_margin": args.imitation_margin,
        "demonstration_terminal_exclusion_steps": (
            args.demonstration_terminal_exclusion_steps
        ),
        "action_mask_mode": args.action_mask_mode,
    }


def _adopt_or_validate_warm_start_structure(
    args: argparse.Namespace, agent: DQNAgent
) -> None:
    """Make implicit structure truthful and reject explicit source conflicts."""
    provided = set(getattr(args, "_provided_options", ()))
    checks: dict[str, tuple[Any, Any]] = {
        "--network-version": (args.network_version, agent.network_version),
        "--hidden": (tuple(args.hidden), tuple(agent.hidden_sizes)),
        "--disable-dueling": (not args.disable_dueling, agent.use_dueling),
    }
    conflicts = {
        option: values
        for option, values in checks.items()
        if option in provided and values[0] != values[1]
    }
    if conflicts:
        raise RuntimeError(
            "Explicit warm-start network options conflict with the authenticated source: "
            + ", ".join(
                f"{option}=requested:{values[0]!r}/source:{values[1]!r}"
                for option, values in conflicts.items()
            )
        )
    args.network_version = agent.network_version
    args.hidden = list(agent.hidden_sizes)
    args.disable_dueling = not agent.use_dueling


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
        idle_limit_floor_steps=args.idle_limit_floor_steps,
        max_episode_steps=configured_step_limit,
    )
    train_env = SnakeGameEnv(game_config)
    train_env.reset(seed=deterministic_episode_seed(args.seed, 0))
    step_limit = episode_step_limit(game_config, args.max_steps)

    resume_path = _resume_path(args)
    output_path = Path(args.output)
    latest_path = Path(args.latest_output)
    if resume_path is None and (
        not args.warm_start_from
        or "--network-version" in set(getattr(args, "_provided_options", ()))
    ):
        _validate_output_network_identity(
            args.network_version, output_path, latest_path
        )
    warm_start_policy_options_enabled = any(
        (
            args.policy_anchor_weight > 0,
            args.teacher_replay_steps > 0,
            args.demonstration_batch_fraction > 0,
            args.imitation_loss_weight > 0,
        )
    )
    gated_episode_zero_baseline = (
        args.require_paired_promotion and args.full_eval_confirmation_interval > 0
    )
    reference_options_require_warm_start = (
        args.require_paired_promotion or args.regression_stop_patience > 0
    ) and not gated_episode_zero_baseline
    if (
        resume_path is None
        and not args.warm_start_from
        and (
            warm_start_policy_options_enabled
            or reference_options_require_warm_start
        )
    ):
        raise RuntimeError(
            "Policy anchoring, teacher/demonstration replay, imitation learning, and "
            "non-gated paired evaluation guards require --warm-start-from (or a latest "
            "checkpoint that already contains them)."
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
        agent = DQNAgent.restore_training_checkpoint(
            str(resume_path), device=args.device
        )
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
            agent.policy_anchor_weight > 0 or agent.teacher_replay_steps > 0
        ) and not agent.policy_anchor_enabled:
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
            metadata = reconcile_pending_best_promotion(
                metadata,
                agent,
                controller,
                args,
                resume_path=resume_path,
                output_path=output_path,
            )
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
            agent = DQNAgent.from_policy_checkpoint(
                str(source),
                target_game_config=game_config,
                device=args.device,
                expected_sha256=source_checkpoint_sha256,
                agent_options=_agent_training_options(args),
            )
            DQNAgent.validate_policy_sidecar_identity(
                source_metadata, agent.policy_transfer_provenance or {}
            )
            _adopt_or_validate_warm_start_structure(args, agent)
            validate_v3_contract(agent)
            _validate_output_network_identity(
                agent.network_version, output_path, latest_path
            )
            if agent.policy_anchor_weight > 0 or agent.teacher_replay_steps > 0:
                agent.snapshot_policy_anchor()
            warm_start_provenance = dict(agent.policy_transfer_provenance or {})
            if (
                warm_start_provenance.get("cross_map")
                and source_metadata.get("checkpoint_role") != "best_eval"
            ):
                message = (
                    "Cross-map warm start requires an authenticated best_eval source "
                    "sidecar by default."
                )
                if not args.ignore_warm_start_metadata:
                    raise RuntimeError(
                        message
                        + " Use --ignore-warm-start-metadata only for an intentional "
                        "legacy/non-best transfer."
                    )
                print(
                    "Warning:",
                    message,
                    "Proceeding because --ignore-warm-start-metadata was supplied.",
                )
            warm_start_provenance.update(
                {
                    "sidecar_role": source_metadata.get("checkpoint_role"),
                    "sidecar_episode": source_metadata.get("episodes_completed"),
                    "source_sidecar_role": source_metadata.get("checkpoint_role"),
                    "source_sidecar_episode": source_metadata.get("episodes_completed"),
                    "source_sidecar_verified": source_sidecar_verified,
                    "source_sidecar_path": str(sidecar_path(source).resolve()),
                    "source_sidecar_checkpoint_sha256": source_metadata.get(
                        "checkpoint_sha256"
                    ),
                }
            )
            agent.policy_transfer_provenance = warm_start_provenance
            # Preserve existing destinations until source authentication and policy
            # compatibility have both succeeded.
            _prepare_fresh_outputs(args, None)
            print(
                f"Warm-started policy weights from {source.resolve()}; target is "
                "synchronized from that policy, while optimizer, scaler, replay, "
                "epsilon, counters, seeds, best identity, and outputs are fresh."
            )
        else:
            agent = _new_agent(args, game_config, train_env)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_log_{int(time.time())}.jsonl"
    eval_max_episodes = (
        args.adaptive_eval_max_episodes
        if args.require_paired_promotion and args.adaptive_eval_max_episodes > 0
        else args.eval_episodes
    )
    eval_plan = adaptive_evaluation_plan(
        args.eval_episodes,
        eval_max_episodes,
        args.adaptive_eval_growth_factor,
    )
    eval_seeds = [args.eval_seed_base + index for index in range(eval_max_episodes)]
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
                    "action_mask_mode": agent.action_mask_mode,
                    "obs_shape": list(agent.obs_shape),
                    "step_limit": step_limit,
                    "idle_limit_floor_steps": game_config.idle_limit_floor_steps,
                    "effective_policy_anchor_weight": (
                        agent.effective_policy_anchor_weight
                    ),
                    "observation_pinned": current_encoder.is_pinned,
                    "runtime": runtime_info,
                    "eval_seeds": eval_seeds,
                    "collection_log_interval": args.collection_log_interval,
                    "eval_episodes_planned": args.eval_episodes,
                    "eval_episodes_max": eval_max_episodes,
                    "eval_planned_looks": len(eval_plan),
                    "eval_statistical_method": (
                        "paired_normal_bonferroni_v1"
                        if args.require_paired_promotion
                        else None
                    ),
                    "eval_family_confidence": (
                        0.95 if args.require_paired_promotion else None
                    ),
                    "eval_look_confidence": (
                        1.0 - 0.05 / len(eval_plan)
                        if args.require_paired_promotion
                        else None
                    ),
                    "current_learning_rates": controller.learning_rates(
                        agent.optimizer
                    ),
                    "convergence_controller": controller.to_summary_dict(),
                    "args": vars(args),
                }
            )
            + "\n"
        )

    establish_episode_zero_baseline = bool(
        args.warm_start_from
        or (resume_path is None and gated_episode_zero_baseline)
    )
    if establish_episode_zero_baseline:
        # The immutable new-run reference is evaluated once at the complete maximum
        # suite size before any training transition; probes compare against its prefix.
        baseline_started = time.perf_counter()
        baseline = evaluate_agent(agent, game_config, eval_seeds, step_limit)
        baseline_seconds = time.perf_counter() - baseline_started
        baseline_score = float(baseline["score"]["mean"])
        baseline_score_samples = evaluation_samples(baseline, "score")
        controller.record_evaluation_cost(len(baseline_score_samples), baseline_seconds)
        controller_decision = controller.observe(
            baseline_score,
            agent.optimizer,
            sample_scores=baseline_score_samples,
            planned_looks=len(eval_plan),
            is_max_sample=True,
        )
        controller.set_paired_reference(
            baseline_score_samples[: args.eval_episodes]
            if args.full_eval_confirmation_interval > 0
            else baseline_score_samples
        )
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
            # Keep the established baseline kind for analyzer compatibility; the
            # source field distinguishes transferred and freshly initialized policies.
            "evaluation_kind": "warm_start_baseline",
            "baseline_source": "warm_start" if args.warm_start_from else "fresh",
            "episode": 0,
            "best_eval_score": best_eval_score,
            "best_eval_episode": best_eval_episode,
            "current_learning_rates": controller.learning_rates(agent.optimizer),
            "effective_policy_anchor_weight": agent.effective_policy_anchor_weight,
            "convergence_decision": controller_decision,
            "convergence_controller": controller.to_dict(),
            "eval_episodes_actual": len(baseline_score_samples),
            "eval_episodes_planned": eval_max_episodes,
            "eval_episodes_max": eval_max_episodes,
            "eval_expansion_stage": 1,
            "eval_planned_looks": len(eval_plan),
            "eval_statistical_method": (
                "paired_normal_bonferroni_v1" if args.require_paired_promotion else None
            ),
            "eval_family_confidence": (0.95 if args.require_paired_promotion else None),
            "eval_look_confidence": (
                1.0 - 0.05 / len(eval_plan) if args.require_paired_promotion else None
            ),
            "eval_statistical_state": controller_decision["statistical_state"],
            "eval_patience_deferred": controller_decision["patience_deferred"],
            "eval_seconds": baseline_seconds,
            "eval_total_episodes": controller.evaluation_episodes,
            "eval_total_seconds": controller.evaluation_seconds,
        }
        for group in ("reward", "score", "steps"):
            for name, value in baseline[group].items():
                baseline_record[f"eval_{group}_{name}"] = value
        baseline_record["eval_terminal_events"] = baseline["terminal_events"]
        baseline_record["eval_truncated_count"] = baseline["truncated_count"]
        baseline_record["eval_seeds"] = baseline.get("seeds", eval_seeds)
        baseline_record["eval_reward_samples"] = evaluation_samples(baseline, "reward")
        baseline_record["eval_score_samples"] = baseline_score_samples
        baseline_record["eval_step_samples"] = evaluation_samples(baseline, "step")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")
        print(
            f"Episode-0 immutable baseline fixed-suite score {baseline_score:.3f}; "
            f"saved episode-0 best {output_path} and latest {latest_path}."
        )

    # Training environments are not launched until any required episode-0 baseline
    # and its two new-run checkpoints have been durably recorded.
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
                    next_action_mask=(
                        [True] * agent.action_dim
                        if replay_done
                        else action_mask(agent, slot.env)
                    ),
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
                demonstration_result = agent.finalize_trajectory(
                    score=slot.env.score,
                    episode_return=slot.total_env_reward,
                    stream_id=slot.stream_id,
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
                        "demonstration_quality_tier": int(
                            demonstration_result["quality_tier"]
                        ),
                        "demonstration_transitions_promoted": int(
                            demonstration_result["promoted_transitions"]
                        ),
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
        teacher_replay_complete = len(agent.replay_buffer) >= agent.teacher_replay_steps
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
            "convergence_controller": controller.to_summary_dict(),
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
            "effective_policy_anchor_weight": agent.effective_policy_anchor_weight,
            "avg_imitation_loss": metric_mean("imitation_loss"),
            "imitation_loss_weight": agent.imitation_loss_weight,
            "avg_demonstration_batch_fraction": metric_mean(
                "demonstration_batch_fraction"
            ),
            "avg_elite_demonstration_batch_fraction": metric_mean(
                "elite_demonstration_batch_fraction"
            ),
            "demonstration_replay_size": (
                len(agent.demonstration_replay)
                if agent.demonstration_replay is not None
                else 0
            ),
            "demonstration_replay_success_count": (
                agent.demonstration_replay.demonstration_count
                - agent.demonstration_replay.elite_demonstration_count
                if agent.demonstration_replay is not None
                else 0
            ),
            "demonstration_replay_elite_count": (
                agent.demonstration_replay.elite_demonstration_count
                if agent.demonstration_replay is not None
                else 0
            ),
            "demonstration_trajectories_seen": agent.demonstration_trajectories_seen,
            "demonstration_transitions_promoted_total": (
                agent.demonstration_transitions_promoted
            ),
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
        evaluation_record: dict[str, Any] | None = None
        if should_evaluate:
            evaluation_started = time.perf_counter()
            reference_evaluation: dict[str, Any] | None = None
            probe_evaluation: dict[str, Any] | None = None
            probe_decision: dict[str, Any] | None = None
            full_reason: str | None = None
            full_attempt_index: int | None = None
            alpha_budget_exhausted = False
            if (
                args.require_paired_promotion
                and args.full_eval_confirmation_interval > 0
                and teacher_replay_complete
            ):
                probe_evaluation, probe_metadata = evaluate_scheduler_probe(
                    agent,
                    game_config,
                    eval_seeds[: args.eval_episodes],
                    step_limit,
                    controller,
                )
                probe_seconds = time.perf_counter() - evaluation_started
                controller.record_evaluation_cost(args.eval_episodes, probe_seconds)
                probe_scores = evaluation_samples(probe_evaluation, "score")
                probe_decision = controller.observe(
                    float(probe_evaluation["score"]["mean"]),
                    agent.optimizer,
                    sample_scores=probe_scores,
                    planned_looks=1,
                    evaluation_scope="scheduler_probe",
                )
                periodic_confirmation = (
                    controller.scheduler_probes
                    % args.full_eval_confirmation_interval
                    == 0
                )
                if probe_metadata["promotion_candidate"]:
                    full_reason = "promotion_candidate"
                elif periodic_confirmation:
                    full_reason = "periodic_confirmation"

                reserved = (
                    reserve_full_evaluation_attempt(
                        controller, episodes=eval_max_episodes
                    )
                    if full_reason is not None
                    else None
                )
                if full_reason is not None and reserved is None:
                    alpha_budget_exhausted = True
                    full_reason = None
                if reserved is not None:
                    full_attempt_index, full_seeds = reserved
                    # Persist alpha/seed reservation before looking at either policy.
                    save_checkpoint(
                        agent,
                        latest_path,
                        episode=episodes_completed,
                        run_seed=args.seed,
                        best_eval_score=best_eval_score,
                        best_eval_episode=best_eval_episode,
                        train_args=args,
                        checkpoint_role="latest",
                        best_checkpoint_path=output_path,
                        episodes_started=episodes_started,
                        warm_start_provenance=warm_start_provenance,
                        convergence_controller=controller,
                    )
                    full_started = time.perf_counter()
                    reference_agent = load_policy_evaluation_reference(
                        output_path, agent, game_config
                    )
                    evaluation, reference_evaluation = evaluate_fresh_full_pair(
                        agent,
                        reference_agent,
                        game_config,
                        full_seeds,
                        step_limit,
                    )
                    full_seconds = time.perf_counter() - full_started
                    controller.record_evaluation_cost(
                        2 * eval_max_episodes, full_seconds
                    )
                    score_samples = evaluation_samples(evaluation, "score")
                    reference_scores = evaluation_samples(reference_evaluation, "score")
                    controller_decision = controller.observe(
                        float(evaluation["score"]["mean"]),
                        agent.optimizer,
                        sample_scores=score_samples,
                        comparison_reference_scores=reference_scores,
                        planned_looks=args.full_eval_max_attempts,
                        evaluation_scope="full_confirmation",
                        is_max_sample=True,
                    )
                    paired = controller_decision["paired_comparison"]
                    adaptive_metadata = {
                        "evaluation_scope": "full_confirmation",
                        "actual_episodes": eval_max_episodes,
                        "planned_episodes": eval_max_episodes,
                        "max_episodes": eval_max_episodes,
                        "execution_episodes": (
                            args.eval_episodes + 2 * eval_max_episodes
                        ),
                        "expansion_stage": 1,
                        "planned_looks": args.full_eval_max_attempts,
                        "stages": [],
                        "statistical_method": paired["method"],
                        "family_confidence": paired["family_confidence"],
                        "look_confidence": paired["look_confidence"],
                        "statistical_state": paired["statistical_state"],
                        "full_reason": full_reason,
                        "full_attempt_index": full_attempt_index,
                    }
                else:
                    evaluation = probe_evaluation
                    adaptive_metadata = {
                        **probe_metadata,
                        "execution_episodes": args.eval_episodes,
                        "full_reason": None,
                        "full_attempt_index": None,
                    }
                    controller_decision = probe_decision
            elif args.require_paired_promotion:
                evaluation, adaptive_metadata = evaluate_adaptive_paired(
                    agent,
                    game_config,
                    eval_seeds,
                    step_limit,
                    controller,
                    base_episodes=args.eval_episodes,
                    max_episodes=eval_max_episodes,
                    growth_factor=args.adaptive_eval_growth_factor,
                    adaptive_enabled=(
                        teacher_replay_complete
                        and eval_max_episodes > args.eval_episodes
                    ),
                )
                evaluation_seconds = time.perf_counter() - evaluation_started
                controller.record_evaluation_cost(
                    int(adaptive_metadata["actual_episodes"]), evaluation_seconds
                )
            else:
                evaluation = evaluate_agent(
                    agent, game_config, eval_seeds[: args.eval_episodes], step_limit
                )
                adaptive_metadata = {
                    "actual_episodes": args.eval_episodes,
                    "planned_episodes": args.eval_episodes,
                    "max_episodes": args.eval_episodes,
                    "expansion_stage": 1,
                    "planned_looks": 1,
                    "stages": [],
                    "statistical_method": None,
                    "family_confidence": None,
                    "look_confidence": None,
                    "statistical_state": None,
                    "evaluation_scope": "full_evaluation",
                    "execution_episodes": args.eval_episodes,
                    "full_reason": None,
                    "full_attempt_index": None,
                }
                evaluation_seconds = time.perf_counter() - evaluation_started
                controller.record_evaluation_cost(args.eval_episodes, evaluation_seconds)
            metrics = completed_summaries[-1]
            for group in ("reward", "score", "steps"):
                for name, value in evaluation[group].items():
                    metrics[f"eval_{group}_{name}"] = value
            metrics["eval_terminal_events"] = evaluation["terminal_events"]
            metrics["eval_truncated_count"] = evaluation["truncated_count"]
            score_samples = evaluation_samples(evaluation, "score")
            metrics["eval_seeds"] = evaluation.get("seeds", eval_seeds)
            metrics["eval_reward_samples"] = evaluation_samples(evaluation, "reward")
            metrics["eval_score_samples"] = score_samples
            metrics["eval_step_samples"] = evaluation_samples(evaluation, "step")
            metrics["eval_episodes_actual"] = adaptive_metadata["actual_episodes"]
            metrics["eval_episodes_planned"] = adaptive_metadata["planned_episodes"]
            metrics["eval_episodes_max"] = adaptive_metadata["max_episodes"]
            metrics["eval_execution_episodes"] = adaptive_metadata.get(
                "execution_episodes", adaptive_metadata["actual_episodes"]
            )
            metrics["eval_expansion_stage"] = adaptive_metadata["expansion_stage"]
            metrics["eval_planned_looks"] = adaptive_metadata["planned_looks"]
            metrics["eval_adaptive_stages"] = adaptive_metadata["stages"]
            metrics["eval_statistical_method"] = adaptive_metadata["statistical_method"]
            metrics["eval_family_confidence"] = adaptive_metadata["family_confidence"]
            metrics["eval_look_confidence"] = adaptive_metadata["look_confidence"]
            metrics["eval_scope"] = adaptive_metadata.get(
                "evaluation_scope", "full_evaluation"
            )
            metrics["eval_full_reason"] = adaptive_metadata.get("full_reason")
            metrics["eval_full_attempt_index"] = adaptive_metadata.get(
                "full_attempt_index"
            )
            metrics["eval_alpha_budget_exhausted"] = alpha_budget_exhausted
            average_score = float(evaluation["score"]["mean"])
            previous_best = best_eval_score
            had_paired_reference = controller.reference_scores is not None
            if controller_decision is None:
                controller_decision = controller.observe(
                    average_score,
                    agent.optimizer,
                    sample_scores=score_samples,
                    defer_reason=(
                        None if teacher_replay_complete else "teacher_replay_warmup"
                    ),
                    planned_looks=int(adaptive_metadata["planned_looks"]),
                    is_max_sample=(len(score_samples) == eval_max_episodes),
                )
            evaluation_seconds = time.perf_counter() - evaluation_started
            metrics["eval_statistical_state"] = controller_decision["statistical_state"]
            metrics["eval_patience_deferred"] = controller_decision["patience_deferred"]
            metrics["eval_seconds"] = evaluation_seconds
            metrics["eval_total_episodes"] = controller.evaluation_episodes
            metrics["eval_total_seconds"] = controller.evaluation_seconds
            metrics["eval_probe_decision"] = probe_decision
            if reference_evaluation is not None:
                metrics["eval_reference_score_mean"] = reference_evaluation["score"][
                    "mean"
                ]
                metrics["eval_reference_score_samples"] = evaluation_samples(
                    reference_evaluation, "score"
                )
            improved = teacher_replay_complete and (
                (
                    not had_paired_reference
                    or controller_decision["paired_promotion_eligible"]
                )
                if controller.require_paired_promotion
                else average_score > previous_best
            )
            metrics["current_learning_rates"] = controller.learning_rates(
                agent.optimizer
            )
            metrics["convergence_decision"] = controller_decision
            metrics["convergence_controller"] = controller.to_summary_dict()
            collection_metrics["current_learning_rates"] = controller.learning_rates(
                agent.optimizer
            )
            collection_metrics["convergence_decision"] = controller_decision
            collection_metrics["convergence_controller"] = (
                controller.to_summary_dict()
            )
            for key in (
                "eval_episodes_actual",
                "eval_episodes_planned",
                "eval_episodes_max",
                "eval_expansion_stage",
                "eval_planned_looks",
                "eval_adaptive_stages",
                "eval_statistical_method",
                "eval_family_confidence",
                "eval_look_confidence",
                "eval_statistical_state",
                "eval_patience_deferred",
                "eval_seconds",
                "eval_total_episodes",
                "eval_total_seconds",
                "eval_scope",
                "eval_execution_episodes",
                "eval_full_reason",
                "eval_full_attempt_index",
                "eval_alpha_budget_exhausted",
            ):
                collection_metrics[key] = metrics[key]
            if improved:
                best_eval_score = average_score
                best_eval_episode = episodes_completed
                if (
                    controller.require_paired_promotion
                    and len(score_samples) != eval_max_episodes
                ):
                    raise RuntimeError(
                        "Paired promotion requires a complete maximum-sized reference"
                    )
                controller.set_paired_reference(
                    evaluation_samples(probe_evaluation, "score")
                    if args.full_eval_confirmation_interval > 0
                    and probe_evaluation is not None
                    else score_samples
                )
                metrics["convergence_controller"] = controller.to_summary_dict()
                collection_metrics["convergence_controller"] = (
                    controller.to_summary_dict()
                )
                pending_best_promotion = _best_promotion_marker(
                    output_path,
                    score=best_eval_score,
                    episode=best_eval_episode,
                )
                # The authenticated latest checkpoint is the write-ahead record. A
                # resume can complete the approved promotion from this exact training
                # state if the process stops before the canonical best/latest pair is
                # fully linked.
                save_checkpoint(
                    agent,
                    latest_path,
                    episode=episodes_completed,
                    run_seed=args.seed,
                    best_eval_score=best_eval_score,
                    best_eval_episode=best_eval_episode,
                    train_args=args,
                    checkpoint_role="latest",
                    best_checkpoint_path=output_path,
                    episodes_started=episodes_started,
                    warm_start_provenance=warm_start_provenance,
                    convergence_controller=controller,
                    pending_best_promotion=pending_best_promotion,
                )
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
                save_checkpoint(
                    agent,
                    latest_path,
                    episode=episodes_completed,
                    run_seed=args.seed,
                    best_eval_score=best_eval_score,
                    best_eval_episode=best_eval_episode,
                    train_args=args,
                    checkpoint_role="latest",
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
            evaluation_record = {
                "record_type": "evaluation",
                "evaluation_kind": metrics["eval_scope"],
                "episode": episodes_completed,
                "best_eval_score": best_eval_score,
                "best_eval_episode": best_eval_episode,
                "convergence_decision": controller_decision,
                "convergence_controller": controller.to_dict(),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key.startswith("eval_")
                },
            }
            for key in [key for key in metrics if key.startswith("eval_")]:
                del metrics[key]
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

        controller_stop = bool(
            controller_decision is not None and controller_decision["should_stop"]
        )
        final_collection = (
            episodes_completed >= initial_completed + args.episodes or controller_stop
        )
        log_collection = (
            should_evaluate
            or final_collection
            or collection_index % args.collection_log_interval == 0
        )
        with log_path.open("a", encoding="utf-8") as stream:
            if log_collection:
                stream.write(json.dumps(collection_metrics, ensure_ascii=False) + "\n")
            if evaluation_record is not None:
                stream.write(json.dumps(evaluation_record, ensure_ascii=False) + "\n")
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
        if controller_stop:
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
