from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import train_dqn as training_module

from dqn_agent import DQNAgent, flatten_observation
from env import Action, GameConfig, RelativeAction, SnakeGameEnv
from play_dqn import absolute_action, run_episode
from train_dqn import (
    EvaluationConvergenceController,
    _prepare_fresh_outputs,
    _release_accelerator_resources,
    accelerator_runtime_info,
    action_mask,
    adaptive_evaluation_plan,
    deterministic_episode_seed,
    episode_step_limit,
    evaluate_agent,
    evaluate_adaptive_paired,
    evaluate_fresh_full_pair,
    evaluate_scheduler_probe,
    load_resume_metadata,
    load_warm_start_metadata,
    parse_args,
    potential_shaping,
    reserve_full_evaluation_attempt,
    restore_convergence_controller,
    save_checkpoint,
    state_potential,
    train,
    validate_resume_agent_options,
    validate_resume_seed,
    validate_resume_identity,
    validate_v3_contract,
)


@pytest.mark.parametrize(
    "options",
    [
        ["--lr-plateau-patience", "-1"],
        ["--lr-plateau-factor", "0"],
        ["--lr-plateau-factor", "1"],
        ["--lr-plateau-min", "0"],
        ["--lr-plateau-min", "-0.1"],
        ["--lr", "0.001", "--lr-plateau-patience", "1", "--lr-plateau-min", "0.002"],
        ["--early-stop-patience", "-1"],
        ["--early-stop-delta", "-0.1"],
        ["--early-stop-delta", "nan"],
        ["--policy-anchor-weight", "-0.1"],
        ["--policy-anchor-weight", "nan"],
        ["--teacher-replay-steps", "-1"],
        ["--teacher-replay-steps", "101", "--replay-capacity", "100"],
        ["--collection-log-interval", "0"],
        ["--max-steps", "-1"],
        ["--eval-vector-envs", "0"],
        ["--max-steps", "0", "--time-feature-mode", "finite_horizon_progress_v1"],
        ["--max-steps", "10", "--time-feature-mode", "constant_zero_v1"],
        ["--idle-limit-floor-steps", "-1"],
        ["--max-idle-steps", "0", "--idle-limit-floor-steps", "1"],
        ["--policy-anchor-weight", "0.1", "--policy-anchor-final-weight", "0.2"],
        ["--policy-anchor-decay-steps", "-1"],
        ["--demonstration-terminal-exclusion-steps", "-1"],
        ["--network-version", "2", "--action-mask-mode", "one_step_survival_v1"],
        [
            "--batch-size",
            "2",
            "--demonstration-capacity",
            "8",
            "--demonstration-batch-fraction",
            "0.1",
        ],
        [
            "--batch-size",
            "2",
            "--demonstration-capacity",
            "8",
            "--demonstration-batch-fraction",
            "0.75",
        ],
        [
            "--batch-size",
            "2",
            "--demonstration-capacity",
            "8",
            "--demonstration-batch-fraction",
            "0.5",
            "--elite-demonstration-batch-fraction",
            "0.1",
        ],
        ["--paired-promotion-min-delta", "-0.1"],
        ["--regression-stop-patience", "-1"],
        ["--regression-stop-delta", "-0.1"],
        ["--adaptive-eval-max-episodes", "-1"],
        [
            "--eval-episodes",
            "4",
            "--adaptive-eval-max-episodes",
            "2",
            "--require-paired-promotion",
        ],
        ["--adaptive-eval-max-episodes", "8"],
        ["--adaptive-eval-growth-factor", "1"],
        ["--adaptive-eval-growth-factor", "nan"],
        ["--bounded-inconclusive-patience", "-1"],
        ["--bounded-inconclusive-patience", "3"],
        ["--inconclusive-scheduler-mode", "bounded_probe_v1"],
        [
            "--require-paired-promotion",
            "--adaptive-eval-max-episodes",
            "8",
            "--eval-episodes",
            "2",
            "--full-eval-confirmation-interval",
            "2",
        ],
        ["--full-eval-max-attempts", "-1"],
        [
            "--require-paired-promotion",
            "--adaptive-eval-max-episodes",
            "8",
            "--eval-episodes",
            "2",
            "--full-eval-confirmation-interval",
            "2",
            "--full-eval-max-attempts",
            "2",
            "--eval-seed-base",
            "100",
            "--full-eval-seed-base",
            "101",
        ],
        [
            "--require-paired-promotion",
            "--adaptive-eval-max-episodes",
            "8",
            "--eval-episodes",
            "2",
            "--full-eval-confirmation-interval",
            "2",
            "--full-eval-max-attempts",
            "2",
            "--eval-seed-base",
            "100",
            "--full-eval-seed-base",
            "104",
        ],
    ],
)
def test_convergence_cli_rejects_invalid_ranges(options: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(options)


def test_convergence_controller_reduces_to_min_then_gates_early_stop() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=2,
        lr_plateau_factor=0.5,
        lr_plateau_min=0.025,
        early_stop_patience=2,
        early_stop_delta=1.0,
    )

    assert controller.observe(10.0, optimizer)["significant_improvement"]
    assert not controller.observe(10.5, optimizer)["lr_reduced"]
    assert controller.observe(10.8, optimizer)["learning_rates"] == [0.05]
    assert not controller.observe(10.9, optimizer)["lr_reduced"]
    reached_min = controller.observe(10.7, optimizer)
    assert reached_min["learning_rates"] == [0.025]
    assert reached_min["at_min_lr"]
    assert reached_min["min_lr_evaluations"] == 0
    assert not controller.observe(10.6, optimizer)["should_stop"]
    assert controller.observe(10.4, optimizer)["should_stop"]


def test_convergence_controller_significant_improvement_resets_patience() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    controller = EvaluationConvergenceController(2, 0.5, 0.01, 3, 1.0)

    controller.observe(5.0, optimizer)
    controller.observe(5.5, optimizer)
    decision = controller.observe(6.0, optimizer)

    assert decision["significant_improvement"]
    assert decision["reference_score"] == 6.0
    assert decision["plateau_evaluations"] == 0
    assert decision["min_lr_evaluations"] == 0


def test_convergence_controller_without_scheduler_preserves_legacy_early_stop() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    controller = EvaluationConvergenceController(0, 0.5, 0.01, 2, 1.0)

    controller.observe(5.0, optimizer)
    assert not controller.observe(5.5, optimizer)["should_stop"]
    decision = controller.observe(5.4, optimizer)

    assert decision["should_stop"]
    assert decision["decision"] == "early_stop_patience"
    assert decision["learning_rates"] == [0.1]


def test_paired_promotion_and_clear_regression_guard() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=4,
        lr_plateau_factor=0.5,
        lr_plateau_min=0.001,
        early_stop_patience=10,
        early_stop_delta=0.1,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        regression_stop_patience=2,
        regression_stop_delta=0.1,
        reference_score=3.0,
    )
    controller.set_paired_reference([1.0, 2.0, 3.0, 4.0])

    noisy = controller.observe(
        3.2,
        optimizer,
        sample_scores=[0.0, 4.0, 2.0, 5.0],
    )
    assert noisy["aggregate_significant_improvement"]
    assert not noisy["paired_promotion_eligible"]
    assert not noisy["significant_improvement"]

    promoted = controller.observe(
        4.0,
        optimizer,
        sample_scores=[2.0, 3.0, 4.0, 5.0],
    )
    assert promoted["paired_promotion_eligible"]
    assert promoted["paired_comparison"]["ci95_low"] == pytest.approx(1.0)
    assert promoted["significant_improvement"]
    controller.set_paired_reference([2.0, 3.0, 4.0, 5.0])

    first_regression = controller.observe(
        2.0,
        optimizer,
        sample_scores=[1.0, 2.0, 3.0, 4.0],
    )
    assert first_regression["clear_regression"]
    assert not first_regression["should_stop"]
    assert first_regression["decision"] == "paired_clear_regression"
    assert first_regression["plateau_evaluations"] == 0
    assert first_regression["min_lr_evaluations"] == 0
    assert not first_regression["lr_reduced"]
    stopped = controller.observe(
        2.0,
        optimizer,
        sample_scores=[1.0, 2.0, 3.0, 4.0],
    )
    assert stopped["should_stop"]
    assert stopped["decision"] == "paired_regression_patience"


def test_paired_inconclusive_ci_does_not_consume_any_patience() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=2,
        lr_plateau_factor=0.5,
        lr_plateau_min=0.001,
        early_stop_patience=2,
        early_stop_delta=0.25,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        reference_score=0.0,
    )
    controller.set_paired_reference([0.0] * 4)
    controller.plateau_evaluations = 1
    controller.min_lr_evaluations = 1

    decision = controller.observe(
        0.3,
        optimizer,
        sample_scores=[-0.1, 0.1, 0.5, 0.7],
        planned_looks=3,
    )

    assert decision["aggregate_significant_improvement"]
    assert decision["statistical_state"] == "inconclusive"
    assert decision["patience_deferred"] is True
    assert decision["plateau_evaluations"] == 1
    assert decision["min_lr_evaluations"] == 1
    assert decision["learning_rates"] == [0.01]
    paired = decision["paired_comparison"]
    assert paired["ci95_low"] < 0 < paired["ci95_high"]
    assert paired["adjusted_ci_low"] < paired["ci95_low"]
    assert paired["adjusted_ci_high"] > paired["ci95_high"]
    assert paired["method"] == "paired_normal_bonferroni_v1"
    assert paired["planned_looks"] == 3


def test_only_confirmed_paired_plateau_reduces_lr() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=1,
        lr_plateau_factor=0.5,
        lr_plateau_min=0.001,
        early_stop_patience=2,
        early_stop_delta=0.25,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        reference_score=1.0,
    )
    controller.set_paired_reference([1.0] * 4)

    decision = controller.observe(
        1.1,
        optimizer,
        sample_scores=[1.1] * 4,
        planned_looks=3,
    )

    assert decision["statistical_state"] == "confirmed_plateau"
    assert decision["lr_reduced"] is True
    assert decision["learning_rates"] == [0.005]


def test_bounded_probe_inconclusive_spends_one_pre_min_plateau_tick() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=2,
        lr_plateau_factor=0.5,
        lr_plateau_min=0.001,
        early_stop_patience=2,
        early_stop_delta=0.25,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        inconclusive_scheduler_mode="bounded_probe_v1",
        bounded_inconclusive_patience=3,
        reference_score=0.0,
    )
    controller.set_paired_reference([0.0] * 4)
    samples = [-0.1, 0.1, 0.5, 0.7]

    first = controller.observe(
        0.3,
        optimizer,
        sample_scores=samples,
        evaluation_scope="scheduler_probe",
    )
    second = controller.observe(
        0.3,
        optimizer,
        sample_scores=samples,
        evaluation_scope="scheduler_probe",
    )
    third = controller.observe(
        0.3,
        optimizer,
        sample_scores=samples,
        evaluation_scope="scheduler_probe",
    )

    assert first["patience_deferred"] is True
    assert second["probe_inconclusive_evaluations"] == 2
    assert third["bounded_inconclusive_triggered"] is True
    assert third["decision"] == "bounded_probe_inconclusive_plateau_patience"
    assert third["plateau_evaluations"] == 1
    assert third["learning_rates"] == [0.01]

    for _ in range(2):
        controller.observe(
            0.3,
            optimizer,
            sample_scores=samples,
            evaluation_scope="scheduler_probe",
        )
    reduced = controller.observe(
        0.3,
        optimizer,
        sample_scores=samples,
        evaluation_scope="scheduler_probe",
    )
    assert reduced["decision"] == "bounded_probe_inconclusive_lr_reduced"
    assert reduced["learning_rates"] == [0.005]


def test_probe_ambiguity_never_spends_min_lr_early_stop_patience() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    controller = EvaluationConvergenceController(
        1,
        0.5,
        0.001,
        1,
        0.25,
        require_paired_promotion=True,
        inconclusive_scheduler_mode="bounded_probe_v1",
        bounded_inconclusive_patience=3,
        reference_score=0.0,
    )
    controller.set_paired_reference([0.0] * 4)

    for _ in range(6):
        decision = controller.observe(
            0.3,
            optimizer,
            sample_scores=[-0.1, 0.1, 0.5, 0.7],
            evaluation_scope="scheduler_probe",
        )

    assert decision["decision"] == "bounded_probe_inconclusive_at_min_lr_deferred"
    assert decision["should_stop"] is False
    assert decision["min_lr_evaluations"] == 0
    assert decision["learning_rates"] == [0.001]


def _evaluation_payload(seeds: list[int], score: float) -> dict[str, object]:
    count = len(seeds)
    samples = [score] * count
    distribution = {
        "mean": score,
        "std": 0.0,
        "ci95_low": score,
        "ci95_high": score,
        "median": score,
        "p10": score,
        "p90": score,
        "min": score,
        "max": score,
    }
    return {
        "reward": dict(distribution),
        "score": dict(distribution),
        "steps": dict(distribution),
        "seeds": seeds,
        "reward_samples": samples,
        "score_samples": samples,
        "step_samples": [1] * count,
        "terminal_events": {"test": count},
        "truncated_count": 0,
        "episodes": count,
    }


@pytest.mark.parametrize(
    ("candidate_score", "expected_state"),
    [(0.25, "inconclusive"), (1.0, "confirmed_improvement")],
)
def test_adaptive_paired_evaluation_uses_disjoint_chunks_and_full_promotion(
    monkeypatch: pytest.MonkeyPatch,
    candidate_score: float,
    expected_state: str,
) -> None:
    agent = make_agent(GameConfig(width=5, height=5))
    controller = EvaluationConvergenceController(
        2,
        0.5,
        0.001,
        2,
        0.25,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        reference_score=0.0,
    )
    controller.set_paired_reference([0.0] * 6)
    seed_chunks: list[list[int]] = []

    def fake_evaluate(
        _agent: DQNAgent,
        _config: GameConfig,
        seeds: list[int],
        _max_steps: int,
        **_options: object,
    ) -> dict[str, object]:
        seed_chunks.append(list(seeds))
        return _evaluation_payload(list(seeds), candidate_score)

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    evaluation, metadata = evaluate_adaptive_paired(
        agent,
        GameConfig(width=5, height=5),
        list(range(6)),
        2,
        controller,
        base_episodes=2,
        max_episodes=6,
        growth_factor=2.0,
        adaptive_enabled=True,
    )

    assert adaptive_evaluation_plan(2, 6, 2.0) == [2, 4, 6]
    assert seed_chunks == [[0, 1], [2, 3], [4, 5]]
    assert evaluation["seeds"] == list(range(6))
    assert metadata["actual_episodes"] == 6
    assert metadata["expansion_stage"] == 3
    assert metadata["planned_looks"] == 3
    assert metadata["statistical_state"] == expected_state


def test_probe_and_fresh_full_attempts_use_disjoint_seed_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GameConfig(width=5, height=5)
    candidate = make_agent(config)
    reference = make_agent(config)
    controller = EvaluationConvergenceController(
        2,
        0.5,
        0.001,
        2,
        0.25,
        require_paired_promotion=True,
        reference_score=0.0,
        full_eval_seed_base=1_000,
        full_eval_max_attempts=2,
    )
    controller.set_paired_reference([0.0, 0.0])
    calls: list[tuple[DQNAgent, list[int]]] = []

    def fake_evaluate(
        agent: DQNAgent,
        _config: GameConfig,
        seeds: list[int],
        _max_steps: int,
        **_options: object,
    ) -> dict[str, object]:
        calls.append((agent, list(seeds)))
        score = 1.0 if agent is candidate else 0.0
        return _evaluation_payload(list(seeds), score)

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    probe, probe_metadata = evaluate_scheduler_probe(
        candidate, config, [100, 101], 2, controller
    )
    first = reserve_full_evaluation_attempt(controller, episodes=6)
    second = reserve_full_evaluation_attempt(controller, episodes=6)

    assert probe["seeds"] == [100, 101]
    assert probe_metadata["promotion_candidate"] is True
    assert first == (0, list(range(1_000, 1_006)))
    assert second == (1, list(range(1_006, 1_012)))
    assert reserve_full_evaluation_attempt(controller, episodes=6) is None
    assert set(probe["seeds"]).isdisjoint(first[1])
    candidate_full, reference_full = evaluate_fresh_full_pair(
        candidate, reference, config, first[1], 2
    )
    assert candidate_full["seeds"] == reference_full["seeds"] == first[1]
    assert calls[-2:] == [(reference, first[1]), (candidate, first[1])]


def test_interrupted_full_pair_consumes_reservation_without_observing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GameConfig(width=5, height=5)
    candidate = make_agent(config)
    reference = make_agent(config)
    controller = EvaluationConvergenceController(
        0,
        0.5,
        0.001,
        0,
        0.1,
        require_paired_promotion=True,
        full_eval_seed_base=1_000,
        full_eval_max_attempts=2,
    )
    reserved = reserve_full_evaluation_attempt(controller, episodes=4)
    assert reserved is not None
    calls = 0

    def interrupt_candidate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _evaluation_payload(reserved[1], 0.0)

    monkeypatch.setattr(training_module, "evaluate_agent", interrupt_candidate)
    with pytest.raises(KeyboardInterrupt):
        evaluate_fresh_full_pair(candidate, reference, config, reserved[1], None)

    assert controller.full_eval_attempts == 1
    assert controller.evaluations == 0


def test_full_confirmation_uses_pre_registered_attempt_alpha_not_probe_selection() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    controller = EvaluationConvergenceController(
        2,
        0.5,
        0.001,
        2,
        0.25,
        require_paired_promotion=True,
        paired_promotion_min_delta=0.1,
        reference_score=5.0,
        full_eval_max_attempts=8,
    )
    controller.set_paired_reference([5.0, 5.0])

    probe = controller.observe(
        6.0,
        optimizer,
        sample_scores=[6.0, 6.0],
        evaluation_scope="scheduler_probe",
    )
    full = controller.observe(
        6.0,
        optimizer,
        sample_scores=[6.0] * 6,
        comparison_reference_scores=[5.0] * 6,
        planned_looks=8,
        evaluation_scope="full_confirmation",
        is_max_sample=True,
    )

    assert probe["paired_promotion_eligible"] is False
    assert full["paired_promotion_eligible"] is True
    assert full["paired_comparison"]["planned_looks"] == 8
    assert full["paired_comparison"]["look_confidence"] == pytest.approx(
        1.0 - 0.05 / 8
    )


def test_controller_summary_omits_large_reference_cache_and_v3_migrates() -> None:
    controller = EvaluationConvergenceController(2, 0.5, 0.001, 2, 0.25)
    controller.set_paired_reference([float(index) for index in range(600)])
    controller.record_evaluation_cost(600, 12.5)
    summary = controller.to_summary_dict()

    assert "reference_scores" not in summary["state"]
    assert summary["state"]["reference_scores_count"] == 600
    assert len(json.dumps(summary)) < len(json.dumps(controller.to_dict())) / 5

    legacy = controller.to_dict()
    legacy["version"] = 3
    for key in (
        "inconclusive_scheduler_mode",
        "bounded_inconclusive_patience",
        "full_eval_confirmation_interval",
        "full_eval_seed_base",
        "full_eval_max_attempts",
    ):
        legacy["config"].pop(key)
    for key in (
        "probe_inconclusive_evaluations",
        "evaluation_episodes",
        "evaluation_seconds",
        "full_eval_attempts",
        "scheduler_probes",
    ):
        legacy["state"].pop(key)

    restored = EvaluationConvergenceController.from_dict(legacy)
    assert restored.inconclusive_scheduler_mode == "defer_v1"
    assert restored.bounded_inconclusive_patience == 0
    assert restored.full_eval_attempts == 0
    assert restored.evaluation_episodes == 0


def test_v2_controller_migration_clears_potentially_polluted_patience() -> None:
    controller = EvaluationConvergenceController(
        2, 0.5, 0.001, 2, 0.25, require_paired_promotion=True
    )
    payload = controller.to_dict()
    payload["version"] = 2
    payload["state"].update(
        {
            "plateau_evaluations": 1,
            "min_lr_evaluations": 1,
            "regression_evaluations": 1,
            "reductions": 2,
        }
    )

    restored = EvaluationConvergenceController.from_dict(payload)

    assert restored.plateau_evaluations == 0
    assert restored.min_lr_evaluations == 0
    assert restored.regression_evaluations == 0
    assert restored.reductions == 2
    assert restored.migration_note == "v2_patience_counters_cleared_for_bonferroni_v3"


def test_controller_serialization_restore_and_explicit_conflict(tmp_path: Path) -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=2)
    agent = make_agent(config)
    agent.optimizer.param_groups[0]["lr"] = 5e-5
    controller = EvaluationConvergenceController(2, 0.5, 1e-5, 4, 0.75)
    controller.reference_score = 3.0
    controller.require_paired_promotion = True
    controller.paired_promotion_min_delta = 0.1
    controller.regression_stop_patience = 3
    controller.regression_stop_delta = 0.2
    controller.adaptive_eval_max_episodes = 8
    controller.adaptive_eval_growth_factor = 2.0
    controller.full_eval_seed_base = 400_000
    controller.full_eval_max_attempts = 4
    controller.full_eval_attempts = 1
    controller.record_evaluation_cost(17, 1.25)
    controller.set_paired_reference([3.0] * 8)
    controller.plateau_evaluations = 1
    checkpoint = tmp_path / "latest.pt"
    args = parse_args(
        [
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "2",
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    save_checkpoint(
        agent,
        checkpoint,
        episode=2,
        run_seed=args.seed,
        best_eval_score=3.0,
        best_eval_episode=2,
        train_args=args,
        checkpoint_role="latest",
        convergence_controller=controller,
    )
    metadata = load_resume_metadata(checkpoint, ignore_mismatch=False)
    loaded_agent = DQNAgent.load(str(checkpoint), device="cpu")
    resume_args = parse_args(
        [
            "--resume-from",
            str(checkpoint),
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    restored = restore_convergence_controller(
        metadata, resume_args, loaded_agent, legacy_reference_score=3.0
    )
    assert restored.to_dict() == controller.to_dict()
    assert resume_args.lr_plateau_patience == 2
    assert resume_args.adaptive_eval_max_episodes == 8
    assert resume_args.full_eval_seed_base == 400_000
    assert resume_args.full_eval_max_attempts == 4
    assert restored.full_eval_attempts == 1
    assert restored.evaluation_episodes == 17
    assert restored.evaluation_seconds == pytest.approx(1.25)
    assert metadata["base_learning_rate"] == pytest.approx(agent.lr)
    assert metadata["current_learning_rates"] == [5e-5]

    corrupted_metadata = json.loads(json.dumps(metadata))
    corrupted_metadata["convergence_controller"]["state"]["reference_scores"] = [
        2.0,
        3.0,
    ]
    corrupted_args = parse_args(
        [
            "--resume-from",
            str(checkpoint),
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    with pytest.raises(RuntimeError, match="reference cache length"):
        restore_convergence_controller(
            corrupted_metadata,
            corrupted_args,
            loaded_agent,
            legacy_reference_score=3.0,
        )

    wrong_mean_metadata = json.loads(json.dumps(metadata))
    wrong_mean_metadata["convergence_controller"]["state"]["reference_scores"] = [
        4.0
    ] * 8
    wrong_mean_args = parse_args(
        [
            "--resume-from",
            str(checkpoint),
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    with pytest.raises(RuntimeError, match="sample mean"):
        restore_convergence_controller(
            wrong_mean_metadata,
            wrong_mean_args,
            loaded_agent,
            legacy_reference_score=3.0,
        )

    reset_metadata = json.loads(json.dumps(metadata))
    reset_metadata["convergence_controller"]["config"]["adaptive_eval_max_episodes"] = 0
    reset_metadata["convergence_controller"]["state"]["reference_scores"] = [3.0] * 50
    reset_args = parse_args(
        [
            "--resume-from",
            str(checkpoint),
            "--reset-best-evaluation",
            "--eval-episodes",
            "60",
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    reset_controller = restore_convergence_controller(
        reset_metadata,
        reset_args,
        loaded_agent,
        legacy_reference_score=3.0,
    )
    reset_controller.reset()
    assert reset_controller.reference_score is None
    assert reset_controller.reference_scores is None

    conflict_args = parse_args(
        [
            "--resume-from",
            str(checkpoint),
            "--lr-plateau-factor",
            "0.25",
            "--full-eval-seed-base",
            "500000",
            "--output",
            str(tmp_path / "best.pt"),
            "--latest-output",
            str(checkpoint),
        ]
    )
    with pytest.raises(RuntimeError, match="scheduler/early-stop options conflict"):
        restore_convergence_controller(
            metadata, conflict_args, loaded_agent, legacy_reference_score=3.0
        )


def test_resume_lr_option_compares_base_lr_not_scheduled_optimizer_lr() -> None:
    agent = make_agent(GameConfig(width=5, height=5))
    agent.optimizer.param_groups[0]["lr"] = 5e-5

    validate_resume_agent_options(parse_args(["--lr", str(agent.lr)]), agent)
    with pytest.raises(RuntimeError, match="hyperparameters conflict"):
        validate_resume_agent_options(parse_args(["--lr", "0.0003"]), agent)


@pytest.mark.parametrize(
    "options",
    [
        ["--action-mask-mode", "legal_v1"],
        [
            "--policy-anchor-weight",
            "0.5",
            "--policy-anchor-final-weight",
            "0.1",
        ],
        ["--policy-anchor-decay-steps", "10"],
        ["--demonstration-terminal-exclusion-steps", "2"],
    ],
)
def test_resume_rejects_explicit_behavior_policy_conflicts(
    options: list[str],
) -> None:
    agent = make_agent(GameConfig(width=5, height=5))
    agent.policy_anchor_weight = 0.5
    agent.policy_anchor_final_weight = 0.2
    agent.policy_anchor_decay_steps = 100
    agent.demonstration_terminal_exclusion_steps = 3

    with pytest.raises(RuntimeError, match="hyperparameters conflict"):
        validate_resume_agent_options(parse_args(options), agent)


def make_agent(config: GameConfig, action_dim: int = 3) -> DQNAgent:
    env = SnakeGameEnv(config)
    env.reset(seed=7)
    state = flatten_observation(env, "cpu")
    return DQNAgent(
        state_dim=state.numel(),
        action_dim=action_dim,
        hidden_sizes=(16,),
        batch_size=4,
        replay_capacity=32,
        min_replay_size=4,
        obs_shape=tuple(state.shape),
        network_version=3,
        action_mask_mode=(
            "one_step_survival_v1" if action_dim == len(RelativeAction) else "legal_v1"
        ),
        device="cpu",
        game_config=config,
    )


def test_new_training_defaults_use_survival_mask_with_legacy_safe_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args([])
    assert args.action_mask_mode == "topology_survival_v1"
    assert args.idle_limit_floor_steps == 0
    assert args.policy_anchor_final_weight == pytest.approx(args.policy_anchor_weight)
    assert args.policy_anchor_decay_steps == 0
    assert args.demonstration_terminal_exclusion_steps == 1

    legacy_args = parse_args(["--network-version", "2"])
    assert legacy_args.action_mask_mode == "legal_v1"

    env = SnakeGameEnv(GameConfig(width=5, height=5))
    agent = make_agent(env.config)
    agent.action_mask_mode = "topology_survival_v1"
    monkeypatch.setattr(
        env, "relative_topology_survival_mask", lambda: (False, True, False)
    )
    assert action_mask(agent, env) == [False, True, False]
    agent.action_mask_mode = "legal_v1"
    assert action_mask(agent, env) == [True, True, True]


def test_cpu_runtime_info_and_cleanup_are_safe() -> None:
    info = accelerator_runtime_info(torch.device("cpu"))

    assert info["device"] == "cpu"
    assert info["backend"] == "cpu"
    assert info["device_name"] is None
    assert info["torch_version"] == torch.__version__
    _release_accelerator_resources(torch.device("cpu"))


def test_rocm_runtime_info_reports_actual_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.14-test", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Test Radeon")

    info = accelerator_runtime_info(torch.device("cuda"))

    assert info["backend"] == "rocm"
    assert info["device_name"] == "Test Radeon"
    assert info["hip_version"] == "7.14-test"


def test_accelerator_cleanup_synchronizes_and_releases_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(training_module.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append(("sync", device))
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    _release_accelerator_resources(torch.device("cuda"))

    assert calls == ["gc", ("sync", torch.device("cuda")), "empty"]


def test_train_releases_accelerator_after_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    device = torch.device("cuda")
    monkeypatch.setattr(
        DQNAgent, "_resolve_device", staticmethod(lambda _value: device)
    )

    def interrupt(_args: object) -> None:
        calls.append("train")
        raise KeyboardInterrupt

    monkeypatch.setattr(training_module, "_train", interrupt)
    monkeypatch.setattr(
        training_module,
        "_release_accelerator_resources",
        lambda value: calls.append(("release", value)),
    )

    with pytest.raises(KeyboardInterrupt):
        training_module.train(SimpleNamespace(device="cuda"))

    assert calls == ["train", ("release", device)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_source_checkpoint(
    path: Path,
    *,
    episode: int = 17,
    config: GameConfig | None = None,
    checkpoint_role: str = "latest",
) -> DQNAgent:
    config = config or GameConfig(width=5, height=5, max_episode_steps=2)
    agent = make_agent(config)
    agent.behavior_steps = 321
    agent.learn_step_counter = 45
    agent.epsilon = 0.07
    args = parse_args(
        [
            "--width",
            str(config.width),
            "--height",
            str(config.height),
            "--max-steps",
            str(config.max_episode_steps),
            "--hidden",
            "16",
            "--output",
            str(path.parent / "unused-best.pt"),
            "--latest-output",
            str(path.parent / "unused-latest.pt"),
        ]
    )
    save_checkpoint(
        agent,
        path,
        episode=episode,
        run_seed=123,
        best_eval_score=None,
        best_eval_episode=None,
        train_args=args,
        checkpoint_role=checkpoint_role,
        episodes_started=episode,
    )
    return agent


def pending_best_promotion_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    DQNAgent,
    EvaluationConvergenceController,
    object,
]:
    config = GameConfig(width=5, height=5, max_episode_steps=2)
    best = tmp_path / "transaction-best.pt"
    latest = tmp_path / "transaction-latest.pt"
    args = parse_args(
        [
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "2",
            "--hidden",
            "16",
            "--output",
            str(best),
            "--latest-output",
            str(latest),
        ]
    )
    agent = make_agent(config)
    controller = EvaluationConvergenceController(
        lr_plateau_patience=0,
        lr_plateau_factor=0.5,
        lr_plateau_min=1e-6,
        early_stop_patience=0,
        early_stop_delta=0.1,
        require_paired_promotion=True,
        adaptive_eval_max_episodes=6,
        full_eval_confirmation_interval=8,
        full_eval_seed_base=400_000,
        full_eval_max_attempts=4,
    )
    controller.set_paired_reference([1.0, 1.0])
    save_checkpoint(
        agent,
        best,
        episode=0,
        run_seed=args.seed,
        best_eval_score=1.0,
        best_eval_episode=0,
        train_args=args,
        checkpoint_role="best_eval",
        best_checkpoint_path=best,
        convergence_controller=controller,
    )
    agent.behavior_steps = 777
    agent.learn_step_counter = 23
    controller.full_eval_attempts = 2
    controller.set_paired_reference([2.0, 2.0])
    marker = training_module._best_promotion_marker(best, score=2.0, episode=5)
    save_checkpoint(
        agent,
        latest,
        episode=5,
        run_seed=args.seed,
        best_eval_score=2.0,
        best_eval_episode=5,
        train_args=args,
        checkpoint_role="latest",
        best_checkpoint_path=best,
        episodes_started=6,
        convergence_controller=controller,
        pending_best_promotion=marker,
    )
    return best, latest, agent, controller, args


@pytest.mark.parametrize(
    "crash_window",
    ["marker_only", "best_checkpoint_only", "best_complete"],
)
def test_pending_best_promotion_reconciles_all_write_windows_without_rollback(
    tmp_path: Path, crash_window: str
) -> None:
    best, latest, agent, controller, args = pending_best_promotion_fixture(tmp_path)
    if crash_window == "best_checkpoint_only":
        agent.save(str(best))
    elif crash_window == "best_complete":
        save_checkpoint(
            agent,
            best,
            episode=5,
            run_seed=args.seed,
            best_eval_score=2.0,
            best_eval_episode=5,
            train_args=args,
            checkpoint_role="best_eval",
            best_checkpoint_path=best,
            episodes_started=6,
            convergence_controller=controller,
        )

    metadata = load_resume_metadata(latest, ignore_mismatch=False)
    restored_agent = DQNAgent.restore_training_checkpoint(str(latest), device="cpu")
    restored_controller = EvaluationConvergenceController.from_dict(
        metadata["convergence_controller"]
    )
    reconciled = training_module.reconcile_pending_best_promotion(
        metadata,
        restored_agent,
        restored_controller,
        args,
        resume_path=latest,
        output_path=best,
    )

    best_metadata = json.loads(best.with_suffix(".meta.json").read_text("utf-8"))
    latest_metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    reloaded_latest = DQNAgent.restore_training_checkpoint(str(latest), device="cpu")
    assert reconciled["pending_best_promotion"] is None
    assert latest_metadata["pending_best_promotion"] is None
    assert latest_metadata["episodes_completed"] == 5
    assert latest_metadata["episodes_started"] == 6
    assert reloaded_latest.behavior_steps == 777
    assert reloaded_latest.learn_step_counter == 23
    assert best_metadata["best_eval_score"] == latest_metadata["best_eval_score"] == 2.0
    assert best_metadata["best_eval_episode"] == latest_metadata["best_eval_episode"] == 5
    assert latest_metadata["best_checkpoint_sha256"] == file_sha256(best)
    assert best_metadata["checkpoint_sha256"] == file_sha256(best)
    assert (
        best_metadata["convergence_controller"]["state"]["full_eval_attempts"]
        == latest_metadata["convergence_controller"]["state"]["full_eval_attempts"]
        == 2
    )


def test_pending_best_promotion_rejects_unknown_best_artifact(tmp_path: Path) -> None:
    best, latest, _agent, _controller, args = pending_best_promotion_fixture(tmp_path)
    unknown = make_agent(GameConfig(width=5, height=5, max_episode_steps=2))
    with torch.no_grad():
        next(unknown.policy_net.parameters()).add_(1.0)
    save_checkpoint(
        unknown,
        best,
        episode=99,
        run_seed=args.seed,
        best_eval_score=99.0,
        best_eval_episode=99,
        train_args=args,
        checkpoint_role="best_eval",
        best_checkpoint_path=best,
    )
    metadata = load_resume_metadata(latest, ignore_mismatch=False)
    restored_agent = DQNAgent.restore_training_checkpoint(str(latest), device="cpu")
    restored_controller = EvaluationConvergenceController.from_dict(
        metadata["convergence_controller"]
    )

    with pytest.raises(RuntimeError, match="unknown best artifact"):
        training_module.reconcile_pending_best_promotion(
            metadata,
            restored_agent,
            restored_controller,
            args,
            resume_path=latest,
            output_path=best,
        )


def test_episode_seed_is_stable_and_stream_separated() -> None:
    assert deterministic_episode_seed(42, 5) == deterministic_episode_seed(42, 5)
    assert deterministic_episode_seed(42, 5) != deterministic_episode_seed(42, 6)
    assert deterministic_episode_seed(42, 5, 0) != deterministic_episode_seed(42, 5, 1)


def test_full_resume_rejects_best_checkpoint_role() -> None:
    agent = make_agent(GameConfig(width=5, height=5))
    metadata = {
        "checkpoint_role": "best_eval",
        "network_version": agent.network_version,
        "action_dim": agent.action_dim,
        "action_mask_mode": agent.action_mask_mode,
        "truncation_bootstrap_mode": agent.truncation_bootstrap_mode,
        "time_feature_mode": agent.time_feature_mode,
        "obs_shape": list(agent.obs_shape),
        "behavior_steps": agent.behavior_steps,
        "learn_step_counter": agent.learn_step_counter,
    }

    with pytest.raises(RuntimeError, match="Use --warm-start-from"):
        validate_resume_identity(metadata, agent)


def test_state_potential_is_bounded() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6))
    env.reset(seed=2)
    assert 0.0 <= state_potential(env) <= 1.0


def test_terminal_potential_shaping_has_no_bootstrap_value() -> None:
    env = SnakeGameEnv(GameConfig(width=6, height=6, max_episode_steps=1))
    env.reset(seed=2)
    previous = state_potential(env)
    assert potential_shaping(
        previous, env, gamma=0.99, scale=1.5, terminal=True
    ) == pytest.approx(-1.5 * previous)


def test_output_paths_must_be_distinct() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output", "same.pt", "--latest-output", "same.pt"])


def test_resume_source_cannot_be_the_best_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--resume-from",
                str(source),
                "--output",
                str(source),
                "--latest-output",
                str(tmp_path / "latest.pt"),
            ]
        )
    assert "immutable best" in capsys.readouterr().err


def test_warm_start_arguments_require_one_distinct_existing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    common = [
        "--output",
        str(tmp_path / "best.pt"),
        "--latest-output",
        str(tmp_path / "latest.pt"),
    ]
    with pytest.raises(SystemExit):
        parse_args(
            ["--warm-start-from", str(source), "--resume-from", str(source), *common]
        )
    with pytest.raises(SystemExit):
        parse_args(["--warm-start-from", str(source), "--fresh", *common])
    with pytest.raises(SystemExit):
        parse_args(["--warm-start-from", str(tmp_path / "missing.pt"), *common])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--resume-epsilon",
                "0.4",
                *common,
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--output",
                str(source),
                "--latest-output",
                str(tmp_path / "latest.pt"),
            ]
        )
    # source.meta.json must not be usable as an output checkpoint either.
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--warm-start-from",
                str(source),
                "--output",
                str(source.with_suffix(".meta.json")),
                "--latest-output",
                str(tmp_path / "latest.pt"),
            ]
        )


def test_output_checkpoint_and_sidecar_artifacts_cannot_cross(tmp_path: Path) -> None:
    latest = tmp_path / "collision.pt"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output",
                str(latest.with_suffix(".meta.json")),
                "--latest-output",
                str(latest),
            ]
        )


def test_parallel_collection_defaults_and_validation() -> None:
    args = parse_args([])
    assert args.num_envs == 1
    assert args.rollout_steps == 1
    assert args.updates_per_collection == 0
    with pytest.raises(SystemExit):
        parse_args(["--num-envs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--rollout-steps", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--updates-per-collection", "-1"])


def test_resume_seed_change_requires_explicit_permission() -> None:
    args = parse_args(["--seed", "9"])
    with pytest.raises(RuntimeError, match="seed changed"):
        validate_resume_seed({"run_seed": 8}, args)
    args.allow_seed_change = True
    validate_resume_seed({"run_seed": 8}, args)


def test_resume_evaluation_execution_migrates_legacy_and_gates_refill_size() -> None:
    legacy_args = parse_args([])
    training_module.restore_evaluation_execution(
        {"evaluation_identity": {"eval_episodes": 50}}, legacy_args
    )
    assert legacy_args.eval_execution_mode == "all_active_v1"

    metadata = {
        "evaluation_identity": {
            "evaluation_execution": {
                "schema": "refill_slots_v1",
                "vector_envs": 32,
            }
        }
    }
    adopted = parse_args([])
    training_module.restore_evaluation_execution(metadata, adopted)
    assert adopted.eval_execution_mode == "refill_slots_v1"
    assert adopted.eval_vector_envs == 32
    conflicting = parse_args(["--eval-vector-envs", "16"])
    with pytest.raises(RuntimeError, match="vector size conflicts"):
        training_module.restore_evaluation_execution(metadata, conflicting)


def test_full_resume_rejects_finite_horizon_to_timeless_identity_switch() -> None:
    finite_config = GameConfig(width=5, height=5, max_episode_steps=100)
    agent = make_agent(finite_config)
    args = parse_args(
        [
            "--max-steps",
            "0",
            "--time-feature-mode",
            "constant_zero_v1",
        ]
    )

    with pytest.raises(RuntimeError, match="hyperparameters conflict"):
        validate_resume_agent_options(args, agent)
    with pytest.raises(RuntimeError, match="environment/MDP differs"):
        training_module.validate_resume_environment(
            agent,
            GameConfig(width=5, height=5, max_episode_steps=0),
            allow_change=False,
        )


def test_fresh_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    best.write_bytes(b"best")
    latest.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    args = parse_args(["--output", str(best), "--latest-output", str(latest)])
    with pytest.raises(RuntimeError, match="existing output artifacts"):
        _prepare_fresh_outputs(args, None)
    args.overwrite_fresh_output = True
    _prepare_fresh_outputs(args, None)
    assert not best.exists()
    assert not latest.with_suffix(".meta.json").exists()


def test_relative_checkpoint_action_decodes_against_heading() -> None:
    config = GameConfig(width=6, height=6)
    env = SnakeGameEnv(config)
    env.reset(seed=1)
    agent = make_agent(config)
    assert env.direction is Action.RIGHT
    assert absolute_action(agent, env, int(RelativeAction.STRAIGHT)) is Action.RIGHT
    assert absolute_action(agent, env, int(RelativeAction.LEFT)) is Action.UP
    assert absolute_action(agent, env, int(RelativeAction.RIGHT)) is Action.DOWN


def test_fixed_seed_evaluation_is_repeatable() -> None:
    config = GameConfig(width=5, height=5, max_idle_steps=20)
    torch.manual_seed(3)
    agent = make_agent(config)
    first = evaluate_agent(agent, config, [10, 11, 12], max_steps=50)
    second = evaluate_agent(agent, config, [10, 11, 12], max_steps=50)
    assert first == second
    assert first["seeds"] == [10, 11, 12]
    assert len(first["score_samples"]) == 3
    assert first["score"]["mean"] == pytest.approx(sum(first["score_samples"]) / 3)


def test_zero_max_steps_is_truly_unbounded_and_uses_timeless_identity() -> None:
    args = parse_args(["--max-steps", "0"])
    config = GameConfig(width=5, height=5, max_episode_steps=args.max_steps)

    assert episode_step_limit(config, args.max_steps) is None
    assert args.time_feature_mode == "constant_zero_v1"
    finite = parse_args(["--max-steps", "100"])
    assert finite.time_feature_mode == "finite_horizon_progress_v1"
    assert episode_step_limit(
        GameConfig(width=5, height=5, max_episode_steps=100), finite.max_steps
    ) == 100


def test_refill_evaluator_preserves_seed_order_and_matches_all_active(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = GameConfig(width=5, height=5, max_idle_steps=10)
    agent = make_agent(config)
    capacities: list[int] = []
    original_encoder = training_module.BatchObservationEncoder

    def tracking_encoder(*args: object, **kwargs: object) -> object:
        capacities.append(int(kwargs["max_batch_size"]))
        return original_encoder(*args, **kwargs)

    monkeypatch.setattr(training_module, "BatchObservationEncoder", tracking_encoder)
    seeds = [19, 3, 41, 7, 23]
    refill = evaluate_agent(
        agent,
        config,
        seeds,
        max_steps=None,
        vector_envs=2,
        execution_mode="refill_slots_v1",
        heartbeat_seconds=1e-12,
    )
    all_active = evaluate_agent(
        agent,
        config,
        seeds,
        max_steps=None,
        vector_envs=2,
        execution_mode="all_active_v1",
    )

    assert capacities == [2, 5]
    assert refill == all_active
    assert refill["seeds"] == seeds
    progress = capsys.readouterr().out
    assert "Evaluation progress:" in progress
    assert "score" not in progress.lower()


def test_fixed_evaluation_counts_environment_time_limit() -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=1)
    agent = make_agent(config)
    result = evaluate_agent(agent, config, [1, 2, 3], max_steps=10)
    assert result["truncated_count"] == 3
    assert result["terminal_events"] == {"time_limit": 3}


def test_console_episode_obeys_checkpoint_horizon() -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=1)
    agent = make_agent(config)
    result = run_episode(agent, SnakeGameEnv(config), 0.0, False, False)
    assert result["steps"] == 1


def test_v3_contract_rejects_observation_without_horizon_channel() -> None:
    config = GameConfig(width=5, height=5)
    agent = DQNAgent(
        state_dim=19 * 5 * 5,
        action_dim=3,
        hidden_sizes=(16,),
        obs_shape=(19, 5, 5),
        network_version=3,
        device="cpu",
        game_config=config,
    )
    with pytest.raises(RuntimeError, match="20 observation channels"):
        validate_v3_contract(agent)


def test_resume_metadata_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    config = GameConfig(width=5, height=5)
    agent = make_agent(config)
    checkpoint = tmp_path / "latest.pt"
    agent.save(str(checkpoint))
    sidecar = checkpoint.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"checkpoint_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        load_resume_metadata(checkpoint, ignore_mismatch=False)
    assert load_resume_metadata(checkpoint, ignore_mismatch=True) == {}


@pytest.mark.parametrize("failure", ["wrong_role", "tampered_checkpoint"])
def test_policy_evaluation_reference_requires_authenticated_best(
    tmp_path: Path, failure: str
) -> None:
    config = GameConfig(width=5, height=5, max_episode_steps=2)
    best = tmp_path / "best.pt"
    agent = make_agent(config)
    args = parse_args(
        [
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "2",
            "--hidden",
            "16",
            "--output",
            str(best),
            "--latest-output",
            str(tmp_path / "latest.pt"),
        ]
    )
    save_checkpoint(
        agent,
        best,
        episode=3,
        run_seed=args.seed,
        best_eval_score=2.0,
        best_eval_episode=3,
        train_args=args,
        checkpoint_role="best_eval",
        best_checkpoint_path=best,
    )
    if failure == "wrong_role":
        metadata = json.loads(best.with_suffix(".meta.json").read_text("utf-8"))
        metadata["checkpoint_role"] = "latest"
        best.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        match = "best_eval sidecar"
    else:
        best.write_bytes(best.read_bytes() + b"tampered")
        match = "SHA-256 authentication"

    with pytest.raises(RuntimeError, match=match):
        training_module.load_policy_evaluation_reference(best, agent, config)


def test_warm_start_metadata_is_required_and_authenticated(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=False
    )
    assert verified is True
    assert metadata["checkpoint_role"] == "latest"
    assert actual_sha256 == file_sha256(source)

    source.with_suffix(".meta.json").unlink()
    with pytest.raises(RuntimeError, match="metadata is missing"):
        load_warm_start_metadata(source, ignore_mismatch=False)
    assert load_warm_start_metadata(source, ignore_mismatch=True) == (
        {},
        False,
        file_sha256(source),
    )

    source.with_suffix(".meta.json").write_text(
        json.dumps({"checkpoint_sha256": "0" * 64, "checkpoint_role": "latest"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        load_warm_start_metadata(source, ignore_mismatch=False)
    ignored_metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=True
    )
    assert verified is False
    assert ignored_metadata["checkpoint_role"] == "latest"
    assert actual_sha256 == file_sha256(source)


@pytest.mark.parametrize("invalid_json", ["[]", "null", '"metadata"'])
def test_warm_start_non_object_metadata_obeys_ignore_flag(
    tmp_path: Path, invalid_json: str
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    source.with_suffix(".meta.json").write_text(invalid_json, encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        load_warm_start_metadata(source, ignore_mismatch=False)

    metadata, verified, actual_sha256 = load_warm_start_metadata(
        source, ignore_mismatch=True
    )
    assert metadata == {}
    assert verified is False
    assert actual_sha256 == file_sha256(source)


@pytest.mark.parametrize("initialization_mode", ["fresh", "warm"])
def test_legacy_network_cannot_use_v3_output_identity(
    tmp_path: Path, initialization_mode: str
) -> None:
    options = [
        "--episodes",
        "1",
        "--network-version",
        "1",
        "--device",
        "cpu",
        "--disable-amp",
    ]
    if initialization_mode == "fresh":
        options.append("--fresh")
    else:
        source = tmp_path / "legacy-source.pt"
        source.write_bytes(b"source is not read before the output identity gate")
        options.extend(["--warm-start-from", str(source)])

    with pytest.raises(RuntimeError, match="legacy v1/v2 run"):
        train(parse_args(options))


def test_invalid_warm_start_never_deletes_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    best.write_bytes(b"preserve-best")
    latest.write_bytes(b"preserve-latest")
    best_hash = file_sha256(best)
    latest_hash = file_sha256(latest)

    # The source uses hidden size 16, so this intentional size mismatch must fail
    # before --overwrite-fresh-output is allowed to remove either destination.
    args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "6",
            "--height",
            "7",
            "--max-steps",
            "1",
            "--hidden",
            "32",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--overwrite-fresh-output",
            "--log-dir",
            str(tmp_path / "logs"),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(
        RuntimeError, match="Explicit warm-start network options conflict"
    ):
        train(args)
    assert file_sha256(best) == best_hash
    assert file_sha256(latest) == latest_hash


def test_cross_map_training_requires_best_eval_source_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "latest.pt"
    save_source_checkpoint(source, checkpoint_role="latest")
    args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "6",
            "--height",
            "6",
            "--max-steps",
            "1",
            "--hidden",
            "16",
            "--warm-start-from",
            str(source),
            "--output",
            str(tmp_path / "new-best.pt"),
            "--latest-output",
            str(tmp_path / "new-latest.pt"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )

    with pytest.raises(RuntimeError, match="requires an authenticated best_eval"):
        train(args)


def test_warm_start_is_fresh_cross_map_training_and_resume_keeps_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(
        source,
        config=GameConfig(width=8, height=8, max_episode_steps=2),
        checkpoint_role="best_eval",
    )
    source_sidecar = source.with_suffix(".meta.json")
    source_hash = file_sha256(source)
    source_sidecar_hash = file_sha256(source_sidecar)

    best = tmp_path / "new-best.pt"
    latest = tmp_path / "new-latest.pt"
    logs = tmp_path / "logs"
    # An existing latest must not trigger automatic resume when warm start is explicit.
    latest.write_bytes(b"stale destination, not a resumable checkpoint")
    args = parse_args(
        [
            "--episodes",
            "2",
            "--width",
            "10",
            "--height",
            "10",
            "--max-steps",
            "2",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "1",
            "--checkpoint-interval",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "0.003",
            "--min-replay",
            "32",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--overwrite-fresh-output",
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    original_evaluate = training_module.evaluate_agent
    baseline_calls: list[tuple[int, int]] = []

    def checked_evaluate(
        *call_args: object, **call_kwargs: object
    ) -> dict[str, object]:
        evaluated_agent = call_args[0]
        assert isinstance(evaluated_agent, DQNAgent)
        baseline_calls.append(
            (evaluated_agent.behavior_steps, evaluated_agent.learn_step_counter)
        )
        return original_evaluate(*call_args, **call_kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(training_module, "evaluate_agent", checked_evaluate)
    train(args)

    assert baseline_calls == [(0, 0)]
    assert file_sha256(source) == source_hash
    assert file_sha256(source_sidecar) == source_sidecar_hash
    metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    provenance = metadata["warm_start_provenance"]
    assert metadata["episodes_completed"] == 2
    assert metadata["episodes_started"] == 2
    assert metadata["behavior_steps"] <= 4
    assert metadata["learn_step_counter"] == 0
    assert metadata["effective_agent_config"]["batch_size"] == 2
    assert metadata["effective_agent_config"]["lr"] == [0.003]
    assert metadata["obs_shape"] == [20, 10, 10]
    assert provenance["source_path"] == str(source.resolve())
    assert provenance["checkpoint_sha256"] == source_hash
    assert provenance["sidecar_role"] == "best_eval"
    assert provenance["sidecar_episode"] == 17
    assert provenance["source_sidecar_role"] == "best_eval"
    assert provenance["source_sidecar_episode"] == 17
    assert provenance["source_network_version"] == 3
    assert provenance["source_obs_shape"] == [20, 8, 8]
    assert provenance["target_obs_shape"] == [20, 10, 10]
    assert provenance["source_map"] == {"width": 8, "height": 8}
    assert provenance["target_map"] == {"width": 10, "height": 10}
    assert provenance["cross_map"] is True
    assert provenance["source_sidecar_verified"] is True
    assert provenance["optimizer_restored"] is False
    assert provenance["replay_restored"] is False
    assert provenance["rng_restored"] is False
    assert metadata["policy_transfer_provenance"] == provenance
    log_path = next(logs.glob("train_log_*.jsonl"))
    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    run_start = next(
        record for record in records if record["record_type"] == "run_start"
    )
    assert run_start["start_episode"] == 1
    assert run_start["resume_path"] is None
    assert run_start["warm_start_provenance"] == provenance
    baseline = next(
        record
        for record in records
        if record["record_type"] == "evaluation" and record["episode"] == 0
    )
    assert baseline["evaluation_kind"] == "warm_start_baseline"
    assert baseline["eval_score_mean"] == metadata["best_eval_score"]
    assert baseline["convergence_decision"]["significant_improvement"] is True
    assert baseline["current_learning_rates"] == [0.003]
    assert all(
        record.get("episode") != 0
        for record in records
        if record["record_type"] == "episode"
    )

    resume_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "10",
            "--height",
            "10",
            "--max-steps",
            "2",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "1",
            "--checkpoint-interval",
            "1",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(resume_args)
    resumed = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert resumed["episodes_completed"] == 3
    assert resumed["warm_start_provenance"] == provenance
    assert file_sha256(source) == source_hash
    assert file_sha256(source_sidecar) == source_sidecar_hash


def test_warm_start_baseline_survives_regression_and_min_lr_stops_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    source_hash = file_sha256(source)
    source_sidecar_hash = file_sha256(source.with_suffix(".meta.json"))
    best = tmp_path / "stable-best.pt"
    latest = tmp_path / "stable-latest.pt"
    scores = iter((5.0, 3.0, 3.0))

    def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
        score = next(scores)
        distribution = {
            "mean": score,
            "std": 0.0,
            "ci95_low": score,
            "ci95_high": score,
            "median": score,
            "p10": score,
            "p90": score,
            "min": score,
            "max": score,
        }
        return {
            "reward": dict(distribution),
            "score": dict(distribution),
            "steps": dict(distribution),
            "terminal_events": {"test": 1},
            "truncated_count": 0,
            "episodes": 1,
        }

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    args = parse_args(
        [
            "--episodes",
            "10",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-episodes",
            "1",
            "--checkpoint-interval",
            "1",
            "--batch-size",
            "2",
            "--min-replay",
            "32",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--lr-plateau-patience",
            "1",
            "--lr-plateau-factor",
            "0.5",
            "--lr-plateau-min",
            "0.0001",
            "--early-stop-patience",
            "1",
            "--early-stop-delta",
            "1.0",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(tmp_path / "logs"),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )

    train(args)

    best_metadata = json.loads(best.with_suffix(".meta.json").read_text("utf-8"))
    latest_metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert best_metadata["best_eval_score"] == pytest.approx(5.0)
    assert best_metadata["best_eval_episode"] == 0
    assert latest_metadata["best_eval_score"] == pytest.approx(5.0)
    assert latest_metadata["best_eval_episode"] == 0
    assert latest_metadata["episodes_completed"] == 2
    assert latest_metadata["current_learning_rates"] == [0.0001]
    assert latest_metadata["convergence_controller"]["state"]["reductions"] == 1
    assert latest_metadata["convergence_controller"]["state"]["min_lr_evaluations"] == 1
    assert file_sha256(source) == source_hash
    assert file_sha256(source.with_suffix(".meta.json")) == source_sidecar_hash


def test_adaptive_warm_start_promotion_and_resume_keep_full_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    best = tmp_path / "adaptive-best.pt"
    latest = tmp_path / "adaptive-latest.pt"
    logs = tmp_path / "adaptive-logs"
    calls: list[list[int]] = []

    def fake_evaluate(
        _agent: DQNAgent,
        _config: GameConfig,
        seeds: list[int],
        _max_steps: int,
        **_options: object,
    ) -> dict[str, object]:
        current = list(seeds)
        calls.append(current)
        score = 0.0 if len(calls) == 1 else 1.0
        return _evaluation_payload(current, score)

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    common = [
        "--width",
        "5",
        "--height",
        "5",
        "--max-steps",
        "1",
        "--eval-interval",
        "1",
        "--eval-episodes",
        "2",
        "--eval-seed-base",
        "300000",
        "--adaptive-eval-max-episodes",
        "6",
        "--adaptive-eval-growth-factor",
        "2",
        "--checkpoint-interval",
        "1",
        "--batch-size",
        "2",
        "--min-replay",
        "32",
        "--replay-capacity",
        "64",
        "--hidden",
        "16",
        "--require-paired-promotion",
        "--paired-promotion-min-delta",
        "0.1",
        "--early-stop-delta",
        "0.25",
        "--output",
        str(best),
        "--latest-output",
        str(latest),
        "--log-dir",
        str(logs),
        "--device",
        "cpu",
        "--disable-amp",
    ]
    train(parse_args(["--episodes", "1", "--warm-start-from", str(source), *common]))

    assert calls == [
        list(range(300000, 300006)),
        [300000, 300001],
        [300002, 300003],
        [300004, 300005],
    ]
    latest_metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    controller_state = latest_metadata["convergence_controller"]["state"]
    assert latest_metadata["best_eval_episode"] == 1
    assert controller_state["reference_scores"] == [1.0] * 6
    records = [
        json.loads(line)
        for line in next(logs.glob("train_log_*.jsonl")).read_text("utf-8").splitlines()
    ]
    evaluation_record = next(
        record
        for record in records
        if record.get("record_type") == "evaluation" and record.get("episode") != 0
    )
    assert evaluation_record["eval_episodes_actual"] == 6
    assert evaluation_record["eval_episodes_planned"] == 6
    assert evaluation_record["eval_episodes_max"] == 6
    assert evaluation_record["eval_expansion_stage"] == 3
    assert evaluation_record["eval_planned_looks"] == 3
    assert evaluation_record["eval_statistical_method"] == "paired_normal_bonferroni_v1"
    assert evaluation_record["eval_statistical_state"] == "confirmed_improvement"
    assert evaluation_record["eval_patience_deferred"] is False

    resume_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-episodes",
            "2",
            "--eval-seed-base",
            "300000",
            "--checkpoint-interval",
            "1",
            "--batch-size",
            "2",
            "--min-replay",
            "32",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(resume_args)
    assert resume_args.require_paired_promotion is True
    assert resume_args.adaptive_eval_max_episodes == 6
    assert resume_args.adaptive_eval_growth_factor == 2.0
    resumed = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert resumed["convergence_controller"]["state"]["reference_scores"] == [1.0] * 6


def test_teacher_replay_blocks_learning_and_paired_regression_stops_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    save_source_checkpoint(source)
    best = tmp_path / "guarded-best.pt"
    latest = tmp_path / "guarded-latest.pt"
    sample_sets = iter(([5.0] * 4, [3.0] * 4, [3.0] * 4, [3.0] * 4))

    def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
        samples = list(next(sample_sets))
        mean = sum(samples) / len(samples)
        distribution = {
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "median": mean,
            "p10": mean,
            "p90": mean,
            "min": mean,
            "max": mean,
        }
        return {
            "reward": dict(distribution),
            "score": dict(distribution),
            "steps": dict(distribution),
            "seeds": [100, 101, 102, 103],
            "reward_samples": samples,
            "score_samples": samples,
            "step_samples": samples,
            "terminal_events": {"test": 4},
            "truncated_count": 0,
            "episodes": 4,
        }

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    args = parse_args(
        [
            "--episodes",
            "10",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-episodes",
            "4",
            "--checkpoint-interval",
            "1",
            "--batch-size",
            "2",
            "--min-replay",
            "2",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--policy-anchor-weight",
            "0.5",
            "--teacher-replay-steps",
            "2",
            "--demonstration-capacity",
            "16",
            "--demonstration-batch-fraction",
            "0.5",
            "--elite-demonstration-batch-fraction",
            "0.5",
            "--demonstration-min-score",
            "0",
            "--demonstration-min-return",
            "-100",
            "--demonstration-elite-score",
            "0",
            "--demonstration-elite-return",
            "-100",
            "--imitation-loss-weight",
            "0.5",
            "--imitation-margin",
            "0.8",
            "--require-paired-promotion",
            "--paired-promotion-min-delta",
            "0.1",
            "--regression-stop-patience",
            "2",
            "--regression-stop-delta",
            "0.1",
            "--warm-start-from",
            str(source),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(tmp_path / "logs"),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )

    train(args)

    latest_metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert latest_metadata["episodes_completed"] == 3
    assert latest_metadata["learn_step_counter"] == 2
    assert latest_metadata["best_eval_score"] == pytest.approx(5.0)
    assert latest_metadata["best_eval_episode"] == 0
    assert latest_metadata["action_mask_mode"] == "topology_survival_v1"
    assert latest_metadata["effective_agent_config"]["policy_anchor_enabled"] is True
    assert latest_metadata["effective_agent_config"][
        "policy_anchor_final_weight"
    ] == pytest.approx(0.5)
    assert latest_metadata["effective_agent_config"]["policy_anchor_decay_steps"] == 0
    assert latest_metadata["effective_agent_config"][
        "demonstration_terminal_exclusion_steps"
    ] == 1
    assert latest_metadata["effective_agent_config"][
        "demonstration_batch_fraction"
    ] == pytest.approx(0.5)
    assert latest_metadata["effective_agent_config"][
        "imitation_loss_weight"
    ] == pytest.approx(0.5)
    assert latest_metadata["demonstration_replay_size_at_save"] > 0
    assert latest_metadata["demonstration_replay_elite_count_at_save"] > 0
    assert latest_metadata["demonstration_trajectories_seen_lifetime"] > 0
    assert latest_metadata["demonstration_transitions_promoted_lifetime"] > 0
    assert (
        latest_metadata["convergence_controller"]["state"]["regression_evaluations"]
        == 2
    )
    records = [
        json.loads(line)
        for line in next((tmp_path / "logs").glob("train_log_*.jsonl"))
        .read_text("utf-8")
        .splitlines()
    ]
    assert any(
        record.get("convergence_decision", {}).get("decision")
        == "paired_regression_patience"
        for record in records
    )
    prewarm = [
        record
        for record in records
        if record.get("record_type") == "collection"
        and record.get("teacher_replay_complete") is False
    ]
    assert prewarm
    assert all(record["collection_updates"] == 0 for record in prewarm)
    assert any(
        record.get("convergence_decision", {}).get("decision")
        == "teacher_replay_warmup"
        for record in records
    )
    learned_collections = [
        record
        for record in records
        if record.get("record_type") == "collection"
        and record.get("collection_updates", 0) > 0
    ]
    assert learned_collections
    assert all(
        record["avg_imitation_loss"] is not None for record in learned_collections
    )
    assert all(
        record["avg_demonstration_batch_fraction"] == pytest.approx(0.5)
        for record in learned_collections
    )
    assert all(
        "effective_policy_anchor_weight" in record
        for record in learned_collections
    )
    assert (
        max(record.get("demonstration_replay_elite_count", 0) for record in records) > 0
    )


def test_short_training_creates_distinct_latest_and_best(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    logs = tmp_path / "logs"
    args = parse_args(
        [
            "--episodes",
            "4",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-interval",
            "2",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "2",
            "--batch-size",
            "4",
            "--lr",
            "0.001",
            "--min-replay",
            "4",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(args)
    assert latest.exists()
    assert best.exists()
    assert latest != best
    latest_meta = json.loads(
        latest.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    best_meta = json.loads(best.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert latest_meta["checkpoint_role"] == "latest"
    assert best_meta["checkpoint_role"] == "best_eval"
    assert latest_meta["best_checkpoint_path"] == str(best.resolve())
    assert latest_meta["best_checkpoint_sha256"] == best_meta["checkpoint_sha256"]
    assert latest_meta["effective_agent_config"]["lr"] == [0.001]
    assert latest_meta["episodes_completed"] == 4
    assert latest_meta["episodes_started"] == 4
    assert list(logs.glob("train_log_*.jsonl"))

    resume_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-interval",
            "10",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "1",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(resume_args)
    resumed_meta = json.loads(
        latest.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert resumed_meta["episodes_completed"] == 5
    assert resumed_meta["epsilon"] >= 0.05
    assert resumed_meta["effective_agent_config"]["lr"] == [0.001]

    conflicting_lr_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "25",
            "--max-idle-steps",
            "12",
            "--eval-episodes",
            "2",
            "--lr",
            "0.002",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(RuntimeError, match="hyperparameters conflict"):
        train(conflicting_lr_args)

    drift_args = parse_args(
        [
            "--episodes",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "10",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    with pytest.raises(RuntimeError, match="environment/MDP"):
        train(drift_args)


def test_collection_log_interval_keeps_evaluation_and_final_records_compact(
    tmp_path: Path,
) -> None:
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    logs = tmp_path / "logs"
    train(
        parse_args(
            [
                "--episodes",
                "3",
                "--width",
                "5",
                "--height",
                "5",
                "--max-steps",
                "1",
                "--eval-interval",
                "2",
                "--eval-episodes",
                "2",
                "--checkpoint-interval",
                "2",
                "--collection-log-interval",
                "10",
                "--batch-size",
                "2",
                "--min-replay",
                "2",
                "--replay-capacity",
                "16",
                "--hidden",
                "16",
                "--output",
                str(best),
                "--latest-output",
                str(latest),
                "--log-dir",
                str(logs),
                "--device",
                "cpu",
                "--disable-amp",
            ]
        )
    )

    records = [
        json.loads(line)
        for line in next(logs.glob("train_log_*.jsonl"))
        .read_text("utf-8")
        .splitlines()
    ]
    collections = [r for r in records if r["record_type"] == "collection"]
    episodes = [r for r in records if r["record_type"] == "episode"]
    evaluations = [r for r in records if r["record_type"] == "evaluation"]
    assert len(collections) == 2
    assert len(episodes) == 3
    assert len(evaluations) == 1
    assert evaluations[0]["episode"] == 2
    assert "reference_scores" in evaluations[0]["convergence_controller"]["state"]
    for record in [*collections, *episodes]:
        state = record["convergence_controller"]["state"]
        assert "reference_scores" not in state
        assert "reference_scores_count" in state
    assert not any("eval_score_mean" in record for record in episodes)


def test_gated_training_uses_fresh_full_pair_and_persists_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    reference_marker = save_source_checkpoint(source)
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    logs = tmp_path / "logs"
    calls: list[tuple[DQNAgent, list[int]]] = []

    def fake_evaluate(
        evaluated_agent: DQNAgent,
        _config: GameConfig,
        seeds: list[int],
        _max_steps: int,
        **_options: object,
    ) -> dict[str, object]:
        seed_list = list(seeds)
        calls.append((evaluated_agent, seed_list))
        if seed_list[0] >= 400_000:
            score = 0.0 if evaluated_agent is reference_marker else 1.0
        elif len(seed_list) == 2:
            score = 1.0
        else:
            score = 0.0
        return _evaluation_payload(seed_list, score)

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    monkeypatch.setattr(
        training_module,
        "load_policy_evaluation_reference",
        lambda *_args: reference_marker,
    )
    train(
        parse_args(
            [
                "--episodes",
                "1",
                "--width",
                "5",
                "--height",
                "5",
                "--max-steps",
                "1",
                "--eval-interval",
                "1",
                "--eval-episodes",
                "2",
                "--adaptive-eval-max-episodes",
                "6",
                "--full-eval-confirmation-interval",
                "8",
                "--full-eval-seed-base",
                "400000",
                "--full-eval-max-attempts",
                "4",
                "--require-paired-promotion",
                "--paired-promotion-min-delta",
                "0.1",
                "--batch-size",
                "2",
                "--min-replay",
                "2",
                "--replay-capacity",
                "16",
                "--hidden",
                "16",
                "--warm-start-from",
                str(source),
                "--output",
                str(best),
                "--latest-output",
                str(latest),
                "--log-dir",
                str(logs),
                "--device",
                "cpu",
                "--disable-amp",
            ]
        )
    )

    fixed = list(range(100_000, 100_006))
    probe = fixed[:2]
    fresh = list(range(400_000, 400_006))
    assert [seeds for _, seeds in calls] == [fixed, probe, fresh, fresh]
    assert set(probe).isdisjoint(fresh)
    metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    state = metadata["convergence_controller"]["state"]
    assert state["full_eval_attempts"] == 1
    assert state["evaluation_episodes"] == 20
    assert metadata["best_eval_episode"] == 1
    records = [
        json.loads(line)
        for line in next(logs.glob("train_log_*.jsonl"))
        .read_text("utf-8")
        .splitlines()
    ]
    full = next(
        record
        for record in records
        if record.get("evaluation_kind") == "full_confirmation"
    )
    assert full["eval_full_attempt_index"] == 0
    assert full["eval_execution_episodes"] == 14
    assert full["eval_reference_score_samples"] == [0.0] * 6
    assert full["convergence_decision"]["paired_promotion_eligible"] is True


def test_fresh_gated_training_establishes_episode_zero_best_before_first_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    best = tmp_path / "fresh-gated-best.pt"
    latest = tmp_path / "fresh-gated-latest.pt"
    logs = tmp_path / "fresh-gated-logs"
    calls: list[list[int]] = []

    def fake_evaluate(
        _agent: DQNAgent,
        _config: GameConfig,
        seeds: list[int],
        _max_steps: int,
        **_options: object,
    ) -> dict[str, object]:
        seed_list = list(seeds)
        calls.append(seed_list)
        if len(calls) == 2:
            assert best.is_file()
            assert latest.is_file()
            baseline_metadata = json.loads(
                latest.with_suffix(".meta.json").read_text("utf-8")
            )
            assert baseline_metadata["best_eval_episode"] == 0
            assert baseline_metadata["episodes_completed"] == 1
        return _evaluation_payload(seed_list, 1.0)

    monkeypatch.setattr(training_module, "evaluate_agent", fake_evaluate)
    train(
        parse_args(
            [
                "--episodes",
                "1",
                "--width",
                "5",
                "--height",
                "5",
                "--max-steps",
                "1",
                "--eval-interval",
                "1",
                "--eval-episodes",
                "2",
                "--eval-seed-base",
                "100",
                "--adaptive-eval-max-episodes",
                "6",
                "--full-eval-confirmation-interval",
                "8",
                "--full-eval-seed-base",
                "1000",
                "--full-eval-max-attempts",
                "4",
                "--require-paired-promotion",
                "--paired-promotion-min-delta",
                "0.1",
                "--batch-size",
                "2",
                "--min-replay",
                "2",
                "--replay-capacity",
                "16",
                "--hidden",
                "16",
                "--output",
                str(best),
                "--latest-output",
                str(latest),
                "--log-dir",
                str(logs),
                "--device",
                "cpu",
                "--disable-amp",
            ]
        )
    )

    assert calls == [list(range(100, 106)), [100, 101]]
    best_metadata = json.loads(best.with_suffix(".meta.json").read_text("utf-8"))
    latest_metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert best_metadata["checkpoint_role"] == "best_eval"
    assert best_metadata["best_eval_episode"] == latest_metadata["best_eval_episode"] == 0
    assert latest_metadata["best_checkpoint_sha256"] == file_sha256(best)
    assert latest_metadata["convergence_controller"]["state"]["reference_scores"] == [
        1.0,
        1.0,
    ]
    records = [
        json.loads(line)
        for line in next(logs.glob("train_log_*.jsonl")).read_text("utf-8").splitlines()
    ]
    baseline = next(record for record in records if record.get("episode") == 0)
    probe = next(
        record
        for record in records
        if record.get("record_type") == "evaluation" and record.get("episode") == 1
    )
    assert baseline["baseline_source"] == "fresh"
    assert probe["eval_scope"] == "scheduler_probe"


def test_parallel_training_batches_environments_and_logs_throughput(
    tmp_path: Path,
) -> None:
    best = tmp_path / "parallel_best.pt"
    latest = tmp_path / "parallel_latest.pt"
    logs = tmp_path / "parallel_logs"
    args = parse_args(
        [
            "--episodes",
            "5",
            "--num-envs",
            "3",
            "--rollout-steps",
            "2",
            "--updates-per-collection",
            "2",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "3",
            "--eval-interval",
            "5",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "5",
            "--batch-size",
            "2",
            "--min-replay",
            "2",
            "--replay-capacity",
            "64",
            "--hidden",
            "16",
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(args)

    metadata = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert metadata["episodes_completed"] == 5
    assert metadata["episodes_started"] == 5
    assert metadata["train_args"]["num_envs"] == 3

    log_path = next(logs.glob("train_log_*.jsonl"))
    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    episodes = [record for record in records if record["record_type"] == "episode"]
    collections = [
        record for record in records if record["record_type"] == "collection"
    ]
    assert len(episodes) == 5
    assert {record["seed_index"] for record in episodes} == {1, 2, 3, 4, 5}
    assert any(record["collection_transitions"] > 1 for record in collections)
    assert any(record["collection_updates"] > 0 for record in collections)
    for key in (
        "env_steps_per_second",
        "updates_per_second",
        "sampling_seconds",
        "gpu_wait_seconds",
    ):
        assert all(key in record for record in collections)

    resume_args = parse_args(
        [
            "--episodes",
            "2",
            "--num-envs",
            "2",
            "--rollout-steps",
            "2",
            "--updates-per-collection",
            "1",
            "--width",
            "5",
            "--height",
            "5",
            "--max-steps",
            "3",
            "--eval-episodes",
            "2",
            "--checkpoint-interval",
            "1",
            "--resume-from",
            str(latest),
            "--output",
            str(best),
            "--latest-output",
            str(latest),
            "--log-dir",
            str(logs),
            "--device",
            "cpu",
            "--disable-amp",
        ]
    )
    train(resume_args)
    resumed = json.loads(latest.with_suffix(".meta.json").read_text("utf-8"))
    assert resumed["episodes_completed"] == 7
    assert resumed["episodes_started"] == 7
