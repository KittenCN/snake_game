"""Read-only diagnostics for Snake DQN JSONL training logs.

The plateau result produced here is a configurable heuristic.  It is useful for
triage, but it is not a statistical proof that learning has stopped.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WINDOWS = (100, 1_000, 5_000)
PLATEAU_NOTE = (
    "Heuristic diagnostic only; this comparison is not a statistical proof "
    "that training has plateaued."
)


@dataclass(frozen=True)
class LogRecord:
    data: Mapping[str, Any]
    source: str
    line: int


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _display_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else value


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def expand_inputs(patterns: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Expand files and glob expressions, preserving deterministic order."""
    files: list[Path] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        direct = Path(pattern).expanduser()
        matches = (
            [str(direct)] if direct.is_file() else glob.glob(pattern, recursive=True)
        )
        matched_files = sorted(Path(item) for item in matches if Path(item).is_file())
        if not matched_files:
            unmatched.append(pattern)
            continue
        for path in matched_files:
            resolved = str(path.resolve())
            if resolved not in seen:
                files.append(path.resolve())
                seen.add(resolved)
    return files, unmatched


def load_jsonl(paths: Sequence[Path]) -> tuple[list[LogRecord], dict[str, Any]]:
    """Load JSON objects from JSONL while recording, rather than raising on, bad rows."""
    records: list[LogRecord] = []
    stats: dict[str, Any] = {
        "valid_rows": 0,
        "bad_rows": 0,
        "blank_rows": 0,
        "read_errors": [],
        "bad_row_samples": [],
        "per_file": {},
    }
    for path in paths:
        per_file = {"valid_rows": 0, "bad_rows": 0, "blank_rows": 0}
        stats["per_file"][str(path)] = per_file
        try:
            stream = path.open("r", encoding="utf-8-sig")
        except OSError as exc:
            stats["read_errors"].append({"file": str(path), "error": str(exc)})
            continue
        with stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    stats["blank_rows"] += 1
                    per_file["blank_rows"] += 1
                    continue
                try:
                    value = json.loads(raw_line)
                    if not isinstance(value, dict):
                        raise ValueError("JSON value is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    stats["bad_rows"] += 1
                    per_file["bad_rows"] += 1
                    if len(stats["bad_row_samples"]) < 20:
                        stats["bad_row_samples"].append(
                            {"file": str(path), "line": line_number, "error": str(exc)}
                        )
                    continue
                records.append(LogRecord(value, str(path), line_number))
                stats["valid_rows"] += 1
                per_file["valid_rows"] += 1
    return records, stats


def _values(records: Sequence[LogRecord], key: str) -> list[float]:
    return [
        value
        for record in records
        if (value := _number(record.data.get(key))) is not None
    ]


def _series_summary(
    values: Sequence[float], windows: Sequence[int] = DEFAULT_WINDOWS
) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "last": values[-1] if values else None,
        "recent_mean": {
            str(window): _mean(values[-window:]) if values else None
            for window in windows
        },
    }


def _evaluation_entry(record: LogRecord) -> dict[str, Any]:
    decision = record.data.get("convergence_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    paired = decision.get("paired_comparison")
    paired = paired if isinstance(paired, Mapping) else {}
    statistical_state = record.data.get("eval_statistical_state")
    if not isinstance(statistical_state, str):
        statistical_state = decision.get("statistical_state")
    patience_deferred = record.data.get("eval_patience_deferred")
    if not isinstance(patience_deferred, bool):
        patience_deferred = decision.get("patience_deferred")
    adjusted_low = _number(paired.get("adjusted_ci_low"))
    adjusted_high = _number(paired.get("adjusted_ci_high"))
    if adjusted_low is None:
        adjusted_low = _number(paired.get("ci95_low"))
    if adjusted_high is None:
        adjusted_high = _number(paired.get("ci95_high"))
    return {
        "episode": _display_number(_number(record.data.get("episode"))),
        "avg_reward": _first_number(record, ("eval_reward_mean", "eval_avg_reward")),
        "avg_score": _first_number(record, ("eval_score_mean", "eval_avg_score")),
        "avg_steps": _first_number(record, ("eval_steps_mean", "eval_avg_steps")),
        "decision": decision.get("decision"),
        "paired_mean_delta": _number(paired.get("mean_delta")),
        "paired_ci95_low": _number(paired.get("ci95_low")),
        "paired_ci95_high": _number(paired.get("ci95_high")),
        "paired_adjusted_ci_low": adjusted_low,
        "paired_adjusted_ci_high": adjusted_high,
        "paired_promotion_eligible": decision.get("paired_promotion_eligible"),
        "clear_regression": decision.get("clear_regression"),
        "statistical_state": statistical_state,
        "patience_deferred": patience_deferred,
        "eval_episodes_actual": _display_number(
            _number(record.data.get("eval_episodes_actual"))
        ),
        "eval_episodes_planned": _display_number(
            _number(record.data.get("eval_episodes_planned"))
        ),
        "eval_episodes_max": _display_number(
            _number(record.data.get("eval_episodes_max"))
        ),
        "expansion_stage": _display_number(
            _number(record.data.get("eval_expansion_stage"))
        ),
        "planned_looks": _display_number(
            _number(record.data.get("eval_planned_looks"))
        ),
        "statistical_method": record.data.get("eval_statistical_method"),
        "source": record.source,
        "line": record.line,
    }


def _evaluation_summary(records: Sequence[LogRecord]) -> dict[str, Any]:
    evaluations = [
        record
        for record in records
        if _first_number(record, ("eval_reward_mean", "eval_avg_reward")) is not None
        or _first_number(record, ("eval_score_mean", "eval_avg_score")) is not None
    ]
    if not evaluations:
        return {"count": 0, "best": None, "last": None, "best_by": None}
    paired_sources = {
        record.source
        for record in records
        if record.data.get("record_type") == "run_start"
        and isinstance(record.data.get("args"), Mapping)
        and record.data["args"].get("require_paired_promotion") is True
    }
    score_rows = [
        r
        for r in evaluations
        if _first_number(r, ("eval_score_mean", "eval_avg_score")) is not None
    ]
    if score_rows:
        eligible_score_rows = [
            row
            for row in score_rows
            if row.source not in paired_sources
            or row.data.get("evaluation_kind") == "warm_start_baseline"
            or (
                isinstance(row.data.get("convergence_decision"), Mapping)
                and row.data["convergence_decision"].get(
                    "paired_promotion_eligible"
                )
                is True
            )
        ]
        if paired_sources and not eligible_score_rows:
            best = None
            best_by = "paired_no_promoted_evaluation"
        else:
            score_rows = eligible_score_rows or score_rows
            best = max(
                score_rows,
                key=lambda r: (
                    value
                    if (
                        value := _first_number(
                            r, ("eval_score_mean", "eval_avg_score")
                        )
                    )
                    is not None
                    else -math.inf
                ),
            )
            best_by = (
                "paired_promoted_avg_score"
                if best.source in paired_sources
                else "avg_score"
            )
    else:
        best = max(
            evaluations,
            key=lambda r: (
                value
                if (value := _first_number(r, ("eval_reward_mean", "eval_avg_reward")))
                is not None
                else -math.inf
            ),
        )
        best_by = "avg_reward"
    entries = [_evaluation_entry(record) for record in evaluations]
    state_counts = Counter(
        entry["statistical_state"]
        for entry in entries
        if isinstance(entry["statistical_state"], str)
    )
    actual_episode_counts = [
        int(value)
        for entry in entries
        if (value := entry["eval_episodes_actual"]) is not None
    ]
    expanded_count = sum(
        1
        for entry in entries
        if isinstance(entry["expansion_stage"], (int, float))
        and entry["expansion_stage"] > 1
    )
    return {
        "count": len(evaluations),
        "best": _evaluation_entry(best) if best is not None else None,
        "last": _evaluation_entry(evaluations[-1]),
        "best_by": best_by,
        "statistical_state_counts": dict(sorted(state_counts.items())),
        "expanded_count": expanded_count,
        "total_eval_episodes": sum(actual_episode_counts),
        "max_eval_episodes_used": max(actual_episode_counts, default=None),
    }


def _epsilon_summary(records: Sequence[LogRecord]) -> dict[str, Any]:
    points: list[tuple[int, LogRecord, float]] = []
    for record_index, record in enumerate(records, start=1):
        value = _number(record.data.get("epsilon"))
        if value is not None:
            points.append((record_index, record, value))
    if not points:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "observed_floor": None,
            "first_floor_position": None,
        }
    floor = min(point[2] for point in points)
    tolerance = max(1e-12, abs(floor) * 1e-9)
    floor_point = next(point for point in points if point[2] <= floor + tolerance)
    record_index, record, _ = floor_point
    return {
        "count": len(points),
        "first": points[0][2],
        "last": points[-1][2],
        "observed_floor": floor,
        "first_floor_position": {
            "record_index": record_index,
            "episode": _display_number(_number(record.data.get("episode"))),
            "source": record.source,
            "line": record.line,
        },
    }


def _event(record: LogRecord) -> str | None:
    for key in ("termination_event", "terminal_event", "done_reason", "event"):
        value = record.data.get(key)
        if isinstance(value, str) and value:
            return value
    info = record.data.get("info")
    if isinstance(info, Mapping):
        value = info.get("event")
        if isinstance(value, str) and value:
            return value
    return None


def _termination_summary(
    records: Sequence[LogRecord], *, recent_window: int = 100
) -> dict[str, Any]:
    """Summarize episode endings and expose horizon saturation explicitly."""
    event_rows = [
        (record, event)
        for record in records
        if (event := _event(record)) is not None
    ]

    def summarize(rows: Sequence[tuple[LogRecord, str]]) -> dict[str, Any]:
        counts = Counter(event for _, event in rows)
        total = len(rows)
        truncation_count = counts.get("time_limit", 0) + counts.get("truncated", 0)
        return {
            "count": total,
            "counts": dict(counts),
            "rates": {
                event: count / total for event, count in sorted(counts.items())
            }
            if total
            else {},
            "truncation_count": truncation_count,
            "truncation_rate": truncation_count / total if total else None,
            "idle_timeout_rate": counts.get("idle_timeout", 0) / total
            if total
            else None,
        }

    overall = summarize(event_rows)
    recent = summarize(event_rows[-recent_window:])
    recent_rate = recent["truncation_rate"]
    if recent_rate is None:
        horizon_status = "unavailable"
    elif recent_rate >= 0.5:
        horizon_status = "severely_saturated"
    elif recent_rate >= 0.1:
        horizon_status = "saturated"
    else:
        horizon_status = "not_saturated"
    return {
        "available": bool(event_rows),
        **overall,
        "recent_window": recent_window,
        "recent": recent,
        "horizon_status": horizon_status,
    }


def _bucket_name(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 5:
        return "1-4"
    if value < 10:
        return "5-9"
    if value < 20:
        return "10-19"
    if value < 40:
        return "20-39"
    return "40+"


def _first_number(record: LogRecord, keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _number(record.data.get(key))
        if value is not None:
            return value
    return None


def _bucket_summary(
    records: Sequence[LogRecord], group_keys: Sequence[str]
) -> dict[str, Any]:
    grouped: dict[str, list[LogRecord]] = {}
    for record in records:
        value = _first_number(record, group_keys)
        if value is not None:
            grouped.setdefault(_bucket_name(value), []).append(record)
    order = ("0", "1-4", "5-9", "10-19", "20-39", "40+")
    result: dict[str, Any] = {}
    for name in order:
        rows = grouped.get(name)
        if not rows:
            continue
        result[name] = {
            "count": len(rows),
            "mean_score": _mean(_values(rows, "score")),
            "mean_reward": _mean(_values(rows, "reward")),
            "mean_steps": _mean(_values(rows, "steps")),
            "mean_loss": _mean(_values(rows, "avg_loss")),
        }
    return {"count": sum(len(rows) for rows in grouped.values()), "buckets": result}


def diagnose_plateau(
    scores: Sequence[float],
    *,
    window: int = 1_000,
    min_absolute_improvement: float = 0.25,
    min_relative_improvement: float = 0.02,
) -> dict[str, Any]:
    if window <= 0:
        raise ValueError("plateau window must be positive")
    base = {
        "diagnostic_only": True,
        "note": PLATEAU_NOTE,
        "window": window,
        "min_absolute_improvement": min_absolute_improvement,
        "min_relative_improvement": min_relative_improvement,
    }
    if len(scores) < 2 * window:
        return {
            **base,
            "status": "insufficient_data",
            "is_plateau": None,
            "required_scores": 2 * window,
            "available_scores": len(scores),
            "previous_mean": None,
            "recent_mean": None,
            "improvement": None,
            "required_improvement": None,
        }
    previous_mean = _mean(scores[-2 * window : -window])
    recent_mean = _mean(scores[-window:])
    assert previous_mean is not None and recent_mean is not None
    improvement = recent_mean - previous_mean
    required = max(
        min_absolute_improvement, abs(previous_mean) * min_relative_improvement
    )
    is_plateau = improvement < required
    return {
        **base,
        "status": "plateau" if is_plateau else "improving",
        "is_plateau": is_plateau,
        "required_scores": 2 * window,
        "available_scores": len(scores),
        "previous_mean": previous_mean,
        "recent_mean": recent_mean,
        "improvement": improvement,
        "required_improvement": required,
    }


def build_report(
    records: Sequence[LogRecord],
    load_stats: Mapping[str, Any],
    *,
    files: Sequence[Path] = (),
    patterns: Sequence[str] = (),
    unmatched_patterns: Sequence[str] = (),
    plateau_window: int = 1_000,
    plateau_min_absolute_improvement: float = 0.25,
    plateau_min_relative_improvement: float = 0.02,
) -> dict[str, Any]:
    episodes = _values(records, "episode")
    scores = _values(records, "score")
    report = {
        "input": {
            "patterns": list(patterns),
            "files": [str(path) for path in files],
            "unmatched_patterns": list(unmatched_patterns),
        },
        "rows": dict(load_stats),
        "episode_range": {
            "first": _display_number(episodes[0]) if episodes else None,
            "last": _display_number(episodes[-1]) if episodes else None,
            "min": _display_number(min(episodes)) if episodes else None,
            "max": _display_number(max(episodes)) if episodes else None,
            "count": len(episodes),
        },
        "score": _series_summary(scores),
        "evaluation": _evaluation_summary(records),
        "epsilon": _epsilon_summary(records),
        "loss": _series_summary(_values(records, "avg_loss")),
        "td_loss": _series_summary(_values(records, "avg_td_loss")),
        "anchor_loss": _series_summary(_values(records, "avg_anchor_loss")),
        "imitation_loss": _series_summary(_values(records, "avg_imitation_loss")),
        "demonstration_batch_fraction": _series_summary(
            _values(records, "avg_demonstration_batch_fraction")
        ),
        "elite_demonstration_batch_fraction": _series_summary(
            _values(records, "avg_elite_demonstration_batch_fraction")
        ),
        "demonstration_replay_size": _series_summary(
            _values(records, "demonstration_replay_size")
        ),
        "demonstration_replay_elite_count": _series_summary(
            _values(records, "demonstration_replay_elite_count")
        ),
        "termination_events": _termination_summary(records),
        "score_buckets": _bucket_summary(records, ("score",)),
        "snake_length_buckets": _bucket_summary(
            records, ("snake_length", "snake_len", "final_snake_length")
        ),
        "plateau": diagnose_plateau(
            scores,
            window=plateau_window,
            min_absolute_improvement=plateau_min_absolute_improvement,
            min_relative_improvement=plateau_min_relative_improvement,
        ),
    }
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_human(report: Mapping[str, Any]) -> str:
    rows = report["rows"]
    episode_range = report["episode_range"]
    score = report["score"]
    evaluation = report["evaluation"]
    epsilon = report["epsilon"]
    loss = report["loss"]
    plateau = report["plateau"]
    recent_score = score["recent_mean"]
    recent_loss = loss["recent_mean"]
    lines = [
        f"Files: {len(report['input']['files'])} | valid rows: {rows['valid_rows']} | "
        f"bad: {rows['bad_rows']} | blank: {rows['blank_rows']}",
        f"Episodes: {_fmt(episode_range['first'])} -> {_fmt(episode_range['last'])} "
        f"(min={_fmt(episode_range['min'])}, max={_fmt(episode_range['max'])})",
        "Score mean: "
        f"all={_fmt(score['mean'])}, last100={_fmt(recent_score['100'])}, "
        f"last1000={_fmt(recent_score['1000'])}, last5000={_fmt(recent_score['5000'])}",
    ]
    if evaluation["count"]:
        best, last = evaluation["best"], evaluation["last"]
        if best is None:
            lines.append(
                "Evaluation: "
                f"count={evaluation['count']}, no promoted evaluation in these logs; "
                f"last ep={_fmt(last['episode'])} reward={_fmt(last['avg_reward'])} "
                f"score={_fmt(last['avg_score'])}"
            )
        else:
            lines.append(
                "Evaluation: "
                f"count={evaluation['count']}, best ep={_fmt(best['episode'])} "
                f"reward={_fmt(best['avg_reward'])} score={_fmt(best['avg_score'])}; "
                f"last ep={_fmt(last['episode'])} reward={_fmt(last['avg_reward'])} "
                f"score={_fmt(last['avg_score'])}"
            )
        if last["paired_mean_delta"] is not None:
            lines.append(
                "Paired evaluation: "
                f"mean delta={_fmt(last['paired_mean_delta'])}, "
                f"adjusted CI=[{_fmt(last['paired_adjusted_ci_low'])}, "
                f"{_fmt(last['paired_adjusted_ci_high'])}], "
                f"state={last['statistical_state']}, "
                f"patience deferred={last['patience_deferred']}, "
                f"promotion={last['paired_promotion_eligible']}, "
                f"clear regression={last['clear_regression']}, "
                f"decision={last['decision']}"
            )
        if evaluation["total_eval_episodes"]:
            lines.append(
                "Adaptive evaluation: "
                f"episodes={evaluation['total_eval_episodes']}, "
                f"max used={_fmt(evaluation['max_eval_episodes_used'])}, "
                f"expanded={evaluation['expanded_count']}, "
                f"states={evaluation['statistical_state_counts']}"
            )
    else:
        lines.append("Evaluation: no evaluation fields found")
    termination = report["termination_events"]
    if termination["available"]:
        recent = termination["recent"]
        lines.append(
            "Termination: "
            f"counts={termination['counts']}, "
            f"recent{termination['recent_window']} truncation_rate="
            f"{_fmt(recent['truncation_rate'])}, "
            f"idle_timeout_rate={_fmt(recent['idle_timeout_rate'])}, "
            f"horizon={termination['horizon_status']}"
        )
    floor_position = epsilon["first_floor_position"]
    lines.append(
        "Epsilon: "
        f"{_fmt(epsilon['first'])} -> {_fmt(epsilon['last'])}; "
        f"observed floor={_fmt(epsilon['observed_floor'])} at episode="
        f"{_fmt(floor_position['episode'] if floor_position else None)}"
    )
    lines.append(
        "Loss: "
        f"last={_fmt(loss['last'])}, last100={_fmt(recent_loss['100'])}, "
        f"last1000={_fmt(recent_loss['1000'])}, last5000={_fmt(recent_loss['5000'])}"
    )
    imitation = report["imitation_loss"]
    demo_fraction = report["demonstration_batch_fraction"]
    demo_size = report["demonstration_replay_size"]
    lines.append(
        "Demonstrations: "
        f"imitation_loss={_fmt(imitation['last'])}, "
        f"batch_fraction={_fmt(demo_fraction['last'])}, "
        f"replay_size={_fmt(demo_size['last'])}, "
        f"elite={_fmt(report['demonstration_replay_elite_count']['last'])}"
    )
    event_counts = report["termination_events"]["counts"]
    lines.append(
        "Termination events: "
        + (
            ", ".join(f"{key}={value}" for key, value in sorted(event_counts.items()))
            or "not logged"
        )
    )
    score_buckets = report["score_buckets"]["buckets"]
    snake_buckets = report["snake_length_buckets"]["buckets"]
    lines.append(
        "Score buckets: "
        + (
            ", ".join(f"{key}={value['count']}" for key, value in score_buckets.items())
            or "not logged"
        )
    )
    lines.append(
        "Snake-length buckets: "
        + (
            ", ".join(f"{key}={value['count']}" for key, value in snake_buckets.items())
            or "not logged"
        )
    )
    if plateau["status"] == "insufficient_data":
        lines.append(
            f"Plateau diagnostic: insufficient data ({plateau['available_scores']}/"
            f"{plateau['required_scores']} scores)"
        )
    else:
        lines.append(
            f"Plateau diagnostic: {plateau['status']} | previous={_fmt(plateau['previous_mean'])} "
            f"recent={_fmt(plateau['recent_mean'])} improvement={_fmt(plateau['improvement'])} "
            f"required={_fmt(plateau['required_improvement'])}"
        )
    lines.append(f"Note: {plateau['note']}")
    if report["input"]["unmatched_patterns"]:
        lines.append(
            "Unmatched inputs: " + ", ".join(report["input"]["unmatched_patterns"])
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Snake DQN JSONL training logs"
    )
    parser.add_argument(
        "inputs", nargs="+", help="One or more JSONL paths or glob expressions"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the complete report as JSON"
    )
    parser.add_argument("--plateau-window", type=int, default=1_000)
    parser.add_argument("--plateau-min-absolute-improvement", type=float, default=0.25)
    parser.add_argument("--plateau-min-relative-improvement", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plateau_window <= 0:
        raise SystemExit("--plateau-window must be positive")
    if args.plateau_min_absolute_improvement < 0:
        raise SystemExit("--plateau-min-absolute-improvement must be non-negative")
    if args.plateau_min_relative_improvement < 0:
        raise SystemExit("--plateau-min-relative-improvement must be non-negative")
    files, unmatched = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No readable JSONL files matched the supplied inputs")
    records, load_stats = load_jsonl(files)
    report = build_report(
        records,
        load_stats,
        files=files,
        patterns=args.inputs,
        unmatched_patterns=unmatched,
        plateau_window=args.plateau_window,
        plateau_min_absolute_improvement=args.plateau_min_absolute_improvement,
        plateau_min_relative_improvement=args.plateau_min_relative_improvement,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(format_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
