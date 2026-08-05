from __future__ import annotations

import json
from pathlib import Path

import pytest

import analyze_training


def _write_jsonl(path: Path, rows: list[object], *, bad_line: bool = False) -> None:
    lines = [json.dumps(row) for row in rows]
    if bad_line:
        lines.insert(1, "{not-json")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(path: Path, **kwargs: object) -> dict:
    records, stats = analyze_training.load_jsonl([path])
    return analyze_training.build_report(records, stats, files=[path], **kwargs)


def test_report_tolerates_bad_and_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    _write_jsonl(
        path,
        [
            {"episode": 1, "score": 0, "epsilon": 1.0, "event": "hit_wall"},
            {"episode": 2, "score": 5, "epsilon": 0.5, "avg_loss": 2.0},
            {"episode": 3, "epsilon": 0.1, "avg_loss": "bad"},
            {
                "episode": 4,
                "score": 10,
                "epsilon": 0.1,
                "avg_loss": 1.0,
                "eval_avg_reward": 8.0,
                "eval_avg_score": 3.0,
                "snake_length": 13,
                "info": {"event": "hit_self"},
            },
        ],
        bad_line=True,
    )

    report = _report(path, plateau_window=2)

    assert report["rows"]["valid_rows"] == 4
    assert report["rows"]["bad_rows"] == 1
    assert report["rows"]["blank_rows"] == 1
    assert report["episode_range"] == {
        "first": 1,
        "last": 4,
        "min": 1,
        "max": 4,
        "count": 4,
    }
    assert report["score"]["count"] == 3
    assert report["score"]["mean"] == pytest.approx(5.0)
    assert report["score"]["recent_mean"]["100"] == pytest.approx(5.0)
    assert report["loss"]["count"] == 2
    assert report["loss"]["last"] == pytest.approx(1.0)
    assert report["termination_events"]["counts"] == {"hit_wall": 1, "hit_self": 1}
    assert report["score_buckets"]["buckets"]["0"]["count"] == 1
    assert report["score_buckets"]["buckets"]["5-9"]["count"] == 1
    assert report["score_buckets"]["buckets"]["10-19"]["count"] == 1
    assert report["snake_length_buckets"]["buckets"]["10-19"]["count"] == 1


def test_evaluation_and_epsilon_floor_positions(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write_jsonl(
        path,
        [
            {"episode": 10, "score": 1, "epsilon": 1.0, "eval_avg_reward": 4.0},
            {"episode": 20, "score": 2, "epsilon": 0.1, "eval_avg_reward": 9.0},
            {"episode": 30, "score": 3, "epsilon": 0.01, "eval_avg_reward": 7.0},
            {
                "episode": 40,
                "score": 4,
                "epsilon": 0.01,
                "eval_reward_mean": 12.0,
                "eval_score_mean": 5.0,
                "eval_steps_mean": 20.0,
            },
        ],
    )

    report = _report(path, plateau_window=2)

    assert report["evaluation"]["count"] == 4
    assert report["evaluation"]["best"]["episode"] == 40
    assert report["evaluation"]["best"]["avg_score"] == pytest.approx(5.0)
    assert report["evaluation"]["last"]["episode"] == 40
    assert report["epsilon"]["first"] == pytest.approx(1.0)
    assert report["epsilon"]["last"] == pytest.approx(0.01)
    assert report["epsilon"]["observed_floor"] == pytest.approx(0.01)
    assert report["epsilon"]["first_floor_position"]["episode"] == 30
    assert report["epsilon"]["first_floor_position"]["record_index"] == 3


def test_zero_eval_reward_ranks_above_negative_reward(tmp_path: Path) -> None:
    path = tmp_path / "zero-eval.jsonl"
    _write_jsonl(
        path,
        [
            {"episode": 1, "score": 1, "eval_avg_reward": 0.0},
            {"episode": 2, "score": 1, "eval_avg_reward": -1.0},
        ],
    )

    report = _report(path, plateau_window=1)

    assert report["evaluation"]["best"]["episode"] == 1


def test_plateau_is_explicitly_a_non_statistical_heuristic() -> None:
    flat = analyze_training.diagnose_plateau(
        [10.0] * 20,
        window=10,
        min_absolute_improvement=0.5,
        min_relative_improvement=0.0,
    )
    improving = analyze_training.diagnose_plateau(
        [10.0] * 10 + [11.0] * 10,
        window=10,
        min_absolute_improvement=0.5,
        min_relative_improvement=0.0,
    )
    insufficient = analyze_training.diagnose_plateau([1.0, 2.0], window=2)

    assert flat["status"] == "plateau"
    assert flat["is_plateau"] is True
    assert flat["diagnostic_only"] is True
    assert "not a statistical proof" in flat["note"]
    assert improving["status"] == "improving"
    assert improving["is_plateau"] is False
    assert insufficient["status"] == "insufficient_data"
    assert insufficient["is_plateau"] is None


def test_glob_cli_json_and_human_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_jsonl(first, [{"episode": 1, "score": 1, "epsilon": 1.0}])
    _write_jsonl(second, [{"episode": 2, "score": 2, "epsilon": 0.5}])

    assert (
        analyze_training.main(
            [str(tmp_path / "*.jsonl"), "--json", "--plateau-window", "1"]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["input"]["files"]) == 2
    assert report["rows"]["valid_rows"] == 2
    assert report["plateau"]["status"] == "improving"

    assert analyze_training.main([str(first), "--plateau-window", "1"]) == 0
    output = capsys.readouterr().out
    assert "Score mean:" in output
    assert "Score buckets:" in output
    assert "Snake-length buckets:" in output
    assert "diagnostic only" in output.lower()


def test_no_matching_files_exits_cleanly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No readable JSONL files"):
        analyze_training.main([str(tmp_path / "missing-*.jsonl")])
