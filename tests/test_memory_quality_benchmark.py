from __future__ import annotations

import json
from pathlib import Path

from eval.memory_quality import benchmark
from eval.memory_quality.benchmark import run_benchmark


def test_memory_quality_benchmark_reports_reproducible_scores(tmp_path: Path) -> None:
    output_path = tmp_path / "memory_quality.json"

    report = run_benchmark(tmp_path / "workspace", output_path=output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["scores"] == report["scores"]

    scores = report["scores"]
    assert scores["recall_at_1"] == 1.0
    assert scores["recall_at_3"] == 1.0
    assert scores["source_ref_coverage"] == 1.0
    assert scores["supersede_success_rate"] == 1.0
    assert scores["duplicate_reinforcement_rate"] == 1.0
    assert scores["injected_item_count"] >= 2
    assert scores["injection_char_count"] > 0
    assert scores["p50_latency_ms"] >= 0
    assert scores["p95_latency_ms"] >= scores["p50_latency_ms"]

    assert report["dataset"]["query_count"] == 100
    assert report["dataset"]["seeded_memory_count"] == 120
    assert report["dataset"]["active_item_count"] == 100
    assert report["dataset"]["superseded_item_count"] == 20
    assert report["dataset"]["scenario_counts"] == {
        "event": 20,
        "preference": 20,
        "preference_update": 20,
        "procedure": 20,
        "profile": 20,
    }
    assert report["failure_examples"] == []


def test_memory_quality_benchmark_keeps_evidence_for_each_query(
    tmp_path: Path,
) -> None:
    report = run_benchmark(tmp_path / "workspace")

    for query in report["queries"]:
        assert query["expected_id"] in query["top3_ids"]
        assert query["source_ref"]
        assert query["latency_ms"] >= 0

    replacement = report["checks"]["replacement"]
    assert replacement["sample_old_status"] == "superseded"
    assert replacement["superseded_count"] == 20
    assert replacement["replacement_count"] == 20
    assert replacement["sample_old_id"] not in replacement["sample_update_query_top_ids"]
    assert replacement["sample_new_id"] in replacement["sample_update_query_top_ids"]

    duplicate = report["checks"]["duplicate_reinforcement"]
    assert duplicate["result"].startswith("reinforced:")
    assert duplicate["reinforcement"] == 2


def test_memory_quality_cli_defaults_use_repo_level_runtime_dirs() -> None:
    args = benchmark._build_parser().parse_args([])

    assert args.workspace == Path("memory_quality_workspace")
