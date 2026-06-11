from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory2.retriever import Retriever
from memory2.store import MemoryStore2

VEC_DIM = 128
SCENARIO_SIZE = 20


@dataclass(frozen=True)
class SeedMemory:
    key: str
    memory_type: str
    summary: str
    embedding: list[float]
    source_ref: str
    extra: dict[str, object] | None = None
    happened_at: str | None = None
    emotional_weight: int = 0


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    scenario: str
    question: str
    query_vec: list[float]
    expected_key: str
    memory_types: list[str] | None = None


class _NoopEmbedder:
    async def embed(self, query: str) -> list[float]:
        raise RuntimeError("memory_quality benchmark uses direct vectors")


def _basis(index: int) -> list[float]:
    if index < 0 or index >= VEC_DIM:
        raise ValueError(f"basis index out of range: {index}")
    vec = [0.0] * VEC_DIM
    vec[index] = 1.0
    return vec


def _source_ref(message_id: str, tag: str) -> str:
    return f"{json.dumps([message_id], ensure_ascii=False)}#{tag}"


def _item_id(result: str) -> str:
    if ":" not in result:
        raise ValueError(f"unexpected upsert result: {result!r}")
    return result.split(":", 1)[1]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, pct))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scope_extra(**extra: object) -> dict[str, object]:
    payload = {"scope_channel": "telegram", "scope_chat_id": "100"}
    payload.update(extra)
    return payload


def _profile_seed(index: int) -> SeedMemory:
    cities = [
        "Hangzhou",
        "Shanghai",
        "Shenzhen",
        "Beijing",
        "Chengdu",
        "Nanjing",
        "Suzhou",
        "Wuhan",
        "Xi'an",
        "Guangzhou",
    ]
    hobbies = [
        "weekend hiking",
        "coffee tasting",
        "frontend demos",
        "city walks",
        "reading product essays",
        "cycling",
        "Japanese food",
        "board games",
        "keyboard collecting",
        "photography",
    ]
    city = cities[index % len(cities)]
    hobby = hobbies[index % len(hobbies)]
    return SeedMemory(
        key=f"profile_{index:02d}",
        memory_type="profile",
        summary=f"User profile {index:02d}: lives in {city} and enjoys {hobby}.",
        embedding=_basis(index),
        source_ref=_source_ref(f"telegram:100:p{index:02d}", "profile"),
        extra=_scope_extra(topic="profile", city=city),
    )


def _preference_seed(index: int) -> SeedMemory:
    subjects = [
        "working music",
        "meeting notes",
        "travel planning",
        "lunch choices",
        "code review style",
        "news summaries",
        "learning plans",
        "shopping comparisons",
        "fitness reminders",
        "dashboard layout",
    ]
    styles = [
        "quiet jazz",
        "bullet-point summaries",
        "routes with backup plans",
        "light Cantonese food",
        "direct risk-first feedback",
        "source-backed briefs",
        "small daily tasks",
        "price and durability tradeoffs",
        "gentle evening nudges",
        "dense table views",
    ]
    subject = subjects[index % len(subjects)]
    style = styles[index % len(styles)]
    return SeedMemory(
        key=f"preference_{index:02d}",
        memory_type="preference",
        summary=f"User preference {index:02d}: for {subject}, prefers {style}.",
        embedding=_basis(SCENARIO_SIZE + index),
        source_ref=_source_ref(f"telegram:100:pr{index:02d}", "preference"),
        extra=_scope_extra(topic="preference", subject=subject),
        emotional_weight=1 + index % 4,
    )


def _procedure_seed(index: int) -> SeedMemory:
    tasks = [
        "comparing headphones",
        "checking Python package updates",
        "planning a weekend trip",
        "debugging flaky tests",
        "summarizing a paper",
        "preparing interview answers",
        "selecting a monitor",
        "researching a framework",
        "writing resume bullets",
        "checking MCP tools",
    ]
    task = tasks[index % len(tasks)]
    return SeedMemory(
        key=f"procedure_{index:02d}",
        memory_type="procedure",
        summary=(
            f"User procedure {index:02d}: when {task}, gather two sources, "
            "compare tradeoffs, then give a concise recommendation."
        ),
        embedding=_basis(SCENARIO_SIZE * 2 + index),
        source_ref=_source_ref(f"telegram:100:proc{index:02d}", "procedure"),
        extra=_scope_extra(
            topic="procedure",
            steps=[
                "collect two sources",
                "compare tradeoffs",
                "recommend one option",
            ],
            trigger_tags={
                "scope": "tool_triggered",
                "tools": ["web_search"],
                "keywords": [task.split()[0], "compare", "recommend"],
            },
        ),
    )


def _event_seed(index: int) -> SeedMemory:
    event_names = [
        "bought a compact mechanical keyboard",
        "booked a Hangzhou hotel",
        "finished an Agent interview draft",
        "changed a Telegram bot setting",
        "tested a memory benchmark",
        "watched a Bilibili review",
        "joined a study group",
        "renewed a cloud server",
        "fixed a scheduler bug",
        "saved a travel checklist",
    ]
    event = event_names[index % len(event_names)]
    day = 1 + index
    return SeedMemory(
        key=f"event_{index:02d}",
        memory_type="event",
        summary=f"User event {index:02d}: {event} on 2026-04-{day:02d}.",
        embedding=_basis(SCENARIO_SIZE * 3 + index),
        source_ref=_source_ref(f"telegram:100:e{index:02d}", "event"),
        extra=_scope_extra(topic="event"),
        happened_at=f"2026-04-{day:02d}",
    )


def _preference_update_pair(index: int) -> tuple[SeedMemory, SeedMemory]:
    contexts = [
        "team dinner",
        "morning focus music",
        "travel hotel",
        "daily reminders",
        "meeting recap",
        "shopping advice",
        "learning schedule",
        "dashboard density",
        "exercise plan",
        "research report",
    ]
    old_values = [
        "spicy Sichuan food",
        "fast electronic music",
        "cheapest hostels",
        "late-night reminders",
        "long narrative summaries",
        "lowest price only",
        "large weekend batches",
        "large visual cards",
        "high-intensity workouts",
        "opinion-only answers",
    ]
    new_values = [
        "light Cantonese food",
        "quiet jazz",
        "clean business hotels",
        "gentle evening reminders",
        "action-first bullet summaries",
        "price, durability, and warranty tradeoffs",
        "small daily learning tasks",
        "dense table views",
        "low-impact steady plans",
        "source-backed answers",
    ]
    context = contexts[index % len(contexts)]
    old_value = old_values[index % len(old_values)]
    new_value = new_values[index % len(new_values)]
    emb = _basis(SCENARIO_SIZE * 4 + index)
    old = SeedMemory(
        key=f"preference_update_{index:02d}_old",
        memory_type="preference",
        summary=f"Old preference {index:02d}: for {context}, user preferred {old_value}.",
        embedding=emb,
        source_ref=_source_ref(f"telegram:100:u{index:02d}:old", "preference-old"),
        extra=_scope_extra(topic="preference_update", version="old"),
    )
    new = SeedMemory(
        key=f"preference_update_{index:02d}_new",
        memory_type="preference",
        summary=f"Updated preference {index:02d}: for {context}, user now prefers {new_value}.",
        embedding=emb,
        source_ref=_source_ref(f"telegram:100:u{index:02d}:new", "preference-new"),
        extra=_scope_extra(topic="preference_update", version="new"),
        emotional_weight=3,
    )
    return old, new


def _seed_memories(store: MemoryStore2) -> tuple[dict[str, str], dict[str, Any]]:
    seeds: list[SeedMemory] = []
    seeds.extend(_profile_seed(index) for index in range(SCENARIO_SIZE))
    seeds.extend(_preference_seed(index) for index in range(SCENARIO_SIZE))
    seeds.extend(_procedure_seed(index) for index in range(SCENARIO_SIZE))
    seeds.extend(_event_seed(index) for index in range(SCENARIO_SIZE))
    for index in range(SCENARIO_SIZE):
        old, new = _preference_update_pair(index)
        seeds.extend([old, new])

    ids: dict[str, str] = {}
    for seed in seeds:
        ids[seed.key] = _item_id(
            store.upsert_item(
                seed.memory_type,
                seed.summary,
                seed.embedding,
                source_ref=seed.source_ref,
                extra=seed.extra,
                happened_at=seed.happened_at,
                emotional_weight=seed.emotional_weight,
            )
        )

    duplicate_seed = _preference_seed(0)
    duplicate_result = store.upsert_item(
        duplicate_seed.memory_type,
        duplicate_seed.summary,
        duplicate_seed.embedding,
        source_ref=_source_ref("telegram:100:duplicate", "duplicate"),
        extra=duplicate_seed.extra,
    )

    replacements = 0
    for index in range(SCENARIO_SIZE):
        old_key = f"preference_update_{index:02d}_old"
        new_key = f"preference_update_{index:02d}_new"
        old_id = ids[old_key]
        new_id = ids[new_key]
        old_item = store.get_items_by_ids([old_id])[0]
        new_item = store.get_items_by_ids([new_id])[0]
        store.mark_superseded(old_id)
        replacements += store.record_replacements(
            old_items=[old_item],
            new_item=new_item,
            source_ref=_source_ref(f"telegram:100:u{index:02d}:new", "supersede"),
        )

    duplicate_item = store.get_items_by_ids([ids["preference_00"]])[0]
    return ids, {
        "duplicate_result": duplicate_result,
        "duplicate_item_id": ids["preference_00"],
        "duplicate_reinforcement": duplicate_item.get("extra_json", {}).get(
            "_reinforcement"
        ),
        "replacement_count": replacements,
    }


def _query_cases() -> list[QueryCase]:
    cases: list[QueryCase] = []
    for index in range(SCENARIO_SIZE):
        cases.append(
            QueryCase(
                query_id=f"profile_{index:02d}",
                scenario="profile",
                question=f"What profile fact number {index:02d} is known about the user?",
                query_vec=_basis(index),
                expected_key=f"profile_{index:02d}",
                memory_types=["profile", "event"],
            )
        )
    for index in range(SCENARIO_SIZE):
        cases.append(
            QueryCase(
                query_id=f"preference_{index:02d}",
                scenario="preference",
                question=f"What stable preference number {index:02d} should be remembered?",
                query_vec=_basis(SCENARIO_SIZE + index),
                expected_key=f"preference_{index:02d}",
                memory_types=["preference"],
            )
        )
    for index in range(SCENARIO_SIZE):
        cases.append(
            QueryCase(
                query_id=f"procedure_{index:02d}",
                scenario="procedure",
                question=f"What procedure number {index:02d} should the agent follow?",
                query_vec=_basis(SCENARIO_SIZE * 2 + index),
                expected_key=f"procedure_{index:02d}",
                memory_types=["procedure"],
            )
        )
    for index in range(SCENARIO_SIZE):
        cases.append(
            QueryCase(
                query_id=f"event_{index:02d}",
                scenario="event",
                question=f"What event number {index:02d} happened before?",
                query_vec=_basis(SCENARIO_SIZE * 3 + index),
                expected_key=f"event_{index:02d}",
                memory_types=["event", "profile", "preference"],
            )
        )
    for index in range(SCENARIO_SIZE):
        cases.append(
            QueryCase(
                query_id=f"preference_update_{index:02d}",
                scenario="preference_update",
                question=f"What is the latest preference update number {index:02d}?",
                query_vec=_basis(SCENARIO_SIZE * 4 + index),
                expected_key=f"preference_update_{index:02d}_new",
                memory_types=["preference"],
            )
        )
    return cases


def _run_queries(store: MemoryStore2, ids: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in _query_cases():
        started = time.perf_counter()
        hits = store.vector_search(
            query_vec=case.query_vec,
            top_k=3,
            memory_types=case.memory_types,
            score_threshold=0.0,
            include_superseded=False,
            scope_channel="telegram",
            scope_chat_id="100",
            require_scope_match=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        expected_id = ids[case.expected_key]
        top_ids = [str(item.get("id")) for item in hits]
        expected_item = store.get_items_by_ids([expected_id])[0]
        rows.append(
            {
                "query_id": case.query_id,
                "scenario": case.scenario,
                "question": case.question,
                "expected_id": expected_id,
                "expected_summary": expected_item["summary"],
                "source_ref": expected_item.get("source_ref") or "",
                "top1_id": top_ids[0] if top_ids else "",
                "top3_ids": top_ids[:3],
                "top3_summaries": [str(item.get("summary", "")) for item in hits],
                "hit_at_1": bool(top_ids and top_ids[0] == expected_id),
                "hit_at_3": expected_id in top_ids[:3],
                "latency_ms": round(latency_ms, 4),
            }
        )
    return rows


def _active_source_ref_coverage(store: MemoryStore2) -> tuple[float, int, int]:
    active_items, active_count = store.list_items_for_dashboard(
        status="active",
        page_size=200,
    )
    with_source = sum(1 for item in active_items if item.get("source_ref"))
    coverage = with_source / active_count if active_count else 0.0
    return round(coverage, 4), with_source, active_count


def _build_injection_metrics(store: MemoryStore2, queries: list[dict[str, Any]]) -> dict:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for item_id in query["top3_ids"]:
            if item_id in seen:
                continue
            seen.add(item_id)
            selected.extend(store.get_items_by_ids([item_id]))

    retriever = Retriever(
        store=store,
        embedder=_NoopEmbedder(),
        score_threshold=0.0,
        score_thresholds={
            "procedure": 0.0,
            "preference": 0.0,
            "event": 0.0,
            "profile": 0.0,
        },
        inject_max_chars=3000,
        inject_max_event_profile=6,
        inject_max_procedure_preference=8,
    )
    for item in selected:
        item.setdefault("score", 1.0)
    block, injected_ids = retriever.build_injection_block(selected)
    return {
        "injection_char_count": len(block),
        "injected_item_count": len(injected_ids),
        "injected_item_ids": injected_ids,
    }


def _scenario_counts(queries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        scenario = str(query.get("scenario") or "unknown")
        counts[scenario] = counts.get(scenario, 0) + 1
    return dict(sorted(counts.items()))


def run_benchmark(
    workspace: Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    store = MemoryStore2(workspace / "memory_quality.db", vec_dim=VEC_DIM)
    try:
        ids, seed_checks = _seed_memories(store)
        queries = _run_queries(store, ids)
        coverage, with_source, active_count = _active_source_ref_coverage(store)
        active_items, active_total = store.list_items_for_dashboard(
            status="active",
            page_size=200,
        )
        _superseded_items, superseded_total = store.list_items_for_dashboard(
            status="superseded",
            page_size=200,
        )
        injection = _build_injection_metrics(store, queries)

        replacement_rows = store.list_replacements()
        sample_old_id = ids["preference_update_00_old"]
        sample_new_id = ids["preference_update_00_new"]
        sample_old_item = store.get_item_for_dashboard(sample_old_id)
        sample_update_query = next(
            q for q in queries if q["query_id"] == "preference_update_00"
        )
        duplicate_item = store.list_items_for_dashboard(
            q="working music",
            memory_type="preference",
            status="active",
            page_size=10,
        )[0][0]

        latencies = [float(q["latency_ms"]) for q in queries]
        recall_at_1 = sum(1 for q in queries if q["hit_at_1"]) / len(queries)
        recall_at_3 = sum(1 for q in queries if q["hit_at_3"]) / len(queries)
        update_queries = [
            q for q in queries if q.get("scenario") == "preference_update"
        ]
        update_successes = 0
        for index, query in enumerate(update_queries):
            old_id = ids[f"preference_update_{index:02d}_old"]
            new_id = ids[f"preference_update_{index:02d}_new"]
            if old_id not in query["top3_ids"] and new_id in query["top3_ids"]:
                update_successes += 1
        supersede_success_rate = (
            update_successes / len(update_queries) if update_queries else 0.0
        )
        duplicate_success = bool(
            str(seed_checks["duplicate_result"]).startswith("reinforced:")
            and int(duplicate_item.get("reinforcement", 0) or 0) == 2
        )
        failure_examples = [
            {
                "query_id": q["query_id"],
                "scenario": q["scenario"],
                "question": q["question"],
                "expected_summary": q["expected_summary"],
                "top3_summaries": q["top3_summaries"],
            }
            for q in queries
            if not q["hit_at_3"]
        ]

        report = {
            "benchmark": "memory_quality",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "dataset": {
                "query_count": len(queries),
                "seeded_memory_count": len(ids),
                "active_item_count": active_total,
                "superseded_item_count": superseded_total,
                "source_ref_items": with_source,
                "scenario_counts": _scenario_counts(queries),
                "memory_types": sorted(
                    {str(item.get("memory_type")) for item in active_items}
                ),
                "vector_dim": VEC_DIM,
                "sqlite_vec_enabled": bool(getattr(store, "_vec_enabled", False)),
            },
            "scores": {
                "recall_at_1": round(recall_at_1, 4),
                "recall_at_3": round(recall_at_3, 4),
                "source_ref_coverage": coverage,
                "supersede_success_rate": round(supersede_success_rate, 4),
                "duplicate_reinforcement_rate": 1.0 if duplicate_success else 0.0,
                "p50_latency_ms": round(_percentile(latencies, 0.50), 4),
                "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
                "max_latency_ms": round(max(latencies) if latencies else 0.0, 4),
                "injection_char_count": injection["injection_char_count"],
                "injected_item_count": injection["injected_item_count"],
            },
            "checks": {
                "source_ref": {
                    "active_items": active_count,
                    "with_source_ref": with_source,
                },
                "duplicate_reinforcement": {
                    "result": seed_checks["duplicate_result"],
                    "item_id": seed_checks["duplicate_item_id"],
                    "reinforcement": duplicate_item.get("reinforcement"),
                },
                "replacement": {
                    "sample_old_id": sample_old_id,
                    "sample_new_id": sample_new_id,
                    "sample_old_status": (
                        sample_old_item.get("status") if sample_old_item else ""
                    ),
                    "superseded_count": superseded_total,
                    "replacement_count": len(replacement_rows),
                    "sample_update_query_top_ids": sample_update_query["top3_ids"],
                },
                "injection": injection,
            },
            "queries": queries,
            "failure_examples": failure_examples,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report
    finally:
        store.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline memory quality benchmark."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("memory_quality_workspace"),
        help="Temporary benchmark workspace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = args.output
    if output is None:
        results_dir = Path("memory_quality_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output = results_dir / "memory_quality_latest.json"
    report = run_benchmark(args.workspace, output_path=output)
    print(json.dumps(report["scores"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
