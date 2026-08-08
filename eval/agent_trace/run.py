"""Agent 任务执行评测 CLI。

用法：

  # 框架自测（内建样例，验证指标计算正确）：
  python -m eval.agent_trace.run --selftest

  # 分析真实运行时工作区，产出描述性指标（工具频次 / 延迟分位）：
  python -m eval.agent_trace.run --workspace ~/.akashic/workspace

  # 在真实运行时数据上计算工具选择准确率（需 gold 标注）：
  python -m eval.agent_trace.run --workspace ~/.akashic/workspace --gold gold.json

gold.json 格式：
  [{"session_key": "telegram:123", "expected_tools": ["web_search","summarize"]}, ...]
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.agent_trace.dataset import GOLD_SAMPLE, load_gold
from eval.agent_trace.metrics import (
    TraceResult,
    is_task_success,
    latency_percentiles,
    score_results,
    tool_selection_prf,
)


def _extract_tool_names(tool_chain_json: str | None) -> set[str]:
    if not tool_chain_json:
        return set()
    try:
        groups = json.loads(tool_chain_json)
    except Exception:
        return set()
    names: set[str] = set()
    for group in groups:
        for call in group.get("calls") or []:
            name = (call.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _parse_ts(ts: str) -> float:
    """把 ISO 时间戳转成 epoch 秒（尽力解析）。"""
    if not ts:
        return 0.0
    s = ts.replace("Z", "+00:00")
    try:
        from datetime import datetime

        if "." in s:
            dt = datetime.fromisoformat(s)
        else:
            dt = datetime.fromisoformat(s + ".000000")
        return dt.timestamp()
    except Exception:
        m = re.search(r"(\d{10})", ts)
        return float(m.group(1)) if m else 0.0


def analyze_workspace(workspace: Path) -> dict[str, Any]:
    """读取真实运行时 sessions.db，产出描述性指标。"""
    from session.store import SessionStore

    db = workspace / "sessions.db"
    if not db.exists():
        raise FileNotFoundError(f"找不到 sessions.db: {db}")
    store = SessionStore(str(db))
    try:
        sessions, total = store.list_sessions_for_dashboard(
            page=1, page_size=200
        )
        tool_freq: dict[str, int] = {}
        latencies: list[float] = []
        assistant_turns = 0
        message_count = 0
        for meta in sessions:
            key = meta["key"]
            msgs, _ = store.list_messages_for_dashboard(
                session_key=key, page=1, page_size=200, sort_by="seq", sort_order="asc"
            )
            message_count += len(msgs)
            last_user_ts: float = 0.0
            for m in msgs:
                role = (m.get("role") or "").lower()
                ts = _parse_ts(str(m.get("ts") or ""))
                if role == "user":
                    last_user_ts = ts
                elif role in ("assistant", "agent"):
                    assistant_turns += 1
                    if last_user_ts and ts:
                        delta = ts - last_user_ts
                        if 0 < delta < 600:
                            latencies.append(delta)
                    names = _extract_tool_names(m.get("tool_chain"))
                    for nm in names:
                        tool_freq[nm] = tool_freq.get(nm, 0) + 1
    finally:
        store.close()

    return {
        "session_count": total,
        "message_count": message_count,
        "assistant_turn_count": assistant_turns,
        "latency": latency_percentiles(latencies),
        "tool_frequency": dict(
            sorted(tool_freq.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def _build_results_from_gold(
    workspace: Path, gold: list[dict[str, Any]]
) -> list[TraceResult]:
    from session.store import SessionStore

    db = workspace / "sessions.db"
    if not db.exists():
        raise FileNotFoundError(f"找不到 sessions.db: {db}")
    store = SessionStore(str(db))
    results: list[TraceResult] = []
    try:
        for item in gold:
            key = str(item.get("session_key") or item.get("task_id") or "")
            expected = set(item.get("expected_tools", []))
            msgs, _ = store.list_messages_for_dashboard(
                session_key=key, page=1, page_size=200
            )
            predicted: set[str] = set()
            for m in msgs:
                if (m.get("role") or "").lower() in ("assistant", "agent"):
                    predicted |= _extract_tool_names(m.get("tool_chain"))
            results.append(
                TraceResult(
                    task_id=key,
                    expected_tools=expected,
                    predicted_tools=predicted,
                    turn_seconds=0.0,
                    success=is_task_success(expected, predicted),
                )
            )
    finally:
        store.close()
    return results


def _run_selftest() -> dict[str, Any]:
    results = [
        TraceResult(
            task_id=item["task_id"],
            expected_tools=set(item["expected_tools"]),
            predicted_tools=set(item["predicted_tools"]),
            turn_seconds=float(item["turn_seconds"]),
            success=is_task_success(
                set(item["expected_tools"]), set(item["predicted_tools"])
            ),
        )
        for item in GOLD_SAMPLE
    ]
    report = score_results(results)
    report["note"] = "methodology smoke test on built-in GOLD_SAMPLE"
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Agent 任务执行评测")
    p.add_argument("--selftest", action="store_true", help="运行内建样例自测")
    p.add_argument("--workspace", type=Path, default=None, help="运行时工作区目录")
    p.add_argument("--gold", type=Path, default=None, help="gold 标注 JSON")
    p.add_argument("--output", type=Path, default=None, help="输出 JSON 路径")
    args = p.parse_args()

    if args.selftest:
        report = _run_selftest()
    elif args.workspace is not None:
        if args.gold is not None:
            gold = load_gold(args.gold)
            results = _build_results_from_gold(args.workspace, gold)
            report = score_results(results)
            report["mode"] = "workspace+gold"
        else:
            report = analyze_workspace(args.workspace)
            report["mode"] = "workspace-descriptive"
    else:
        p.error("需指定 --selftest 或 --workspace")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"saved: {args.output}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
