"""Agent 任务执行评测指标。

覆盖三类简历可用的量化指标：

- 工具选择质量：把一次任务实际调用的工具集合与「期望工具集合」比较，
  计算 precision / recall / f1（tool_selection_accuracy 即其中的 recall 视角）。
- 任务完成率：以「期望工具是否被覆盖 + 是否出现关键失败」作为完成判据。
- 端到端延迟：基于用户消息与助手回复的时间戳差，计算 P50 / P95 / max。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceResult:
    task_id: str
    expected_tools: set[str]
    predicted_tools: set[str]
    turn_seconds: float
    success: bool = True
    error: str | None = None


def tool_selection_prf(
    expected: set[str], predicted: set[str]
) -> dict[str, float]:
    """工具选择的 precision / recall / f1。

    precision = 预测命中的期望工具 / 预测工具总数
    recall    = 预测命中的期望工具 / 期望工具总数（即 tool_selection_accuracy）
    """
    if not expected and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = expected & predicted
    num_same = len(common)
    precision = num_same / len(predicted)
    recall = num_same / len(expected)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def is_task_success(
    expected: set[str],
    predicted: set[str],
    *,
    recall_threshold: float = 0.5,
    allow_extra: bool = True,
) -> bool:
    """任务是否完成。

    规则：期望工具被覆盖到 recall_threshold 比例，即视为完成；
    默认允许预测工具多于期望（allow_extra=True）。
    """
    if not expected:
        return True
    if not predicted:
        return False
    recall = len(expected & predicted) / len(expected)
    return recall >= recall_threshold


def latency_percentiles(turn_seconds: list[float]) -> dict[str, float]:
    """延迟分位（秒）。"""
    if not turn_seconds:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(turn_seconds)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo, hi = int(pos // 1), min(int(pos // 1) + 1, n - 1)
        frac = pos - int(pos // 1)
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "p50": round(_pct(0.50), 4),
        "p95": round(_pct(0.95), 4),
        "max": round(ordered[-1], 4),
    }


def score_results(results: list[TraceResult]) -> dict[str, Any]:
    """聚合一组 TraceResult 为总报告。"""
    valid = [r for r in results if not r.error]
    n = len(valid)
    if n == 0:
        return {
            "overall": {
                "tool_selection_precision": 0.0,
                "tool_selection_recall": 0.0,
                "tool_selection_f1": 0.0,
                "task_success_rate": 0.0,
                "n": 0,
            },
            "latency": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "tool_frequency": {},
            "errors": len(results),
        }

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    successes = 0
    latency: list[float] = []
    freq: Counter[str] = Counter()

    for r in valid:
        prf = tool_selection_prf(r.expected_tools, r.predicted_tools)
        precisions.append(prf["precision"])
        recalls.append(prf["recall"])
        f1s.append(prf["f1"])
        if is_task_success(r.expected_tools, r.predicted_tools):
            successes += 1
        latency.append(r.turn_seconds)
        freq.update(r.predicted_tools)

    return {
        "overall": {
            "tool_selection_precision": round(sum(precisions) / n, 4),
            "tool_selection_recall": round(sum(recalls) / n, 4),
            "tool_selection_f1": round(sum(f1s) / n, 4),
            "task_success_rate": round(successes / n, 4),
            "n": n,
        },
        "latency": latency_percentiles(latency),
        "tool_frequency": dict(freq.most_common()),
        "errors": len(results) - n,
    }
