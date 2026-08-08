"""Agent 任务执行评测数据集。

提供两类数据：

1. ``GOLD_SAMPLE``：内建样例任务集，用于验证评测框架本身能否跑通并产出
   合理的工具选择 / 完成率 / 延迟指标（methodology smoke test）。
2. ``load_gold(path)``：从 JSON 加载用户自己的 gold 标注，格式同 ``GOLD_SAMPLE``，
   用于针对真实运行时数据计算生产级指标。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 每条：task_id, expected_tools(期望工具集合), predicted_tools(实际调用),
# turn_seconds(端到端耗时，秒)。predicted 故意保留少量不完美，
# 以体现评测框架能区分 good/bad 执行。
GOLD_SAMPLE: list[dict[str, Any]] = [
    {
        "task_id": "t01_search_news",
        "expected_tools": {"web_search", "summarize"},
        "predicted_tools": {"web_search", "summarize"},
        "turn_seconds": 3.2,
    },
    {
        "task_id": "t02_read_file",
        "expected_tools": {"read_file"},
        "predicted_tools": {"read_file"},
        "turn_seconds": 0.9,
    },
    {
        "task_id": "t03_edit_and_test",
        "expected_tools": {"edit_file", "shell"},
        "predicted_tools": {"edit_file", "shell"},
        "turn_seconds": 6.4,
    },
    {
        "task_id": "t04_weekly_review",
        "expected_tools": {"memory_search", "summarize"},
        "predicted_tools": {"memory_search", "summarize", "write_file"},
        "turn_seconds": 4.1,
    },
    {
        "task_id": "t05_compare_products",
        "expected_tools": {"web_search", "compare"},
        "predicted_tools": {"web_search", "compare"},
        "turn_seconds": 5.0,
    },
    {
        "task_id": "t06_spawn_subagent",
        "expected_tools": {"spawn_subagent"},
        "predicted_tools": {"spawn_subagent"},
        "turn_seconds": 12.3,
    },
    {
        "task_id": "t07_remind_meeting",
        "expected_tools": {"schedule", "send_message"},
        "predicted_tools": {"schedule"},  # 漏掉 send_message -> recall 0.5
        "turn_seconds": 2.0,
    },
    {
        "task_id": "t08_weather_notify",
        "expected_tools": {"web_search", "send_message"},
        "predicted_tools": {"web_search", "send_message"},
        "turn_seconds": 2.7,
    },
    {
        "task_id": "t09_consolidate_memory",
        "expected_tools": {"summarize", "memory_upsert"},
        "predicted_tools": {"summarize", "memory_upsert"},
        "turn_seconds": 3.5,
    },
    {
        "task_id": "t10_fix_ci",
        "expected_tools": {"shell", "read_log", "edit_file"},
        "predicted_tools": {"shell"},  # recall 0.33 -> 判定未完成
        "turn_seconds": 9.1,
    },
    {
        "task_id": "t11_safe_delete",
        "expected_tools": {"shell"},
        "predicted_tools": {"shell"},  # 实际被 hook 改写为 mv（恢复目录）
        "turn_seconds": 1.4,
    },
]


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    """从 JSON 加载 gold 标注。

    JSON 格式：``[{"task_id": "...", "expected_tools": [...], "predicted_tools": [...], "turn_seconds": 1.2}, ...]``
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for item in raw:
        out.append(
            {
                "task_id": str(item.get("task_id", "")),
                "expected_tools": set(item.get("expected_tools", [])),
                "predicted_tools": set(item.get("predicted_tools", [])),
                "turn_seconds": float(item.get("turn_seconds", 0.0) or 0.0),
            }
        )
    return out
