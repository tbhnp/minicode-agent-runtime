"""标准 RAG 链路的共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """一次召回（混合检索后、重排前/后的候选片段）。"""

    chunk_id: str
    doc_id: str
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class RagDoc:
    """待入库的文档（会被切分为多个 chunk）。"""

    id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class RagResult:
    """RAG 查询的最终结果。"""

    query: str
    chunks: list[RetrievedChunk]
    context: str
