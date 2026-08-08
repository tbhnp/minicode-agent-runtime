"""本地向量库：numpy 余弦相似度，内存存储。

对标标准 RAG 的 dense retrieval 阶段。无外部向量数据库依赖，
可在缺 numpy 时退化为纯 Python 实现，保证可移植与可测试。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NP = True
except Exception:  # pragma: no cover - 依赖可选
    np = None
    _HAS_NP = False


@dataclass
class StoredChunk:
    id: str
    doc_id: str
    text: str
    index: int
    embedding: list[float]
    meta: dict = field(default_factory=dict)


class LocalVectorStore:
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._by_id: dict[str, StoredChunk] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunk: StoredChunk) -> None:
        if chunk.id in self._by_id:
            return
        self._by_id[chunk.id] = chunk
        self._chunks.append(chunk)

    def get(self, chunk_id: str) -> StoredChunk | None:
        return self._by_id.get(chunk_id)

    def search(self, query_vec: list[float], top_k: int = 10) -> list[tuple[StoredChunk, float]]:
        if not self._chunks:
            return []
        if _HAS_NP:
            q = np.asarray(query_vec, dtype=float)
            qn = float(np.linalg.norm(q)) or 1.0
            q = q / qn
            mat = np.asarray([c.embedding for c in self._chunks], dtype=float)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = mat @ q / norms
            idx = np.argsort(-sims)[:top_k]
            return [(self._chunks[int(i)], float(sims[int(i)])) for i in idx]
        scored = [(c, _cosine(query_vec, c.embedding)) for c in self._chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
