"""重排（Rerank）阶段：标准 RAG 链路的关键精排一步。

为何需要 rerank：混合召回（向量 + BM25）对候选做了粗排，但 embedding 语义召回
常把"语义相关但答非所问"的片段排得很靠前。rerank 用更细的信号（词面重叠、
短语命中、或交叉编码器）对召回结果重排，显著提升"答案片段"的置顶率。

提供：
- Reranker 协议（可插拔）；
- LexicalReranker：基于查询-候选词法重叠的重排，无网络依赖，可作为默认/兜底；
- CrossEncoderReranker：对接标准 /rerank API（Cohere / Jina / DashScope 等），
  用交叉编码器对 (query, doc) 联合打分，精度最高。
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from agent.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class LexicalReranker:
    """基于查询-候选词法重叠（F1 + 短语命中）的重排，无需网络。"""

    def __init__(self, alpha: float = 0.7) -> None:
        self._alpha = alpha

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return candidates[:top_k]
        scored: list[tuple[RetrievedChunk, float]] = []
        for c in candidates:
            toks = _tokenize(c.text)
            if not toks:
                c.rerank_score = 0.0
                scored.append((c, 0.0))
                continue
            overlap = sum(1 for t in toks if t in q_tokens)
            precision = overlap / len(toks)
            recall = overlap / len(q_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            phrase = 0.0
            q_phrase = query.strip()
            if q_phrase and q_phrase in c.text:
                phrase = 1.0
            rerank = self._alpha * f1 + (1 - self._alpha) * (overlap / len(q_tokens)) + 0.5 * phrase
            c.rerank_score = rerank
            scored.append((c, rerank))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


class CrossEncoderReranker:
    """对接标准重排 API：对 (query, document) 联合打分，返回 relevance_score。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "rerank",
        requester=None,
        timeout_s: float = 10.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._key = api_key
        self._model = model
        self._requester = requester
        self._timeout_s = timeout_s

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self._requester is None:
            from core.net.http import get_default_http_requester

            self._requester = get_default_http_requester("external_default")
        documents = [c.text for c in candidates]
        resp = await self._requester.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
            timeout_s=self._timeout_s,
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
        ranked: list[RetrievedChunk] = []
        for item in data:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            if 0 <= idx < len(candidates):
                candidates[idx].rerank_score = score
                ranked.append(candidates[idx])
        return ranked
