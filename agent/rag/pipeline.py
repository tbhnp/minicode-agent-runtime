"""标准 RAG 流水线：分块 → 嵌入 → 混合召回（向量 + BM25，RRF 融合）→ 重排 → 取 Top-K。

对标大厂 RAG 工程要求：
- Hybrid retrieval：dense（向量）+ sparse（BM25）互补召回；
- RRF（Reciprocal Rank Fusion）无权重融合，鲁棒且免调参；
- Rerank：对召回候选精排，提升答案片段置顶率。

embedder 需实现 async embed(text) / embed_batch(texts) -> list[list[float]]，
可直接复用 memory2.embedder.Embedder（DashScope text-embedding-v3）。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.rag.bm25 import Bm25Index
from agent.rag.chunker import chunk_text
from agent.rag.rerank import LexicalReranker, Reranker
from agent.rag.types import RagDoc, RagResult, RetrievedChunk
from agent.rag.vector_store import LocalVectorStore, StoredChunk

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(
        self,
        embedder: Any,
        *,
        reranker: Reranker | None = None,
        vector_store: LocalVectorStore | None = None,
        bm25: Bm25Index | None = None,
        top_k: int = 5,
        hybrid_top_n: int = 20,
        rrf_k: int = 60,
        chunk_max_chars: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker or LexicalReranker()
        self._vs = vector_store or LocalVectorStore()
        self._bm25 = bm25 or Bm25Index()
        self._top_k = max(1, int(top_k))
        self._hybrid_top_n = max(1, int(hybrid_top_n))
        self._rrf_k = max(1, int(rrf_k))
        self._cmax = chunk_max_chars
        self._cov = chunk_overlap

    async def ingest(self, documents: list[RagDoc]) -> int:
        """切分 + 批量嵌入 + 写入向量库与 BM25 索引。返回新增 chunk 数。"""
        texts: list[str] = []
        meta: list[tuple[str, str, int, dict]] = []
        for doc in documents:
            chunks = chunk_text(doc.text, max_chars=self._cmax, overlap=self._cov)
            for i, c in enumerate(chunks):
                cid = f"{doc.id}#c{i}"
                texts.append(c)
                meta.append((cid, doc.id, i, doc.meta))

        if not texts:
            return 0

        embs = await self._embedder.embed_batch(texts)
        added = 0
        for (cid, doc_id, idx, dmeta), text, emb in zip(meta, texts, embs):
            self._vs.add(
                StoredChunk(
                    id=cid,
                    doc_id=doc_id,
                    text=text,
                    index=idx,
                    embedding=emb,
                    meta=dmeta,
                )
            )
            self._bm25.add(cid, doc_id, text)
            added += 1
        return added

    async def ingest_files(self, paths: list[str | Path], *, doc_id_prefix: str = "file") -> int:
        """多格式文档入库：解析（PDF/DOCX/MD/txt，PDF/DOCX 优先 MinerU）→ 切分 → 嵌入 → 索引。

        解析放在 to_thread 中执行，避免 MinerU 重解析阻塞事件循环，也为其
        可能的异步路径提供干净的 asyncio 上下文。
        """
        from agent.rag.ingest import parse_documents, sections_to_ragdocs

        sections = await asyncio.to_thread(parse_documents, paths)
        if not sections:
            return 0
        docs = sections_to_ragdocs(sections, doc_id_prefix=doc_id_prefix)
        return await self.ingest(docs)

    async def query(self, text: str, top_k: int | None = None) -> RagResult:
        top_k = top_k or self._top_k
        qvec = await self._embedder.embed(text)

        vec_hits = {c.id: s for c, s in self._vs.search(qvec, self._hybrid_top_n)}
        bm25_hits = {cid: s for cid, _, _, s in self._bm25.search(text, self._hybrid_top_n)}

        # Reciprocal Rank Fusion
        fused: dict[str, float] = {}
        for rank, cid in enumerate(sorted(vec_hits, key=lambda x: -vec_hits[x])):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        for rank, cid in enumerate(sorted(bm25_hits, key=lambda x: -bm25_hits[x])):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        candidates: list[RetrievedChunk] = []
        for cid, fscore in sorted(fused.items(), key=lambda x: -x[1]):
            sc = self._vs.get(cid)
            if sc is None:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=sc.doc_id,
                    text=sc.text,
                    vector_score=vec_hits.get(cid, 0.0),
                    bm25_score=bm25_hits.get(cid, 0.0),
                    fused_score=fscore,
                    meta=sc.meta,
                )
            )

        reranked = await self._reranker.rerank(text, candidates, top_k)
        context = "\n\n".join(f"[doc:{c.doc_id}]\n{c.text}" for c in reranked)
        return RagResult(query=text, chunks=reranked, context=context)
