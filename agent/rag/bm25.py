"""BM25 词法检索：标准 RAG 的 sparse / lexical 召回。

与 dense 向量召回互补，擅长精确词面匹配（专有名词、ID、代码等），
通过 RRF 与向量召回融合可显著提升命中率。
"""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class _Doc:
    __slots__ = ("doc_id", "chunk_id", "text", "tf", "length")

    def __init__(self, doc_id: str, chunk_id: str, text: str, tf: dict[str, int], length: int) -> None:
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text
        self.tf = tf
        self.length = length


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: list[_Doc] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0
        self._n = 0

    def add(self, chunk_id: str, doc_id: str, text: str) -> None:
        toks = _tokenize(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        d = _Doc(doc_id=doc_id, chunk_id=chunk_id, text=text, tf=tf, length=len(toks))
        self._docs.append(d)
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self._n += 1
        self._avgdl = sum(x.length for x in self._docs) / max(1, self._n)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, str, str, float]]:
        q_toks = _tokenize(query)
        if not q_toks or not self._docs:
            return []
        out: list[tuple[str, str, str, float]] = []
        for d in self._docs:
            score = 0.0
            for t in set(q_toks):
                if t not in d.tf:
                    continue
                df_t = self._df.get(t, 0)
                idf = math.log(1 + (self._n - df_t + 0.5) / (df_t + 0.5))
                tf_t = d.tf[t]
                denom = tf_t + self._k1 * (1 - self._b + self._b * d.length / max(1, self._avgdl))
                score += idf * (tf_t * (self._k1 + 1)) / denom
            if score > 0:
                out.append((d.chunk_id, d.doc_id, d.text, score))
        out.sort(key=lambda x: x[3], reverse=True)
        return out[:top_k]
