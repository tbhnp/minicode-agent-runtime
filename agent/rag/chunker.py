"""文本分块：中文友好的按句切分 + 滑动窗口重叠。

对标标准 RAG 链路的第一步：把长文档切成适合 embedding / 检索的 chunk。
- 优先按句末标点（。！？!? 与换行）切分；
- 将句子累积到接近 max_chars 时切出新块；
- 新块携带前一块末尾 overlap 个字符，缓解跨块语义割裂；
- 超长单句做硬切分。
"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")


def chunk_text(
    text: str,
    *,
    max_chars: int = 400,
    overlap: int = 80,
    min_chars: int = 20,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""

    for s in sentences:
        # 超长单句：硬切分为多个带重叠的片段
        if len(s) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, max_chars - overlap)
            for i in range(0, len(s), step):
                piece = s[i : i + max_chars]
                if piece:
                    chunks.append(piece)
            continue

        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap and len(cur) > overlap else cur
            cur = (tail + " " + s) if tail else s
        else:
            cur = (cur + " " + s) if cur else s

    if cur:
        chunks.append(cur)

    kept = [c for c in chunks if len(c) >= min_chars]
    return kept or [text]
