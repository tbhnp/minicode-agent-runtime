"""RAG 运行期单例：从环境变量懒加载 RagPipeline（12-factor 配置）。

配置（标准环境变量）：
  RAG_EMBEDDING_BASE_URL  嵌入服务地址（OpenAI 兼容 /embeddings，如 DashScope）
  RAG_EMBEDDING_API_KEY   嵌入服务密钥
  RAG_EMBEDDING_MODEL     嵌入模型名（默认 text-embedding-v3）

未配置时 get_rag_pipeline() 返回 None，调用方（dashboard 端点）应返回 501。
这样 RAG 能力在不依赖外部嵌入服务时也不会影响进程启动。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PIPELINE = None
_INITED = False


def get_rag_pipeline():
    global _PIPELINE, _INITED
    if _INITED:
        return _PIPELINE
    _INITED = True

    base_url = os.environ.get("RAG_EMBEDDING_BASE_URL")
    api_key = os.environ.get("RAG_EMBEDDING_API_KEY")
    if not base_url or not api_key:
        logger.info("[rag] embedding env not set; RAG endpoints disabled")
        return None

    model = os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-v3")
    try:
        from core.net.http import HttpRequester, RequestBudget, RetryPolicy
        import httpx
        from memory2.embedder import Embedder
        from agent.rag.pipeline import RagPipeline

        # 自包含 HTTP 客户端，不依赖全局共享 http 资源配置。
        requester = HttpRequester(
            client=httpx.AsyncClient(timeout=30.0),
            retry_policy=RetryPolicy(),
            default_timeout_s=30.0,
            default_budget=RequestBudget(total_timeout_s=40.0),
        )
        embedder = Embedder(
            base_url=base_url, api_key=api_key, model=model, requester=requester
        )
        _PIPELINE = RagPipeline(embedder)
        logger.info("[rag] pipeline ready (model=%s)", model)
    except Exception as e:  # pragma: no cover - 运行期依赖异常
        logger.warning("[rag] failed to build pipeline: %s", e)
        _PIPELINE = None
    return _PIPELINE
