"""进程内事件流 Hub：把 observe_writer 发出的结构化 trace 事件以 pub/sub 方式

广播给订阅者（按 session_key 过滤，支持通配 "*"），供 SSE / WebSocket 端点实时推送

Agent 思考流、工具调用、RAG 检索、记忆写入等事件。



设计要点：

- 单例：整个进程共享一个 hub（dashboard 与 agent 同进程）。

- 线程安全：agent emit 在主事件循环线程；若跨线程 publish（如测试线程），

  通过 event loop 的 call_soon_threadsafe 调度 put_nowait，避免破坏 asyncio.Queue。

- 背压：每个订阅者队列有上限，满则丢弃并记日志，不阻塞主循环。
"""

from __future__ import annotations



import asyncio

import logging

import threading

from typing import Any



logger = logging.getLogger("observe.stream_hub")



_SUBSCRIBER_QUEUE_MAX = 256





class StreamHub:

    def __init__(self) -> None:

        self._loop: asyncio.AbstractEventLoop | None = None

        self._loop_thread: threading.Thread | None = None

        # 通配订阅（session_key == "*"）

        self._wild: list[asyncio.Queue] = []

        # 按 session_key 精确订阅

        self._by_session: dict[str, list[asyncio.Queue]] = {}



    # ── 订阅管理 ────────────────────────────────────────────────────────────



    def subscribe(self, session_key: str) -> asyncio.Queue:

        self._record_loop()

        queue: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)

        key = session_key or "*"

        if key == "*":

            self._wild.append(queue)

        else:

            self._by_session.setdefault(key, []).append(queue)

        return queue



    def unsubscribe(self, queue: asyncio.Queue, session_key: str) -> None:

        key = session_key or "*"

        target = self._wild if key == "*" else self._by_session.get(key, [])

        try:

            target.remove(queue)

        except ValueError:

            pass



    def _record_loop(self) -> None:

        try:

            self._loop = asyncio.get_running_loop()

            self._loop_thread = threading.current_thread()

        except RuntimeError:

            pass



    # ── 发布 ────────────────────────────────────────────────────────────────



    def publish(self, event: dict[str, Any]) -> None:

        session_key = str(event.get("session_key", "") or "")

        targets = list(self._wild)

        if session_key:

            targets = targets + self._by_session.get(session_key, [])

        if not targets:

            return

        for queue in targets:

            self._put(queue, event)



    def _put(self, queue: asyncio.Queue, event: dict[str, Any]) -> None:

        if (

            self._loop_thread is not None

            and threading.current_thread() is not self._loop_thread

            and self._loop is not None

        ):

            # 跨线程发布：调度到事件循环线程安全执行

            self._loop.call_soon_threadsafe(self._do_put, queue, event)

            return

        self._do_put(queue, event)



    @staticmethod

    def _do_put(queue: asyncio.Queue, event: dict[str, Any]) -> None:

        try:

            queue.put_nowait(event)

        except asyncio.QueueFull:

            logger.debug("stream hub subscriber queue full, drop event")

        except Exception:  # pragma: no cover - 防御性

            logger.warning("stream hub put failed", exc_info=True)





_HUB: StreamHub | None = None





def get_stream_hub() -> StreamHub:

    """进程内单例：dashboard 与 agent 共享同一 hub。"""

    global _HUB

    if _HUB is None:

        _HUB = StreamHub()

    return _HUB
