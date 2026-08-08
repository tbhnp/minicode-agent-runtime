"""StreamingObserveWriter：在不改动 agent 主链路的前提下，把 observe_writer

发出的结构化 trace 事件（TurnTrace / RagTrace / MemoryWriteTrace /

ProactiveDecisionTrace）同时广播到进程内 StreamHub，供 SSE / WebSocket 实时推送。



- 保持原 writer（如 TraceWriter 写 SQLite）的既有行为，emit 时原样透传。

- 仅额外把事件以 dict 形式 publish 到 hub，失败不影响主流程。

- run() 委托给底层 writer，保证 bootstrap/app.py 的后台 task 语义不变。
"""

from __future__ import annotations



import dataclasses

import logging

from typing import Any



from core.observe.events import (

    MemoryWriteTrace,

    ProactiveDecisionTrace,

    RagTrace,

    TurnTrace,

)

from core.observe.stream_hub import StreamHub, get_stream_hub



logger = logging.getLogger("observe.streaming_writer")



_EVENT_TYPES = (TurnTrace, RagTrace, ProactiveDecisionTrace, MemoryWriteTrace)





class StreamingObserveWriter:

    def __init__(self, inner: Any, hub: StreamHub | None = None) -> None:

        self._inner = inner

        self._hub = hub or get_stream_hub()



    def emit(self, event: Any) -> None:

        # 1) 保持原 writer 行为（写 SQLite / 其他 sink）

        if self._inner is not None:

            try:

                self._inner.emit(event)

            except Exception:  # pragma: no cover - 防御性

                logger.warning("inner observe writer emit failed", exc_info=True)



        # 2) 广播到 stream hub（仅结构化 trace 事件）

        if isinstance(event, _EVENT_TYPES):

            try:

                payload = dataclasses.asdict(event)

                payload["_type"] = type(event).__name__

                self._hub.publish(payload)

            except Exception:  # pragma: no cover - 防御性

                logger.warning("stream hub publish failed", exc_info=True)



    async def run(self) -> None:

        if self._inner is not None:

            result = self._inner.run()

            if hasattr(result, "__await__"):

                await result
