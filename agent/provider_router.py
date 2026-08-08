"""多模型路由 + 限流 + Token 成本治理。

对标大厂 Agent 后端要求（字节 TRAE「模型路由 / 限流 / 成本治理」）：

- ModelRouter 在多个 LLMProvider 之上做统一调度；

- 路由策略：指定模型 / 成本优先 / 轮询 / 降级链（按 priority）；

- 每模型令牌桶限流，超限自动降级到下一候选；

- 每次调用统计 prompt/completion token 与估算成本（USD），上报 Prometheus；

- 单次调用失败（重试耗尽 / 安全 / 上下文超限）自动降级到下一候选路由。
"""

from __future__ import annotations



import asyncio

import logging

import time

from dataclasses import dataclass, field

from typing import Any



from agent.provider import LLMResponse

from core.observe.prom import (

    observe_llm,

    observe_llm_cost,

    observe_llm_duration,

    observe_llm_tokens,

)



logger = logging.getLogger("llm.router")



# 成本参考（USD / 1K tokens），可按模型覆盖；缺省 0（仅统计 token）。

DEFAULT_COST_TABLE: dict[str, dict[str, float]] = {

    "qwen3.6-plus": {"prompt": 0.002, "completion": 0.006},

    "qwen-flash": {"prompt": 0.0004, "completion": 0.0008},

}





class TokenBucket:

    """简单令牌桶：每秒补充 rate/60 个令牌，供按分钟限速使用。"""



    def __init__(self, rate_per_min: float) -> None:

        self._rate = max(0.0, float(rate_per_min))

        self._tokens = self._rate

        self._updated = time.monotonic()

        self._lock = asyncio.Lock()



    async def acquire(self, n: int = 1) -> bool:

        async with self._lock:

            now = time.monotonic()

            elapsed = now - self._updated

            self._tokens = min(self._rate, self._tokens + elapsed * (self._rate / 60.0))

            self._updated = now

            if self._tokens >= n:

                self._tokens -= n

                return True

            return False





@dataclass

class ModelRoute:

    name: str

    provider: Any  # LLMProvider-like，需实现 async chat(...)

    priority: int = 0

    cost_table: dict[str, float] = field(default_factory=dict)

    max_rpm: float = 0.0  # 0 = 不限流

    timeout_s: float = 90.0

    _bucket: TokenBucket | None = field(default=None, repr=False)



    def bucket(self) -> TokenBucket | None:

        if self.max_rpm > 0 and self._bucket is None:

            self._bucket = TokenBucket(self.max_rpm)

        return self._bucket





class Strategy:

    EXPLICIT = "explicit"

    COST = "cost"

    ROUND_ROBIN = "round_robin"

    FALLBACK = "fallback"





def _est_tokens(text: str | None) -> int:

    if not text:

        return 0

    return max(1, len(text) // 4)





class ModelRouter:

    def __init__(self, routes: list[ModelRoute], cost_table: dict | None = None) -> None:

        if not routes:

            raise ValueError("at least one route required")

        self._routes = sorted(routes, key=lambda r: r.priority)

        self._by_name = {r.name: r for r in routes}

        self._cost = dict(DEFAULT_COST_TABLE)

        if cost_table:

            self._cost.update(cost_table)

        self._rr = 0

        self._lock = asyncio.Lock()



    def route_names(self) -> list[str]:

        return list(self._by_name)



    async def chat(

        self,

        *,

        messages: list[dict],

        tools=None,

        max_tokens: int,

        model_hint: str | None = None,

        strategy: str = Strategy.FALLBACK,

        tool_choice: str | dict = "auto",

        extra_body: dict | None = None,

        on_content_delta=None,

        **kw,

    ) -> LLMResponse:

        candidates = self._select(strategy, model_hint)

        last_err: Exception | None = None

        for route in candidates:

            bucket = route.bucket()

            if bucket is not None and not await bucket.acquire(1):

                logger.info("[router] route=%s rate-limited, fallback", route.name)

                observe_llm(route.name, strategy, "rate_limited")

                continue

            try:

                start = time.perf_counter()

                resp = await route.provider.chat(

                    messages=messages,

                    tools=tools or [],

                    model=route.name,

                    max_tokens=max_tokens,

                    tool_choice=tool_choice,

                    extra_body=extra_body,

                    on_content_delta=on_content_delta,

                    **kw,

                )

                observe_llm_duration(route.name, time.perf_counter() - start)

            except Exception as e:  # 降级到下一候选

                observe_llm(route.name, strategy, "error")

                last_err = e

                logger.warning("[router] route=%s failed: %s", route.name, e)

                continue

            self._record_cost(route, resp, messages)

            observe_llm(route.name, strategy, "ok")

            return resp

        if last_err:

            raise last_err

        raise RuntimeError("all routes failed")



    def _select(self, strategy: str, model_hint) -> list[ModelRoute]:

        if model_hint and model_hint in self._by_name:

            return [self._by_name[model_hint]]

        if strategy == Strategy.COST:

            return sorted(self._routes, key=self._cost_per_1k)

        if strategy == Strategy.ROUND_ROBIN:

            order = self._routes[self._rr :] + self._routes[: self._rr]

            self._rr = (self._rr + 1) % len(self._routes)

            return order

        return list(self._routes)  # FALLBACK / 默认：按 priority



    def _cost_per_1k(self, route: ModelRoute) -> float:

        t = self._cost.get(route.name, {})

        return t.get("prompt", 0.0) + t.get("completion", 0.0)



    def _record_cost(self, route: ModelRoute, resp: LLMResponse, messages) -> None:

        prompt_tokens = resp.prompt_tokens

        completion_tokens = resp.completion_tokens

        if prompt_tokens is None:

            prompt_tokens = _est_tokens(

                " ".join(str(m.get("content", "")) for m in messages)

            )

        if completion_tokens is None:

            completion_tokens = _est_tokens(resp.content)

        if prompt_tokens:

            observe_llm_tokens(route.name, "prompt", prompt_tokens)

        if completion_tokens:

            observe_llm_tokens(route.name, "completion", completion_tokens)

        cost = (

            (prompt_tokens / 1000.0) * self._cost.get(route.name, {}).get("prompt", 0.0)

            + (completion_tokens / 1000.0)

            * self._cost.get(route.name, {}).get("completion", 0.0)

        )

        observe_llm_cost(route.name, cost)
