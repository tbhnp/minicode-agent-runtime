"""轻量 Prometheus 指标定义，供 Dashboard API 与工具 Hook 共享。

prometheus_client 为可选依赖：未安装时所有指标退化为 no-op，
保证代码在缺少该依赖的运行环境下仍可正常导入与运行。
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    _PROM_AVAILABLE = False


class _Noop:
    """prometheus_client 缺失时的占位实现，链式调用安全。"""

    def __getattr__(self, _name: str):  # noqa: D401
        return self

    def __call__(self, *args, **kwargs):  # noqa: D401
        return self


if _PROM_AVAILABLE:
    HTTP_REQUESTS_TOTAL = Counter(
        "dashboard_http_requests_total",
        "Dashboard HTTP 请求总数",
        ["method", "endpoint", "status"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "dashboard_http_request_duration_seconds",
        "Dashboard HTTP 请求耗时（秒）",
        ["method", "endpoint"],
    )
    MEMORY_ITEMS_TOTAL = Gauge(
        "agent_memory_items_total", "当前活跃记忆条目数"
    )
    SESSIONS_TOTAL = Gauge("agent_sessions_total", "当前会话总数")
    PROACTIVE_DELIVERIES_TOTAL = Gauge(
        "agent_proactive_deliveries_total", "主动推送消息总数"
    )
    TOOL_RISK_DENIED_TOTAL = Counter(
        "agent_tool_risk_denied_total",
        "被风险策略拦截的工具调用总数",
        ["risk"],
    )
    LLM_REQUESTS_TOTAL = Counter(
        "agent_llm_requests_total",
        "LLM 调用总数",
        ["model", "strategy", "status"],
    )
    LLM_TOKENS_TOTAL = Counter(
        "agent_llm_tokens_total",
        "LLM token 消耗",
        ["model", "kind"],
    )
    LLM_COST_USD_TOTAL = Counter(
        "agent_llm_cost_usd_total",
        "LLM 估算成本(USD)",
        ["model"],
    )
    LLM_REQUEST_DURATION = Histogram(
        "agent_llm_request_duration_seconds",
        "LLM 请求耗时（秒）",
        ["model"],
    )
else:  # pragma: no cover
    HTTP_REQUESTS_TOTAL = _Noop()
    HTTP_REQUEST_DURATION = _Noop()
    MEMORY_ITEMS_TOTAL = _Noop()
    SESSIONS_TOTAL = _Noop()
    PROACTIVE_DELIVERIES_TOTAL = _Noop()
    TOOL_RISK_DENIED_TOTAL = _Noop()
    LLM_REQUESTS_TOTAL = _Noop()
    LLM_TOKENS_TOTAL = _Noop()
    LLM_COST_USD_TOTAL = _Noop()
    LLM_REQUEST_DURATION = _Noop()


def observe_tool_denied(risk: str) -> None:
    """记录一次被风险策略拦截的工具调用。"""
    if _PROM_AVAILABLE:
        try:
            TOOL_RISK_DENIED_TOTAL.labels(risk=risk or "unknown").inc()
        except Exception:
            pass


def observe_llm(model: str, strategy: str, status: str) -> None:
    """记录一次 LLM 调用的路由策略与状态（ok / error / rate_limited）。"""
    if _PROM_AVAILABLE:
        try:
            LLM_REQUESTS_TOTAL.labels(
                model=model or "unknown", strategy=strategy or "default", status=status
            ).inc()
        except Exception:
            pass


def observe_llm_tokens(model: str, kind: str, n: int) -> None:
    """记录 prompt / completion token 消耗。"""
    if _PROM_AVAILABLE and n:
        try:
            LLM_TOKENS_TOTAL.labels(model=model or "unknown", kind=kind).inc(int(n))
        except Exception:
            pass


def observe_llm_cost(model: str, usd: float) -> None:
    """记录估算的 LLM 成本（USD）。"""
    if _PROM_AVAILABLE and usd:
        try:
            LLM_COST_USD_TOTAL.labels(model=model or "unknown").inc(float(usd))
        except Exception:
            pass


def observe_llm_duration(model: str, seconds: float) -> None:
    """记录单次 LLM 请求耗时（秒）。"""
    if _PROM_AVAILABLE:
        try:
            LLM_REQUEST_DURATION.labels(model=model or "unknown").observe(float(seconds))
        except Exception:
            pass
