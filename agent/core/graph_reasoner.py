from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, TypedDict

from agent.core.reasoner import Reasoner
from agent.core.runtime_support import SessionLike, TurnRunResult
from agent.core.types import ReasonerResult


class _GraphState(TypedDict, total=False):
    msg: Any
    session: SessionLike
    skill_names: list[str] | None
    base_history: list[dict] | None
    retrieved_memory_block: str
    initial_messages: list[dict]
    request_time: datetime | None
    preloaded_tools: set[str] | None
    preflight_injected: bool
    on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None
    on_progress: Callable[[dict[str, object]], Awaitable[None]] | None
    graph_nodes: list[str]
    turn_result: TurnRunResult
    reasoner_result: ReasonerResult


class LangGraphReasoner(Reasoner):
    """LangGraph-backed orchestration wrapper for the existing Reasoner.

    The project keeps ToolRegistry, MCP, memory retrieval, trace writing, and
    the tested ReAct loop intact. LangGraph is introduced as a pluggable state
    graph boundary so production deployments can later add checkpointing,
    branch routing, and human approval without replacing the runtime.
    """

    def __init__(self, delegate: Reasoner) -> None:
        self._delegate = delegate
        self._turn_graph = self._build_turn_graph()
        self._react_graph = self._build_react_graph()

    def set_stream_sink_factory(
        self,
        factory: Callable[
            [object], Callable[[dict[str, str] | str], Awaitable[None]] | None
        ]
        | None,
    ) -> None:
        setter = getattr(self._delegate, "set_stream_sink_factory", None)
        if callable(setter):
            setter(factory)

    def set_progress_sink_factory(
        self,
        factory: Callable[
            [object], Callable[[dict[str, object]], Awaitable[None]] | None
        ]
        | None,
    ) -> None:
        setter = getattr(self._delegate, "set_progress_sink_factory", None)
        if callable(setter):
            setter(factory)

    async def run_turn(
        self,
        *,
        msg,
        session: SessionLike,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> TurnRunResult:
        state: _GraphState = {
            "msg": msg,
            "session": session,
            "skill_names": skill_names,
            "base_history": base_history,
            "retrieved_memory_block": retrieved_memory_block,
            "graph_nodes": [],
        }
        final_state = await self._turn_graph.ainvoke(state)
        return final_state["turn_result"]

    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        on_progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ) -> ReasonerResult:
        state: _GraphState = {
            "initial_messages": initial_messages,
            "request_time": request_time,
            "preloaded_tools": preloaded_tools,
            "preflight_injected": preflight_injected,
            "on_content_delta": on_content_delta,
            "on_progress": on_progress,
            "graph_nodes": [],
        }
        final_state = await self._react_graph.ainvoke(state)
        return final_state["reasoner_result"]

    def _build_turn_graph(self):
        StateGraph, END = _load_langgraph()
        graph = StateGraph(_GraphState)

        async def prepare_turn(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("prepare_turn")
            return state

        async def run_delegate_turn(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("run_delegate_turn")
            state["turn_result"] = await self._delegate.run_turn(
                msg=state["msg"],
                session=state["session"],
                skill_names=state.get("skill_names"),
                base_history=state.get("base_history"),
                retrieved_memory_block=state.get("retrieved_memory_block", ""),
            )
            return state

        async def finalize_turn(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("finalize_turn")
            result = state["turn_result"]
            result.context_retry = {
                **dict(result.context_retry or {}),
                "reasoner_backend": "langgraph",
                "graph_nodes": list(state.get("graph_nodes", [])),
            }
            return state

        graph.add_node("prepare_turn", prepare_turn)
        graph.add_node("run_delegate_turn", run_delegate_turn)
        graph.add_node("finalize_turn", finalize_turn)
        graph.set_entry_point("prepare_turn")
        graph.add_edge("prepare_turn", "run_delegate_turn")
        graph.add_edge("run_delegate_turn", "finalize_turn")
        graph.add_edge("finalize_turn", END)
        return graph.compile()

    def _build_react_graph(self):
        StateGraph, END = _load_langgraph()
        graph = StateGraph(_GraphState)

        async def prepare_react(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("prepare_react")
            return state

        async def run_delegate_react(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("run_delegate_react")
            state["reasoner_result"] = await self._delegate.run(
                state["initial_messages"],
                request_time=state.get("request_time"),
                preloaded_tools=state.get("preloaded_tools"),
                preflight_injected=state.get("preflight_injected", True),
                on_content_delta=state.get("on_content_delta"),
                on_progress=state.get("on_progress"),
            )
            return state

        async def finalize_react(state: _GraphState) -> _GraphState:
            state.setdefault("graph_nodes", []).append("finalize_react")
            result = state["reasoner_result"]
            result.metadata = {
                **dict(result.metadata or {}),
                "reasoner_backend": "langgraph",
                "graph_nodes": list(state.get("graph_nodes", [])),
            }
            return state

        graph.add_node("prepare_react", prepare_react)
        graph.add_node("run_delegate_react", run_delegate_react)
        graph.add_node("finalize_react", finalize_react)
        graph.set_entry_point("prepare_react")
        graph.add_edge("prepare_react", "run_delegate_react")
        graph.add_edge("run_delegate_react", "finalize_react")
        graph.add_edge("finalize_react", END)
        return graph.compile()


def _load_langgraph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "reasoner_backend='langgraph' requires the langgraph package. "
            "Install project dependencies from requirements.txt before enabling it."
        ) from exc
    return StateGraph, END


__all__ = ["LangGraphReasoner"]
