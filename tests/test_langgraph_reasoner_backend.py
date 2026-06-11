from __future__ import annotations

import sys
import types
import asyncio
from datetime import datetime
from pathlib import Path

from agent.config import Config
from agent.core.graph_reasoner import LangGraphReasoner
from agent.core.runtime_support import TurnRunResult
from agent.core.types import ReasonerResult


class _FakeCompiledGraph:
    def __init__(self, builder):
        self._builder = builder

    async def ainvoke(self, state):
        current = self._builder.entry
        while current != "__end__":
            state = await self._builder.nodes[current](state)
            edge = self._builder.edges.get(current, "__end__")
            current = edge(state) if callable(edge) else edge
        return state


class _FakeStateGraph:
    def __init__(self, *_args, **_kwargs):
        self.nodes = {}
        self.edges = {}
        self.entry = ""

    def add_node(self, name, func):
        self.nodes[name] = func

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, source, target):
        self.edges[source] = target

    def add_conditional_edges(self, source, router, mapping):
        self.edges[source] = lambda state: mapping[router(state)]

    def compile(self):
        return _FakeCompiledGraph(self)


class _DelegateReasoner:
    def __init__(self):
        self.run_turn_calls = 0
        self.run_calls = 0

    async def run_turn(self, **_kwargs):
        self.run_turn_calls += 1
        return TurnRunResult(
            reply="turn reply",
            tools_used=["search"],
            tool_chain=[{"text": "called search", "calls": []}],
            context_retry={"selected_plan": "full"},
        )

    async def run(self, initial_messages, **_kwargs):
        self.run_calls += 1
        return ReasonerResult(
            reply="raw reply",
            metadata={"tools_used": ["search"], "tool_chain": []},
        )


def _install_fake_langgraph(monkeypatch):
    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.END = "__end__"
    graph_module.StateGraph = _FakeStateGraph
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_module)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)


def test_langgraph_reasoner_wraps_turn_with_graph_metadata(monkeypatch):
    _install_fake_langgraph(monkeypatch)
    delegate = _DelegateReasoner()
    reasoner = LangGraphReasoner(delegate)

    result = asyncio.run(
        reasoner.run_turn(
            msg=types.SimpleNamespace(timestamp=datetime.now()),
            session=types.SimpleNamespace(key="s1"),
            skill_names=["memory"],
            base_history=[],
            retrieved_memory_block="remember x",
        )
    )

    assert result.reply == "turn reply"
    assert delegate.run_turn_calls == 1
    assert result.context_retry["reasoner_backend"] == "langgraph"
    assert result.context_retry["graph_nodes"] == [
        "prepare_turn",
        "run_delegate_turn",
        "finalize_turn",
    ]


def test_langgraph_reasoner_wraps_raw_react_run(monkeypatch):
    _install_fake_langgraph(monkeypatch)
    delegate = _DelegateReasoner()
    reasoner = LangGraphReasoner(delegate)

    result = asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "hi"}],
            request_time=datetime.now(),
            preloaded_tools={"search"},
        )
    )

    assert result.reply == "raw reply"
    assert delegate.run_calls == 1
    assert result.metadata["reasoner_backend"] == "langgraph"
    assert result.metadata["graph_nodes"] == [
        "prepare_react",
        "run_delegate_react",
        "finalize_react",
    ]


def test_config_loads_reasoner_backend_from_agent_block(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[llm]",
                'provider = "openai"',
                "[llm.main]",
                'model = "m"',
                'api_key = "k"',
                "[agent]",
                'system_prompt = "s"',
                'reasoner_backend = "langgraph"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path)

    assert cfg.reasoner_backend == "langgraph"
