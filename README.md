# MiniCode Agent Runtime

MiniCode Agent Runtime is a personal agent runtime inspired by Claude Code-style agentic coding systems. It focuses on long-running task execution, dynamic tool routing, context engineering, long-term memory, subagent isolation, permission control, checkpoint rollback, and trace-based observability.

## Highlights

- **Agentic Loop**: Query Loop + Tool Use execution flow for multi-step tasks.
- **Dynamic Tool Routing**: ToolRegistry and `tool_search` reduce always-on tool schema cost and tool mis-selection.
- **Skill System**: `SKILL.md` based skill directories for reusable high-level capabilities.
- **Context Engineering**: Session history, long-term memory, static prompt blocks, and tool results are managed separately with budget control.
- **Long-Term Memory**: SQLite / sqlite-vec based memory retrieval with `source_ref`, `supersede`, and consolidation support.
- **SubAgent Isolation**: Research, scripting, and general subagent profiles use different tool permissions and isolated task execution.
- **Permission Governance**: Tool risk levels and guard hooks control read, write, and external side-effect tools.
- **Checkpoint Rollback**: File mutations can be checkpointed and restored for safer agentic coding workflows.
- **Trace and Evaluation**: Turn traces, tool chains, memory/RAG traces, proactive traces, and offline memory benchmarks support debugging and regression checks.

## Architecture

```text
Inbound Message
  -> AgentLoop
  -> CoreRunner
  -> AgentCore
  -> ContextStore.prepare()
  -> Reasoner / Tool Use Loop
  -> ToolExecutor + Hooks
  -> ContextStore.commit()
  -> Outbound Message
```

Core components:

- `agent/`: agent runtime, reasoner, tools, subagents, hooks, skills.
- `bootstrap/`: runtime wiring and toolset registration.
- `core/`: memory, observability, networking, shared runtime contracts.
- `memory2/`: semantic long-term memory, retrieval, deduplication, consolidation.
- `proactive_v2/`: proactive event and drift execution pipeline.
- `eval/`: memory and long-context evaluation utilities.
- `tests/`: unit and regression tests.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.example.toml config.toml
python main.py init
python main.py
```

Edit `config.toml` before running:

```toml
[llm.main]
model = "your-model"
api_key = "your-api-key"
base_url = "https://your-provider/v1"

[llm.fast]
model = "your-fast-model"
api_key = "your-api-key"
base_url = "https://your-provider/v1"
```

Use environment variables where possible:

```toml
api_key = "${QWEN_API_KEY}"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_file_checkpoint_tools.py tests/test_tool_executor.py tests/test_tool_search.py -q
```

Some integration tests require optional services, real LLM credentials, Telegram/QQ channels, or platform-specific APIs.

## Public Release Notes

This public folder intentionally excludes local/private materials:

- real `config.toml`
- `.git` history
- private submodules
- local MCP server pointers
- runtime memory databases
- logs and generated artifacts
- resume/interview documents

Because of that, a fresh clone may not have every local integration enabled by default. The core runtime, tool routing, memory modules, subagent structure, permission hooks, checkpoint rollback, and tests remain available in code.
