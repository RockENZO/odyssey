# Odyssey — Versatile AI Agent for Your Terminal

A local-first AI agent that runs in your terminal. Uses a configurable Ollama endpoint for model inference, with local SQLite and ChromaDB persistence and tools for coding, web research, tasks and memory.

## Recorded demo

![Odyssey creates a Python function and verifies it with a real tool call](docs/demo/odyssey-demo.gif)

A genuine CLI session using local `qwen3.8:27b-mlx`: the agent writes `greet.py`,
runs a Python assertion through its `bash` tool, and receives
`PASS: greeting verified`. Recorded from commit `c0137fa` on 2 November 2026 in a
disposable workspace. Idle gaps are shortened and viewing holds are added; this
is a functional demonstration, not a latency benchmark.

[Video (MP4)](docs/demo/odyssey-demo.mp4) ·
[Original terminal recording](docs/demo/session.cast) ·
[Saved tool results](docs/demo/tool-trace.jsonl) ·
[Recording details](docs/demo/README.md)

## Features

- **Interactive REPL** — `uv run odyssey` drops you into a continuous `> ` prompt. Chat back and forth, the AI uses tools autonomously.
- **17 tools** — File operations, bash, search, web research, memory, task management, journaling, sub-agents.
- **Local model inference by default** — The default Ollama endpoint is localhost. Web search and URL-reading tools send requests to external sites; a remote Ollama endpoint receives the conversation (including tool results such as file contents, shell output, fetched pages and retrieved memories) and memory text or queries sent for embedding. This is not an offline or no-egress guarantee.
- **Autonomous tool loop** — The LLM decides which tools to call and in what order to fulfill your request.
- **Coding + Daily life** — Read/edit files, run commands, grep/glob codebases, search the web, save memories, manage tasks, journal.

## Architecture

```
$ uv run odyssey
  │
  ▼  REPL (prompt_toolkit)
  │  > "research quantum computing and save to report.md"
  │
  ▼  Tool-Calling Loop
  │  LLM → tool_call → execute → result → LLM → tool_call → ... → done
  │
  ┌────────┬────────┬────────┬────────┬────────┬────────┐
  │ bash   │ read   │ write  │ edit   │ glob   │ grep   │
  ├────────┼────────┼────────┼────────┼────────┼────────┤
  │ web_   │ read_  │ save_  │ query_ │ add_   │ save_  │
  │ search │ url    │ memory │ memory │ task   │ journal│
  └────────┴────────┴────────┴────────┴────────┴────────┘
  │
  ▼  Ollama (local LLM)
     fast_model / deep_model from the loaded configuration
```

## Tools

| Tool | What it does |
|---|---|
| `bash` | Run shell commands (git, build, test, scripts) |
| `read` | Read file contents with line numbers |
| `write` | Create or overwrite files |
| `edit` | Exact-string replacement editing |
| `glob` | Find files by pattern (`**/*.py`) |
| `grep` | Regex content search via ripgrep |
| `web_search` | Search the web via DuckDuckGo |
| `read_url` | Fetch and extract content from a URL |
| `save_memory` | Save to long-term vector memory |
| `query_memory` | Semantic search over saved memories |
| `add_task` / `list_tasks` / `complete_task` | Persistent task management |
| `save_journal` / `get_journal_summary` | Journal with AI sentiment analysis |
| `todo_write` | Track multi-step progress |
| `task` | Spawn sub-agents for parallel work |

## What the agent can and cannot do today

- **Workspace boundary.** In the REPL the `read`, `write`, `edit`, `glob` and `grep` tools are confined to the project root (the nearest parent with a `.git` directory, otherwise the launch directory). Paths outside it, including symlinks that point outside, are refused. Set `ODYSSEY_ALLOW_OUTSIDE_WORKSPACE=1` to lift this. **`bash` is not sandboxed and is not covered by the boundary**; it starts in the launch directory and can reach anything your user can.
- **Tool-call ids.** Every tool call gets an id, and its result is stored with that id and the tool name. Executed calls are also kept in `Conversation.tool_log` with arguments, result and duration.
- **No resume yet.** The conversation lives in memory. `/save` writes a jsonl copy for reading; there is no way to continue a session or a half-finished task after the process exits.
- **No acceptance check.** A task ends when the model answers without calling a tool. Nothing runs the tests for it.
- **`todo_write` is a text echo.** It returns the list it was given and keeps no state.
- **Sub-agents are sequential.** `task` runs the same loop again in a fresh conversation, with the same tools and shell; it has no separate workspace or permissions.
- **`langgraph` is declared but not used yet.** It is a dependency for the planned graph-based engine, not part of the current loop.

## Quick Start

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) running locally
- Installed chat models supporting tool calls, selected explicitly in the configuration. Memory tools also require an embedding model compatible with Ollama’s embed API. The code currently defaults all three model fields to `qwen3.6:35b-a3b-coding-nvfp4`; availability and embedding support must be checked on your installation.

### Install & Run

```bash
git clone https://github.com/RockENZO/odyssey.git
cd odyssey
# Sync the project environment:
uv sync
uv run odyssey

# Or install globally so you can run `odyssey` from anywhere
uv tool install --force .
odyssey
```

This drops you into the REPL. Try:

```
> read the README and summarize it
> list all .py files in this project
> search the web for "local AI agents"
```

### Commands in the REPL

| Command | What it does |
|---|---|
| `/exit` or `/quit` | Exit Odyssey |
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/model` | Show current LLM model |
| `/status` | Check Ollama connection |
| `/history` | Show recent conversation |
| `/save` | Save session to file |
| `!command` | Run directly in shell (passthrough) |

### Configuration

The default configuration directory comes from `platformdirs` and differs by operating system; it is not universally `~/.config/odyssey`. Print it with:

```bash
# From the checkout:
uv run odyssey config-path
# After global installation, from any directory:
odyssey config-path
```

Create `config.toml` in that directory. Alternatively, set `ODYSSEY_CONFIG_DIR` to a chosen directory before launching. `ODYSSEY_DATA_DIR` overrides local persistence. Both overrides expand `~` and create missing parent directories automatically; an existing file or inaccessible directory raises an error without overwriting it. The `config-path` command creates the configuration directory but does not start the agent or contact Ollama. Use the exact installed model names; the following shows the checked-in defaults, not validated recommendations:

```toml
ollama_host = "http://localhost:11434"
fast_model = "qwen3.6:35b-a3b-coding-nvfp4"
deep_model = "qwen3.6:35b-a3b-coding-nvfp4"
embedding_model = "qwen3.6:35b-a3b-coding-nvfp4"
```

## How It Works

Odyssey is an **interactive CLI tool, not a background daemon**. Each `uv run odyssey` session:

1. Starts the REPL and waits for your input
2. For each message, runs the **tool-calling loop**: the LLM receives your request + tool schemas, decides which tool to call, executes it, gets the result, and repeats until it can answer
3. Returns to the `> ` prompt for your next message
4. When you `/exit`, the CLI process ends; persisted files remain and Ollama may continue running

The only persistent background process is **Ollama** itself (`ollama serve`), which keeps the LLM model loaded in memory.

## Project Structure

```
src/odyssey/
├── cli/           # REPL entry point (prompt_toolkit)
├── core/          # Tool-calling loop, conversation state, tool-call ids, workspace boundary
├── tools/         # 17 tool implementations
│   ├── bash.py          # shell execution
│   ├── file_ops.py      # read, write, edit
│   ├── search_tools.py  # glob, grep
│   ├── web.py           # web_search, read_url
│   ├── memory.py        # save/query memory
│   ├── task_manager.py  # task CRUD
│   ├── journal_tools.py # journal + sentiment
│   ├── sub_agent.py     # task sub-agent
│   └── registry.py      # tool registration
├── llm/           # Ollama async client
├── storage/       # SQLite + ChromaDB persistence
└── config.py      # settings
```

## Tech Stack

| Component | Choice |
|---|---|
| Agent architecture | Tool-calling loop (LLM chooses tools) |
| Model inference | Configurable Ollama chat and embedding model names |
| Vector store | ChromaDB |
| Structured storage | SQLite |
| REPL UI | prompt_toolkit |
| CLI rendering | Rich |
| Web search | DuckDuckGo |


## Startup and persistence regression tests

These isolated tests exercise real SQLite persistence, first-run directories, the installed CLI,
the workspace boundary and the tool loop (driven by a scripted fake model) without downloading a
model or contacting Ollama. CI installs the full locked environment and runs the whole suite:

```bash
uv sync --group dev
uv run pytest -q
```

### Shell command lifecycle

On macOS and Linux, each `bash` tool call runs in its own process session.
Timeout or task cancellation kills that process group and reaps the direct
process before releasing the command lock. Reaping has a two-second cleanup
budget; an OS-level failure to reap is reported explicitly. Repeated cancellation
cannot interrupt cleanup. Process creation has no async cancellation gap. Cancelling propagates to the caller rather
than returning a success message. Output uses the active locale encoding and retains normal exit-code reporting.
Output is spooled to temporary files so inherited output handles in an escaped
session cannot hold cleanup open; the 30,000-character display cap is not a disk
usage quota. Only the first 120 KB of each captured stream (four bytes per
character of the display cap) is read back and decoded, so a command that writes
gigabytes of output is truncated without loading the whole file into memory; the
temporary files can still consume disk until the command exits or times out.
Processes that deliberately create another session can escape this
group; this is lifecycle cleanup, not a security sandbox. On non-POSIX systems,
cleanup can terminate only the direct process. Tools run with the current user's
file and shell permissions; use a disposable environment for untrusted tasks.

`web_search`'s HTML fallback and `read_url` stream remote bodies instead of
reading them whole: at most 512 KB for the fallback and 2 MB for `read_url`
before HTML cleaning and the 8,000-character visible-text cap. A larger response
is closed early and reported as truncated. Neither tool validates the URL, so
loopback, private-network and metadata-service URLs are reachable from a tool
call; run untrusted content with that in mind.
