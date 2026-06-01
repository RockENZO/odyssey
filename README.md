# Odyssey — Versatile AI Agent for Your Terminal

A local-first AI agent that runs in your terminal. Powered by local LLMs via Ollama with zero cloud dependencies. Think Claude Code but self-hosted and extensible.

## Features

- **Interactive REPL** — `uv run odyssey` drops you into a continuous `> ` prompt. Chat back and forth, the AI uses tools autonomously.
- **17 tools** — File operations, bash, search, web research, memory, task management, journaling, sub-agents.
- **Fully local** — Runs on your machine via Ollama. No data leaves your computer.
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
     qwen2.5:14b (fast) / qwen3.6:35b (deep)
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

## Quick Start

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) running locally
- A model pulled: `ollama pull qwen2.5:14b`

### Install & Run

```bash
# Run directly (no install needed)
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

`~/.config/odyssey/config.toml`:

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
4. When you `/exit`, the process ends — zero resource usage

The only persistent background process is **Ollama** itself (`ollama serve`), which keeps the LLM model loaded in memory.

## Project Structure

```
src/odyssey/
├── cli/           # REPL entry point (prompt_toolkit)
├── core/          # Tool-calling loop, conversation state
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
| Local LLM | Ollama + Qwen 2.5 14B / Qwen 3.6 35B |
| Vector store | ChromaDB |
| Structured storage | SQLite |
| REPL UI | prompt_toolkit |
| CLI rendering | Rich |
| Web search | DuckDuckGo |
