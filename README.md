# Odyssey — Personal AI Command Center

A multi-agent AI system that lives in your terminal. Powered entirely by local LLMs via Ollama, with zero cloud dependencies.

## Features

| Command | What it does |
|---|---|
| `ody research <topic>` | Searches the web, reads articles, synthesizes findings |
| `ody journal <entry>` | Writes journal entries with AI sentiment analysis |
| `ody remember <text>` | Saves information to long-term vector memory |
| `ody memories [query]` | Recalls saved information with semantic search |
| `ody tasks list\|add\|done` | Natural language task management |
| `ody briefing` | Daily briefing with tasks, news, and insights |
| `ody chat <message>` | General-purpose AI conversation with tool use |
| `ody status` | Check Ollama connection and model health |

## Architecture

```
CLI (Typer + Rich)
     │
Supervisor Agent (LLM intent classification)
     │
     ├── Research Agent → web_search, read_url, save_memory
     ├── Memory Agent   → ChromaDB vector store + SQLite
     ├── Journal Agent  → sentiment analysis, SQLite
     ├── Task Agent     → CRUD via SQLite
     └── Briefing Agent → aggregates all sources
              │
         Ollama (local LLM)
         qwen2.5:14b (fast)
         qwen3.6:35b (deep)
```

### Data Flow

- **Vector memory**: ChromaDB with Ollama embeddings for semantic search
- **Structured data**: SQLite for tasks, journal entries, research notes
- **Web search**: DuckDuckGo with fallback HTML scraping

## Quick Start

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) running locally
- A model pulled (e.g., `ollama pull qwen2.5:14b`)

### Install

```bash
uv sync
```

### Run

```bash
uv run odyssey
```

## Important: How Odyssey Runs

Odyssey is a **CLI tool, not a background server or daemon**.

- Each command (e.g., `uv run odyssey research "topic"`) is ephemeral — it starts, runs the request through the local LLM, prints output, and **exits immediately**. There is nothing to "stop" or "kill" after a command finishes.
- The only process that stays running is **Ollama** (`ollama serve`), the LLM server that Odyssey talks to. Ollama runs as a background service on your Mac and is what loads the AI models into memory.
- Odyssey itself does not have a "shutdown" command. When you're not actively running an `uv run odyssey ...` command, it consumes zero resources (no CPU, no RAM, no GPU).

To verify no Odyssey processes are running:

```bash
ps aux | grep odyssey  # should show nothing unless a command is in progress
```

The `.venv` directory you see in the project folder is just Python's standard virtual environment for isolating dependencies — it is not a running process.

### Configuration

Config lives at `~/.config/odyssey/config.toml`:

```toml
ollama_host = "http://localhost:11434"
fast_model = "qwen2.5:14b"
deep_model = "qwen3.6:35b-a3b-coding-nvfp4"
embedding_model = "qwen2.5:14b"
```

## Tech Stack

| Component | Choice |
|---|---|
| Agent orchestration | LangGraph (supervisor pattern) |
| Local LLM | Ollama + Qwen 2.5 14B / Qwen 3.6 35B |
| Vector store | ChromaDB |
| Structured storage | SQLite |
| CLI framework | Typer |
| Terminal rendering | Rich |
| Web search | DuckDuckGo |

## Project Structure

```
src/odyssey/
├── cli/          # Typer commands (entry points)
├── core/         # Supervisor agent, state management
├── agents/       # Specialist agents (research, memory, journal, task, briefing)
├── tools/        # Tool functions (web_search, save_memory, etc.)
├── llm/          # Ollama client with structured output
├── storage/      # SQLite + ChromaDB persistence
└── config.py      # Settings management
```
