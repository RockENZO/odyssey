import asyncio
import os
import shlex
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from odyssey.config import get_config_dir, get_data_dir
from odyssey.core.state import Conversation
from odyssey.core.tool_loop import run_tool_loop
from odyssey.llm.client import get_fast_model
from odyssey.tools.registry import init_tools

console = Console()

style = Style.from_dict({
    "prompt": "ansicyan bold",
})

_bindings = KeyBindings()


def _repl_cmd_help() -> str:
    return """Available commands:

  /exit, /quit    Exit Odyssey
  /help           Show this help
  /clear          Clear conversation history
  /model          Show current model
  /status         Show system status
  /history        Show recent conversation
  /save           Save memories to file

Prefix a command with `!` to run it directly in the shell without AI interpretation.
"""


async def run_repl(cwd: str) -> None:
    init_tools()
    conv = Conversation(working_dir=cwd, project_root=_find_project_root(cwd))
    history_file = get_config_dir() / "history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        style=style,
        enable_history_search=True,
    )

    console.print(Panel.fit(
        "[bold blue]Odyssey[/bold blue] — Your AI Agent\n"
        f"[dim]Model: {get_fast_model()} | Type /help for commands | Ctrl+C to exit[/dim]",
        border_style="blue",
    ))
    console.print()

    try:
        while True:
            try:
                text = await session.prompt_async("> ", vi_mode=True)
            except EOFError:
                break

            if not text or not text.strip():
                continue

            cmd = text.strip()

            if cmd.startswith("/"):
                result = _handle_slash(cmd, conv)
                if result == "exit":
                    break
                if result:
                    console.print(result)
                continue

            if cmd.startswith("!"):
                _run_shell(cmd[1:])
                continue

            console.print(Rule(style="dim"))
            result = await run_tool_loop(cmd, conv)
            console.print()
            console.print(Markdown(result))
            console.print(Rule(style="dim"))

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[dim]Odyssey session ended.[/dim]")


def _handle_slash(cmd: str, conv: Conversation) -> str | None:
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        return "exit"
    elif command == "/help":
        return _repl_cmd_help()
    elif command == "/clear":
        conv.messages.clear()
        return "[dim]Conversation cleared.[/dim]"
    elif command == "/model":
        return f"[dim]Current model: {get_fast_model()}[/dim]"
    elif command == "/status":
        from odyssey.llm.client import check_connection
        connected = asyncio.run(check_connection())
        status = "Connected" if connected else "Disconnected"
        return f"[dim]Ollama: {status} | Model: {get_fast_model()} | Messages: {len(conv.messages)}[/dim]"
    elif command == "/history":
        if not conv.messages:
            return "[dim]No messages yet.[/dim]"
        lines = []
        for i, msg in enumerate(conv.messages[-10:], 1):
            role = msg["role"]
            content = msg.get("content", "")[:80]
            lines.append(f"  {i}. [{role}] {content}")
        return "\n".join(lines)
    elif command == "/save":
        data_dir = get_data_dir()
        save_file = data_dir / f"session_{asyncio.run(_get_timestamp())}.jsonl"
        import json
        with open(save_file, "w") as f:
            for msg in conv.messages:
                f.write(json.dumps(msg) + "\n")
        return f"[dim]Session saved to {save_file}[/dim]"
    return f"[dim]Unknown command: {command}. Type /help for available commands.[/dim]"


async def _get_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_shell(cmd: str) -> None:
    import subprocess
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout or result.stderr or "[no output]"
        if len(output) > 2000:
            output = output[:2000] + "\n... [truncated]"
        console.print(output.strip())
    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _find_project_root(cwd: str) -> str:
    path = Path(cwd).resolve()
    for parent in [path] + list(path.parents):
        if (parent / ".git").is_dir():
            return str(parent)
    return cwd
