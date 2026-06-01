import asyncio
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*")

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from odyssey import __version__
from odyssey.llm.client import check_connection, get_fast_model, get_deep_model
from odyssey.tools.registry import init_tools

warnings.filterwarnings("ignore", category=ResourceWarning)

app = typer.Typer(
    name="odyssey",
    help="Your personal AI command center — research, journal, tasks, memory, and daily briefings.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _ensure_tools():
    init_tools()


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    _ensure_tools()
    console.print(Panel.fit(
        "[bold blue]Odyssey[/bold blue] — Your Personal AI Command Center\n"
        f"[dim]v{__version__} | Local LLM: {get_fast_model()}[/dim]",
        border_style="blue",
    ))
    console.print("\n[bold]Commands:[/bold]")
    console.print("  [cyan]ody research[/cyan]   Research a topic")
    console.print("  [cyan]ody journal[/cyan]    Write a journal entry")
    console.print("  [cyan]ody tasks[/cyan]     Manage your tasks")
    console.print("  [cyan]ody memory[/cyan]    Save/recall information")
    console.print("  [cyan]ody briefing[/cyan]  Get your daily briefing")
    console.print("  [cyan]ody chat[/cyan]      Talk to Odyssey freely")
    console.print("  [cyan]ody status[/cyan]    Check system status")
    console.print("\nRun [bold]ody --help[/bold] for more details.\n")


@app.command()
def chat(
    message: str = typer.Argument(..., help="What would you like to say?"),
):
    _ensure_tools()
    from odyssey.core.supervisor import run_agent
    result = _run(run_agent(message))
    console.print(Markdown(result.final_output))


@app.command()
def research(
    topic: str = typer.Argument(..., help="What would you like to research?"),
):
    _ensure_tools()
    from odyssey.agents.research_agent import research as research_fn
    console.print(f"[bold blue]Researching: {topic}...[/bold blue]")
    result = _run(research_fn(topic))
    console.print(Markdown(result))


@app.command()
def journal(
    content: str = typer.Argument(..., help="Your journal entry text"),
):
    _ensure_tools()
    from odyssey.agents.journal_agent import handle_journal
    console.print("[bold blue]Reflecting on your entry...[/bold blue]")
    result = _run(handle_journal(content))
    console.print(Markdown(result))


@app.command()
def memories(
    query: Optional[str] = typer.Argument(None, help="Search your memory (optional)"),
):
    _ensure_tools()
    from odyssey.tools.registry import get_tool
    if query:
        console.print("[bold blue]Searching memory...[/bold blue]")
        tool = get_tool("query_memory")
        result = _run(tool.run(query=query))
    else:
        from odyssey.storage.db import list_memories as list_mem
        entries = list_mem(limit=10)
        if not entries:
            result = "No memories saved yet. Use [bold]ody remember[/bold] to save something."
        else:
            lines = ["## Recent Memories\n"]
            for e in entries:
                lines.append(f"- {e['content'][:200]}")
                if e.get("tags"):
                    lines.append(f"  *Tags: {e['tags']}*")
                lines.append("")
            result = "\n".join(lines)
    console.print(Markdown(result))


@app.command()
def remember(
    content: str = typer.Argument(..., help="What to remember"),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags"),
):
    _ensure_tools()
    from odyssey.tools.registry import get_tool
    console.print("[bold blue]Saving to memory...[/bold blue]")
    tool = get_tool("save_memory")
    result = _run(tool.run(content=content, tags=tags))
    console.print(Markdown(result))


@app.command()
def tasks(
    action: str = typer.Argument("list", help="Action: list, add, done"),
    args: list[str] = typer.Argument(None, help="Task details"),
):
    _ensure_tools()
    from odyssey.tools.registry import get_tool

    if action == "list":
        result = _run(get_tool("list_tasks").run(status="all"))
    elif action == "add":
        if not args:
            console.print("[red]Provide task title[/red]")
            raise typer.Exit(1)
        title = " ".join(args)
        result = _run(get_tool("add_task").run(title=title))
    elif action == "done":
        if not args:
            console.print("[red]Provide task ID to mark done[/red]")
            raise typer.Exit(1)
        result = _run(get_tool("complete_task").run(task_id=int(args[0])))
    else:
        result = "Unknown action. Use: list, add, or done."

    console.print(Markdown(result))


@app.command()
def briefing():
    _ensure_tools()
    from odyssey.agents.briefing_agent import generate_briefing
    console.print("[bold blue]Preparing your briefing...[/bold blue]")
    result = _run(generate_briefing())
    console.print(Markdown(result))


@app.command()
def status():
    _ensure_tools()
    connected = _run(check_connection())
    table = Table(title="Odyssey Status")
    table.add_column("Component", style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    table.add_row(
        "Ollama",
        "✅ Connected" if connected else "❌ Disconnected",
        get_fast_model(),
    )
    table.add_row(
        "Fast Model",
        "✅ Ready" if connected else "⚠️ Unavailable",
        get_fast_model(),
    )
    table.add_row(
        "Deep Model",
        "Available",
        get_deep_model(),
    )
    table.add_row(
        "Data Dir",
        "Ready",
        str(_get_data_dir()),
    )
    table.add_row(
        "Version",
        "✅",
        f"v{__version__}",
    )
    console.print(table)


def _get_data_dir():
    from odyssey.config import get_data_dir
    return get_data_dir()


if __name__ == "__main__":
    app()
