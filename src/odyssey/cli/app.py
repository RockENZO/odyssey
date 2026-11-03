import asyncio
import os
import sys
import warnings
from pathlib import Path

import typer
from rich.console import Console

from odyssey import __version__

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*")

app = typer.Typer(
    name="odyssey",
    help="Your versatile AI agent — coding, research, journaling, tasks, and more.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
    engine: str = typer.Option("loop", "--engine", help="Execution engine: 'loop' (default) or 'graph' (LangGraph, resumable, asks before bash/write/edit)"),
    resume: bool = typer.Option(False, "--resume", help="Graph engine: continue the last thread"),
    yes: bool = typer.Option(False, "--yes", help="Graph engine: do not ask for approval"),
    verify: str = typer.Option("", "--verify", help="Graph engine: command that must pass before a turn that changed files counts as done, e.g. \"pytest -q\" (needs odyssey[verify])"),
    mcp_config: str = typer.Option("", "--mcp-config", help="Graph engine: MCP servers file (default: mcp.json in the config directory, see `odyssey config-dir`; needs odyssey[mcp])"),
):
    if version:
        console.print(f"Odyssey v{__version__}")
        raise typer.Exit()
    # Options given before the subcommand (`odyssey --engine graph repl`) must reach it.
    ctx.obj = {"engine": engine, "resume": resume, "yes": yes, "verify": verify, "mcp_config": mcp_config}
    if ctx.invoked_subcommand is not None:
        return
    _launch_repl(engine, resume, yes, verify, mcp_config)


@app.command()
def repl(
    ctx: typer.Context,
    engine: str | None = typer.Option(None, "--engine", help="'loop' or 'graph'"),
    resume: bool = typer.Option(False, "--resume"),
    yes: bool = typer.Option(False, "--yes"),
    verify: str = typer.Option("", "--verify"),
    mcp_config: str = typer.Option("", "--mcp-config"),
):
    root = ctx.obj or {}
    _launch_repl(engine or root.get("engine", "loop"), resume or root.get("resume", False), yes or root.get("yes", False), verify or root.get("verify", ""), mcp_config or root.get("mcp_config", ""))


@app.command()
def status():
    from odyssey.llm.client import check_connection, get_fast_model
    connected = asyncio.run(check_connection())
    from rich.table import Table
    table = Table(title="Odyssey Status")
    table.add_column("Component", style="bold cyan")
    table.add_column("Status")
    table.add_column("Details")
    table.add_row("Ollama", "✅ Connected" if connected else "❌ Disconnected", get_fast_model())
    table.add_row("Version", "✅", f"v{__version__}")
    from odyssey.config import get_data_dir
    table.add_row("Data Dir", "Ready", str(get_data_dir()))
    console.print(table)


@app.command("config-path")
def config_path():
    """Print the configuration directory without contacting Ollama."""
    from odyssey.config import get_config_dir
    typer.echo(str(get_config_dir()))


def _launch_repl(engine: str = "loop", resume: bool = False, yes: bool = False, verify: str = "", mcp_config: str = ""):
    if engine not in ("loop", "graph"):
        raise typer.BadParameter("must be 'loop' or 'graph'", param_hint="--engine")
    if verify and engine != "graph":
        raise typer.BadParameter("only the graph engine can check completion (use --engine graph)", param_hint="--verify")
    if verify:
        try:
            import agent_ledger  # noqa: F401
        except ImportError:
            raise typer.BadParameter("needs the ledger: pip install 'odyssey[verify]'", param_hint="--verify")
    if mcp_config:
        if engine != "graph":
            raise typer.BadParameter("MCP tools need the approval step of the graph engine (use --engine graph)", param_hint="--mcp-config")
        try:
            from odyssey.mcp_client import ConfigError, load_config
            load_config(Path(mcp_config).expanduser())
        except ConfigError as e:
            raise typer.BadParameter(str(e), param_hint="--mcp-config")
    from odyssey.cli.repl import run_repl
    cwd = os.getcwd()
    asyncio.run(run_repl(cwd, engine, resume, yes, verify, mcp_config))


if __name__ == "__main__":
    app()
