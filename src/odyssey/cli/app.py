import asyncio
import os
import sys
import warnings

import typer
from rich.console import Console

from odyssey import __version__
from odyssey.llm.client import check_connection, get_fast_model
from odyssey.cli.repl import run_repl

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
):
    if version:
        console.print(f"Odyssey v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    _launch_repl()


@app.command()
def repl():
    _launch_repl()


@app.command()
def status():
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


def _launch_repl():
    cwd = os.getcwd()
    asyncio.run(run_repl(cwd))


if __name__ == "__main__":
    app()
