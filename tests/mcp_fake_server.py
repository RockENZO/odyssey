"""A tiny stdio MCP server for the tests (not a test file)."""
import asyncio
import sys
from pathlib import Path

from mcp import types
from mcp.server.mcpserver import MCPServer

server = MCPServer("fake")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None


@server.tool(annotations=types.ToolAnnotations(read_only_hint=True))
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def save_note(text: str) -> str:
    """Write a note (changes something outside the conversation)."""
    if OUT is not None:
        OUT.write_text(text, encoding="utf-8")
    return "saved"


@server.tool()
def secret() -> str:
    """A tool the config does not allow."""
    return "never shown to the model"


@server.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


@server.tool()
async def slow() -> str:
    """Takes longer than the test timeout."""
    await asyncio.sleep(30)
    return "late"


if __name__ == "__main__":
    server.run("stdio")
