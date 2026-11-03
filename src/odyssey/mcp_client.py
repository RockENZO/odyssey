"""MCP client: expose chosen tools of configured MCP servers to the graph engine, behind approvals.

Needs ``odyssey[mcp]``. The safety model is deliberately narrow:

* **Only a user-level config is read** (``<config dir>/mcp.json`` or ``--mcp-config PATH``). A
  project directory cannot add servers, because a server is a command that gets run.
* **An explicit allowlist per server.** ``allow`` names the tools the model may see; a server's
  other tools are never registered. There is no wildcard.
* **Every MCP call asks for approval** unless the tool is also listed in ``auto_approve`` (which
  must be a subset of ``allow``). ``--yes`` approves everything, MCP included.
* **A server is launched with a reduced environment** (the SDK's safe defaults plus the ``env``
  of its config entry), not with the whole environment of Odyssey.
* **Results are untrusted text.** An MCP server's output goes to the model like any tool output
  and can try to steer it; approvals are the control, not filtering.
* An MCP tool that is not marked read-only counts as changing the project for the completion
  gate (``--verify``).

Config file::

    {"servers": {"notes": {"command": "uvx", "args": ["some-mcp-server"],
                           "env": {"NOTES_DIR": "/tmp/notes"},
                           "allow": ["search", "add_note"], "auto_approve": ["search"],
                           "timeout": 60}}}

Only the stdio transport is supported. This module is not a sandbox: the server runs with your
permissions.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from odyssey.tools import registry
from odyssey.tools.registry import Tool, ToolOutcome

PREFIX = "mcp__"
_NAME = re.compile(r"^[A-Za-z0-9-]+$")  # no underscore: "__" separates server from tool
_SAFE = re.compile(r"[^A-Za-z0-9_-]")
MAX_NAME = 64
DEFAULT_TIMEOUT = 60.0
_KEYS = {"command", "args", "env", "allow", "auto_approve", "timeout"}


class ConfigError(ValueError):
    """The MCP config file is not valid."""


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    allow: tuple[str, ...] = ()
    auto_approve: tuple[str, ...] = ()
    timeout: float = DEFAULT_TIMEOUT


@dataclass
class ServerStatus:
    name: str
    connected: bool
    tools: list[str] = field(default_factory=list)   # exposed names
    detail: str = ""


@dataclass
class McpRuntime:
    servers: list[ServerStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def tools(self) -> list[str]:
        return [t for s in self.servers for t in s.tools]


def exposed_name(server: str, tool: str) -> str:
    return f"{PREFIX}{server}__{_SAFE.sub('_', tool)}"[:MAX_NAME]


def parse_config(data: Any) -> list[ServerConfig]:
    if not isinstance(data, dict) or set(data) - {"servers"} or not isinstance(data.get("servers"), dict):
        raise ConfigError('the config must be an object with one key, "servers", holding an object of servers')
    configs: list[ServerConfig] = []
    for name, entry in data["servers"].items():
        where = f"server {name!r}"
        if not _NAME.match(name):
            raise ConfigError(f"{where}: the name may only contain letters, digits and '-'")
        if not isinstance(entry, dict) or set(entry) - _KEYS:
            raise ConfigError(f"{where}: expected an object with keys from {sorted(_KEYS)}")
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            raise ConfigError(f'{where}: "command" is required')
        lists = {}
        for key in ("args", "allow", "auto_approve"):
            value = entry.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f'{where}: "{key}" must be a list of strings')
            lists[key] = tuple(value)
        env = entry.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ConfigError(f'{where}: "env" must be an object of strings')
        if "*" in lists["allow"]:
            raise ConfigError(f'{where}: "allow" lists tool names; a wildcard is not supported')
        extra = set(lists["auto_approve"]) - set(lists["allow"])
        if extra:
            raise ConfigError(f'{where}: "auto_approve" must be a subset of "allow" (not allowed: {sorted(extra)})')
        timeout = entry.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ConfigError(f'{where}: "timeout" must be a positive number of seconds')
        configs.append(ServerConfig(name, command, lists["args"], dict(env), lists["allow"], lists["auto_approve"], float(timeout)))
    return configs


def load_config(path: Path) -> list[ServerConfig]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}") from e
    return parse_config(data)


def _text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        parts.append(text if isinstance(text, str) else f"[{getattr(item, 'type', 'unknown')} content omitted]")
    return "\n".join(parts)


def _make_tool(cfg: ServerConfig, remote: Any, session: Any) -> Tool:
    schema = dict(remote.input_schema or {})
    schema.setdefault("type", "object")
    schema.pop("title", None)
    annotations = getattr(remote, "annotations", None)

    async def call(**kwargs: Any) -> ToolOutcome:
        try:
            result = await asyncio.wait_for(session.call_tool(remote.name, kwargs), cfg.timeout)
        except asyncio.TimeoutError:
            return ToolOutcome(f"Error: MCP tool {remote.name} on {cfg.name!r} did not answer within {cfg.timeout:g}s", False)
        except Exception as e:  # a dead server or a protocol error is a failed call, not a crash of the session
            return ToolOutcome(f"Error: MCP tool {remote.name} on {cfg.name!r} failed: {type(e).__name__}: {e}", False)
        text = _text(result)
        if getattr(result, "is_error", False):
            return ToolOutcome(f"Error: {text or 'the MCP server reported an error'}", False)
        return ToolOutcome(text, True)

    return Tool(
        name=exposed_name(cfg.name, remote.name),
        description=f"[MCP server {cfg.name!r}] {remote.description or remote.title or remote.name}",
        fn=call,
        parameters=schema,
        read_only=bool(annotations and getattr(annotations, "read_only_hint", False)),
    )


async def _list_all(session: Any) -> list[Any]:
    from mcp import types

    tools: list[Any] = []
    cursor = None
    for _ in range(50):  # a server that never stops paging must not hang start-up
        page = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor) if cursor else None)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            break
    return tools


@contextlib.asynccontextmanager
async def connect(configs: list[ServerConfig], policy: Any, log_dir: Path) -> AsyncIterator[McpRuntime]:
    """Start the servers, register their allowlisted tools, and unregister and stop them on exit.

    A server that fails to start is reported in ``warnings`` and skipped; the others still load.
    The tools that are not in ``auto_approve`` are added to ``policy.ask_tools``.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise RuntimeError("MCP needs the client library: pip install 'odyssey[mcp]'") from e

    runtime = McpRuntime()
    registered: list[str] = []
    async with AsyncExitStack() as stack:
        log_dir.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            status = ServerStatus(cfg.name, False)
            runtime.servers.append(status)
            if not cfg.allow:
                status.detail = 'not started: "allow" is empty, so no tool would be exposed'
                runtime.warnings.append(f"{cfg.name}: {status.detail}")
                continue
            server_stack = AsyncExitStack()
            try:
                errlog = server_stack.enter_context(open(log_dir / f"mcp-{cfg.name}.log", "a", encoding="utf-8"))
                params = StdioServerParameters(command=cfg.command, args=list(cfg.args), env=cfg.env or None)
                read, write = await server_stack.enter_async_context(stdio_client(params, errlog=errlog))
                session = await server_stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), cfg.timeout)
                remote_tools = await asyncio.wait_for(_list_all(session), cfg.timeout)
            except Exception as e:
                with contextlib.suppress(Exception):
                    await server_stack.aclose()
                status.detail = f"could not start: {type(e).__name__}: {e} (server log: {log_dir / f'mcp-{cfg.name}.log'})"
                runtime.warnings.append(f"{cfg.name}: {status.detail}")
                continue
            stack.push_async_callback(server_stack.aclose)
            by_name = {t.name: t for t in remote_tools}
            for missing in sorted(set(cfg.allow) - set(by_name)):
                runtime.warnings.append(f"{cfg.name}: allowed tool {missing!r} is not offered by the server")
            for tool_name in cfg.allow:
                remote = by_name.get(tool_name)
                if remote is None:
                    continue
                tool = _make_tool(cfg, remote, session)
                if tool.name in registry._tools:
                    runtime.warnings.append(f"{cfg.name}: {tool_name!r} would be exposed as {tool.name!r}, which is already taken; skipped")
                    continue
                registry.register(tool)
                registered.append(tool.name)
                status.tools.append(tool.name)
                if tool_name not in cfg.auto_approve:
                    policy.ask_tools.add(tool.name)
            status.connected = True
        try:
            yield runtime
        finally:
            for name in registered:
                registry.unregister(name)
                policy.ask_tools.discard(name)
