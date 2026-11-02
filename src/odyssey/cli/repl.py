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
from odyssey.core.workspace import ALLOW_OUTSIDE_ENV, set_workspace_root
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

Graph engine only (`odyssey --engine graph`):
  /approve [always]   Run the tool call that is waiting for approval
  /reject [reason]    Refuse it; the model is told why
  /resume             Continue a turn that was interrupted
  /plan               Show the structured plan (todo_write)
  /thread             Show this session's thread id
  /cassette [path]    Save this thread's tool calls as a trajectory-replay cassette (needs odyssey[eval])

Prefix a command with `!` to run it directly in the shell without AI interpretation.
"""


async def run_repl(cwd: str, engine: str = "loop", resume: bool = False, approve_all: bool = False, verify: str = "") -> None:
    init_tools()
    project_root = _find_project_root(cwd)
    conv = Conversation(working_dir=cwd, project_root=project_root)
    if os.environ.get(ALLOW_OUTSIDE_ENV) == "1":
        set_workspace_root(None)
    else:
        set_workspace_root(project_root, base=cwd)
    from odyssey.tools import bash as bash_tool
    bash_tool._working_dir = cwd

    console.print(Panel.fit(
        "[bold blue]Odyssey[/bold blue] — Your AI Agent\n"
        f"[dim]Model: {get_fast_model()} | Type /help for commands | Ctrl+C to exit[/dim]\n"
        f"[dim]Workspace: {project_root if os.environ.get(ALLOW_OUTSIDE_ENV) != '1' else 'unrestricted'} (bash is not sandboxed)[/dim]",
        border_style="blue",
    ))
    console.print()

    if engine == "graph":
        from odyssey.engine.session import default_db_path, open_checkpointer
        async with open_checkpointer(default_db_path()) as saver:
            gs = await _start_graph_session(saver, resume, approve_all, project_root, cwd, verify)
            await _sync_conv(conv, gs)
            await _repl_loop(conv, gs)
    else:
        await _repl_loop(conv, None)


async def _start_graph_session(saver, resume: bool, approve_all: bool, workspace: str = "", base: str = "", verify: str = ""):
    from odyssey.engine.graph import build_graph
    from odyssey.engine.policy import ApprovalPolicy
    from odyssey.engine.session import GraphSession, new_thread_id, recall_thread, remember_thread
    from odyssey.llm.client import chat_completion

    base = base or workspace
    thread_id = new_thread_id()
    policy = ApprovalPolicy(approve_all=approve_all)
    acceptance = None
    if verify:
        import shlex

        from odyssey.engine.session import default_db_path
        from odyssey.engine.verify import ledger_gate
        acceptance = ledger_gate(shlex.split(verify), ledger_path=default_db_path().with_name("ledger.sqlite"))
    graph = build_graph(chat_completion, checkpointer=saver, policy=policy, acceptance=acceptance)
    if resume:
        # The pointer is per launch directory, so starting here never overwrites the thread of another project.
        previous = recall_thread(base)
        if previous:
            values = (await graph.aget_state({"configurable": {"thread_id": previous}})).values
            recorded = values.get("base") or values.get("workspace", "")
            if recorded and recorded != base:
                console.print(f"[yellow]The last thread here was started in {recorded}, not {base}. Starting a new thread.[/yellow]")
            else:
                thread_id = previous
        else:
            console.print("[dim]No earlier thread was started from this directory; starting a new one.[/dim]")
    remember_thread(base, thread_id)
    gs = GraphSession(graph, thread_id, policy, workspace=workspace, base=base)
    console.print(f"[dim]Graph engine | thread {thread_id} | approval needed for: "
                  f"{'nothing (--yes)' if approve_all else ', '.join(sorted(policy.ask_tools))}[/dim]")
    if verify:
        console.print(f"[dim]Completion check: {verify} (recorded in the ledger)[/dim]")
    snapshot = await gs.state()
    if resume and snapshot:
        pending = await gs.pending_approval()
        if pending:
            console.print("[yellow]This thread is waiting for approval:[/yellow]")
            _show_approval(pending)
        elif (await gs._snapshot()).next:
            console.print("[yellow]The last turn did not finish. Type /resume to continue it.[/yellow]")
    return gs


def _show_approval(calls: list[dict]) -> None:
    for call in calls:
        args = " ".join(f"{k}={str(v)[:200]!r}" for k, v in call["arguments"].items())
        console.print(f"  [bold]{call['name']}[/bold] {args}  [dim]({call['id']})[/dim]")
    console.print("[dim]/approve [always] to run it, /reject [reason] to refuse it.[/dim]")


def _show_turn(result) -> None:
    if result.resumed_in_flight:
        call = result.resumed_in_flight[0]
        console.print(f"[yellow]Resumed. The call {call['name']} ({call['id']}) may already have run before the interruption.[/yellow]")
    if result.status == "awaiting_approval":
        console.print("[yellow]Waiting for approval:[/yellow]")
        _show_approval(result.approval)
        return
    console.print()
    console.print(Markdown(result.text or "(no answer)"))
    reason = (result.verification or {}).get("reason", "")
    if result.status == "verified":
        console.print(f"[green]Verified:[/green] {reason.strip().splitlines()[-1] if reason.strip() else 'check passed'}")
    elif result.status == "unverified":
        console.print(f"[yellow]Not verified[/yellow] (the check still fails after the repair rounds):\n{reason[-1500:]}")
    elif result.status == "budget_exhausted":
        console.print(f"[yellow]Status: {result.status}[/yellow]")
    console.print(Rule(style="dim"))


async def _handle_graph_slash(cmd: str, gs, conv: Conversation) -> bool:
    """Handle the graph-only commands. Returns True when the command was one of them."""
    parts = cmd.split(maxsplit=1)
    command, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
    if command not in ("/approve", "/reject", "/resume", "/plan", "/thread", "/clear", "/cassette"):
        return False
    try:
        if command == "/clear":
            gs.new_thread()
            from odyssey.engine.session import remember_thread
            remember_thread(gs.base or gs.workspace, gs.thread_id)
            conv.messages.clear()
            console.print(f"[dim]Conversation cleared. New thread {gs.thread_id}.[/dim]")
            return True
        if command == "/approve":
            _show_turn(await gs.decide(True, always=arg.strip() == "always"))
        elif command == "/reject":
            _show_turn(await gs.decide(False, reason=arg.strip()))
        elif command == "/resume":
            _show_turn(await gs.resume())
        elif command == "/cassette":
            from odyssey.config import get_data_dir
            from odyssey.engine.cassette import save_cassette
            target = Path(arg.strip()).expanduser() if arg.strip() else get_data_dir() / "cassettes" / f"{gs.thread_id}.json"
            count = save_cassette(await gs.state(), target)
            console.print(f"[dim]Saved {count} tool calls to {target}[/dim]")
        elif command == "/plan":
            from odyssey.engine.todos import format_todos
            console.print(format_todos((await gs.state()).get("todos", [])))
        else:
            console.print(f"[dim]Thread {gs.thread_id}. Restart with `odyssey --engine graph --resume` to continue it.[/dim]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
    await _sync_conv(conv, gs)
    return True


async def _sync_conv(conv: Conversation, gs) -> None:
    """Mirror the graph thread's messages into ``conv`` so /history, /save and /status see them."""
    conv.messages[:] = (await gs.state()).get("messages", [])


async def _repl_loop(conv: Conversation, gs) -> None:
    history_file = get_config_dir() / "history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    prompt: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        style=style,
        enable_history_search=True,
    )
    try:
        while True:
            try:
                text = await prompt.prompt_async("> ", vi_mode=True)
            except EOFError:
                break

            if not text or not text.strip():
                continue

            cmd = text.strip()

            if cmd.startswith("/"):
                if gs is not None and await _handle_graph_slash(cmd, gs, conv):
                    continue
                result = await _handle_slash(cmd, conv)
                if result == "exit":
                    break
                if result:
                    console.print(result)
                continue

            if cmd.startswith("!"):
                _run_shell(cmd[1:])
                continue

            console.print(Rule(style="dim"))
            if gs is not None:
                try:
                    _show_turn(await gs.submit(cmd))
                except RuntimeError as e:
                    console.print(f"[red]{e}[/red]")
                await _sync_conv(conv, gs)
                continue
            result = await run_tool_loop(cmd, conv)
            console.print()
            console.print(Markdown(result))
            console.print(Rule(style="dim"))

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[dim]Odyssey session ended.[/dim]")


async def _handle_slash(cmd: str, conv: Conversation) -> str | None:
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
        connected = await check_connection()
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
        save_file = data_dir / f"session_{_get_timestamp()}.jsonl"
        import json
        with open(save_file, "w") as f:
            for msg in conv.messages:
                f.write(json.dumps(msg) + "\n")
        return f"[dim]Session saved to {save_file}[/dim]"
    return f"[dim]Unknown command: {command}. Type /help for available commands.[/dim]"


def _get_timestamp() -> str:
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
        if (parent / ".git").exists():  # a directory, or a file in worktrees and submodules
            return str(parent)
    return cwd
