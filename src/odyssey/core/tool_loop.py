import time
from typing import Any, Awaitable, Callable

from rich.markdown import Markdown
from rich.console import Console

from odyssey.llm.client import chat_completion
from odyssey.core.state import Conversation, ToolRecord
from odyssey.core.tool_calls import ToolCall, parse_tool_calls
from odyssey.tools.registry import get_tool
from odyssey.config import get_data_dir

console = Console()

SYSTEM_PROMPT = """You are Odyssey, a versatile AI agent that lives in the user's terminal.
You have access to tools that let you read and write files, run shell commands, search the web, and more.

Guidelines:
- Use tools to fulfill the user's request step by step
- When you need to explore a codebase, use Glob and Grep
- For file operations, prefer Read, Write, and Edit over Bash
- Use Bash for running commands, git operations, builds, and tests
- When writing files, use Write for new files and Edit for modifications
- Use web_search and read_url for research tasks
- Use save_memory and query_memory for persistent knowledge
- Use add_task, list_tasks, complete_task for task management
- Use save_journal and get_journal_summary for journaling
- Use todo_write to track multi-step tasks
- After executing a tool, analyze the result and decide next steps
- When you're done, provide a clear summary of what you did
- Be concise but thorough in your responses
"""


ChatFn = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]

MAX_TOOL_RESULT_CHARS = 10000


async def execute_tool_call(call: ToolCall, conv: Conversation) -> None:
    """Run one tool call and record the result, with the call's id, on the conversation."""
    tool = get_tool(call.name)
    if tool is None:
        result, ok, duration = f"Error: tool '{call.name}' not found", False, 0.0
    else:
        _show_tool_call(call.name, call.arguments)
        started = time.monotonic()
        outcome = await tool.execute(**call.arguments)
        result, ok = outcome.text, outcome.ok
        duration = time.monotonic() - started
    conv.add_tool_result(call.name, result[:MAX_TOOL_RESULT_CHARS], tool_call_id=call.id)
    conv.tool_log.append(ToolRecord(call.id, call.name, call.arguments, result, ok, duration))


async def run_tool_loop(user_input: str, conv: Conversation, chat: ChatFn | None = None) -> str:
    """Run the model/tool loop until the model answers without calling a tool.

    ``chat`` replaces the model call; tests pass a scripted function so the loop runs
    without Ollama.
    """
    chat = chat or chat_completion
    conv.add_message("user", user_input)
    messages = _build_messages(conv)

    for _ in range(conv.max_tool_rounds):
        response = await chat(messages)
        msg = response["message"]
        content = msg.get("content", "") or ""
        calls = parse_tool_calls(msg.get("tool_calls"))

        if not calls:
            conv.add_message("assistant", content)
            return content

        conv.add_message("assistant", "", tool_calls=calls)
        for call in calls:
            await execute_tool_call(call, conv)
        messages = _build_messages(conv)

    timeout_msg = "Reached maximum tool call iterations. Try breaking your request into smaller steps."
    conv.add_message("assistant", timeout_msg)
    return timeout_msg


def _build_messages(conv: Conversation) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conv.messages)
    return messages


def _show_tool_call(name: str, args: dict[str, Any]) -> None:
    args_str = " ".join(f"{k}={v!r}" for k, v in args.items())
    console.print(f"  [dim]→ {name} {args_str}[/dim]")
