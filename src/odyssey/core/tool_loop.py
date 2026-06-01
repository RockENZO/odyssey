from typing import Any

from rich.markdown import Markdown
from rich.console import Console

from odyssey.llm.client import chat_completion
from odyssey.core.state import Conversation
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


async def run_tool_loop(user_input: str, conv: Conversation) -> str:
    conv.add_message("user", user_input)
    messages = _build_messages(conv)

    for _ in range(conv.max_tool_rounds):
        response = await chat_completion(messages)
        msg = response["message"]
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            conv.add_message("assistant", content)
            return content

        if content:
            content = ""
        conv.add_message("assistant", content, tool_calls=tool_calls)

        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
            else:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                if hasattr(fn_args, "model_dump"):
                    fn_args = fn_args.model_dump()

            if isinstance(fn_args, str):
                import json
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            tool = get_tool(fn_name)
            if tool:
                _show_tool_call(fn_name, fn_args)
                result = await tool.run(**fn_args)
                conv.add_tool_result(tool.name, result[:10000])
            else:
                conv.add_tool_result(fn_name, f"Error: tool '{fn_name}' not found")

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
