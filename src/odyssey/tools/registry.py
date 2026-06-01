from __future__ import annotations

import inspect
from typing import Any, Callable, Awaitable

from rich.console import Console

from odyssey.tools import web, memory, task_manager, journal_tools

console = Console()


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Awaitable[str]],
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters or {}

    async def run(self, **kwargs: Any) -> str:
        try:
            return await self.fn(**kwargs)
        except Exception as e:
            return f"Error executing {self.name}: {e}"

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {},
            },
        }


_tools: dict[str, Tool] = {}


def register(t: Tool) -> None:
    _tools[t.name] = t


def get_tool(name: str) -> Tool | None:
    return _tools.get(name)


def get_all_tools() -> list[Tool]:
    return list(_tools.values())


def get_tool_schemas() -> list[dict[str, Any]]:
    return [t.to_openai_tool() for t in _tools.values()]


def init_tools() -> None:
    register(Tool(
        name="web_search",
        description="Search the web for information on a given query. Returns a list of results with titles and snippets.",
        fn=web.web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                },
            },
            "required": ["query"],
        },
    ))
    register(Tool(
        name="read_url",
        description="Read and extract the main content of a URL. Returns the page content as text.",
        fn=web.read_url,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to read",
                },
            },
            "required": ["url"],
        },
    ))
    register(Tool(
        name="save_memory",
        description="Save a piece of information to your long-term memory. This will persist and can be retrieved later.",
        fn=memory.save_memory,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated tags for categorization",
                },
            },
            "required": ["content"],
        },
    ))
    register(Tool(
        name="query_memory",
        description="Search your long-term memory for information. Useful for recalling past notes, research, or journal entries.",
        fn=memory.query_memory,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in your memory",
                },
            },
            "required": ["query"],
        },
    ))
    register(Tool(
        name="add_task",
        description="Add a new task to your task list. Supports natural language dates and priorities.",
        fn=task_manager.add_task_tool,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The task title or description",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level: high, medium, or low",
                    "enum": ["high", "medium", "low"],
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format, or natural language like 'tomorrow'",
                },
            },
            "required": ["title"],
        },
    ))
    register(Tool(
        name="list_tasks",
        description="List your current tasks. Optionally filter by status.",
        fn=task_manager.list_tasks_tool,
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: pending, done, or all",
                    "enum": ["pending", "done", "all"],
                },
            },
        },
    ))
    register(Tool(
        name="complete_task",
        description="Mark a task as completed.",
        fn=task_manager.complete_task_tool,
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The ID of the task to complete",
                },
            },
            "required": ["task_id"],
        },
    ))
    register(Tool(
        name="save_journal",
        description="Save a journal entry with AI-powered reflection.",
        fn=journal_tools.save_journal_tool,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Your journal entry content",
                },
            },
            "required": ["content"],
        },
    ))
    register(Tool(
        name="get_journal_summary",
        description="Get a summary of your recent journal entries.",
        fn=journal_tools.get_journal_summary_tool,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (default 7)",
                },
            },
        },
    ))
