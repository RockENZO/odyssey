from __future__ import annotations

from typing import Any, Callable, Awaitable

from odyssey.tools import web, memory, task_manager, journal_tools
from odyssey.tools import bash as bash_mod, file_ops, search_tools, sub_agent


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
        name="bash",
        description="Execute a shell command in a persistent bash session. Use for running commands, git operations, builds, tests, and scripts. The working directory is preserved between calls.",
        fn=bash_mod.bash,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
            },
            "required": ["command"],
        },
    ))
    register(Tool(
        name="read",
        description="Read the contents of a file. Returns content with line numbers. Use offset and limit to read specific sections of large files.",
        fn=file_ops.read_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Starting line number (0-indexed, default 0)"},
                "limit": {"type": "integer", "description": "Max lines to read (default 2000)"},
            },
            "required": ["path"],
        },
    ))
    register(Tool(
        name="write",
        description="Create a new file or overwrite an existing one with the given content. Creates parent directories if needed.",
        fn=file_ops.write_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path where to write the file"},
                "content": {"type": "string", "description": "Full content to write to the file"},
            },
            "required": ["path", "content"],
        },
    ))
    register(Tool(
        name="edit",
        description="Edit a file by replacing an exact string match with new content. Use this for targeted modifications without rewriting the entire file. The old_string must match exactly and uniquely.",
        fn=file_ops.edit_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "Exact text to find and replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ))
    register(Tool(
        name="glob",
        description="Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts'). Supports recursive matching with **.",
        fn=search_tools.glob,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to search (e.g., '**/*.py')"},
                "path": {"type": "string", "description": "Base directory to search (default: current dir)"},
            },
            "required": ["pattern"],
        },
    ))
    register(Tool(
        name="grep",
        description="Search file contents using a regex pattern. Uses ripgrep (rg) for fast searching. Returns file paths and line numbers of matches.",
        fn=search_tools.grep,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search (default: current dir)"},
                "include": {"type": "string", "description": "File extension filter (e.g., 'py', 'js', 'ts')"},
            },
            "required": ["pattern"],
        },
    ))
    register(Tool(
        name="web_search",
        description="Search the web for information on a given query. Returns a list of results with titles, URLs, and snippets.",
        fn=web.web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results (default 5)"},
            },
            "required": ["query"],
        },
    ))
    register(Tool(
        name="read_url",
        description="Read and extract the main content of a URL. Returns the page content as text, stripped of HTML.",
        fn=web.read_url,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read"},
            },
            "required": ["url"],
        },
    ))
    register(Tool(
        name="save_memory",
        description="Save a piece of information to persistent long-term memory with vector search. The content will be retrievable later via query_memory.",
        fn=memory.save_memory,
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember"},
                "tags": {"type": "string", "description": "Comma-separated tags for categorization"},
            },
            "required": ["content"],
        },
    ))
    register(Tool(
        name="query_memory",
        description="Search your long-term memory using semantic search. Finds related information saved via save_memory.",
        fn=memory.query_memory,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in your memory"},
            },
            "required": ["query"],
        },
    ))
    register(Tool(
        name="add_task",
        description="Add a new task to your persistent task list.",
        fn=task_manager.add_task_tool,
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The task title"},
                "priority": {"type": "string", "description": "high, medium, or low", "enum": ["high", "medium", "low"]},
                "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
            },
            "required": ["title"],
        },
    ))
    register(Tool(
        name="list_tasks",
        description="List tasks, optionally filtered by status.",
        fn=task_manager.list_tasks_tool,
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "pending, done, or all", "enum": ["pending", "done", "all"]},
            },
        },
    ))
    register(Tool(
        name="complete_task",
        description="Mark a task as completed by its ID.",
        fn=task_manager.complete_task_tool,
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "The ID of the task to complete"},
            },
            "required": ["task_id"],
        },
    ))
    register(Tool(
        name="save_journal",
        description="Save a journal entry with AI-powered reflection and sentiment analysis.",
        fn=journal_tools.save_journal_tool,
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Your journal entry content"},
            },
            "required": ["content"],
        },
    ))
    register(Tool(
        name="get_journal_summary",
        description="Get a summary of recent journal entries.",
        fn=journal_tools.get_journal_summary_tool,
        parameters={
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (default 7)"},
            },
        },
    ))
    register(Tool(
        name="task",
        description="Spawn a sub-agent to handle a complex task. The sub-agent has its own context and tool access. Use this for parallel or independent work.",
        fn=sub_agent.task,
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed description of the task for the sub-agent"},
            },
            "required": ["description"],
        },
    ))
    register(Tool(
        name="todo_write",
        description="Track multi-step progress by creating/updating a structured task list. Use this for complex multi-step operations.",
        fn=_todo_write,
        parameters={
            "type": "object",
            "properties": {
                "items": {"type": "string", "description": "List of tasks with status markers, one per line"},
            },
            "required": ["items"],
        },
    ))


async def _todo_write(items: str) -> str:
    return f"Tasks tracked:\n{items}"
