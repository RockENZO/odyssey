"""Which tool calls need a person's approval before they run."""
from __future__ import annotations

from typing import Any

# Tools that change files or run arbitrary commands.
DEFAULT_ASK_TOOLS = frozenset({"bash", "write", "edit"})


class ApprovalPolicy:
    """Decides per tool call whether to pause for approval.

    ``allow(tool)`` records an "always approve this tool" choice for the life of the
    object. It is not persisted: after a restart the default applies again.
    """

    def __init__(self, ask_tools: frozenset[str] | set[str] = DEFAULT_ASK_TOOLS, approve_all: bool = False):
        self.ask_tools = set(ask_tools)
        self.approve_all = approve_all

    def needs_approval(self, call: dict[str, Any]) -> bool:
        return not self.approve_all and call["name"] in self.ask_tools

    def allow(self, tool_name: str) -> None:
        self.ask_tools.discard(tool_name)
