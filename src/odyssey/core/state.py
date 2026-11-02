from dataclasses import dataclass, field
from typing import Any

from odyssey.core.tool_calls import ToolCall, parse_tool_calls


@dataclass
class ToolRecord:
    """One executed tool call, kept outside the model context as evidence."""
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    duration_s: float


@dataclass
class Conversation:
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 20
    project_root: str = ""
    working_dir: str = ""
    tool_log: list[ToolRecord] = field(default_factory=list)

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        for k, v in extra.items():
            if k == "tool_calls" and v is not None:
                calls = v if all(isinstance(c, ToolCall) for c in v) else parse_tool_calls(v)
                msg["tool_calls"] = [c.to_message() for c in calls]
            else:
                msg[k] = v
        self.messages.append(msg)

    def add_tool_result(self, tool_name: str, content: str, tool_call_id: str | None = None) -> None:
        msg: dict[str, Any] = {"role": "tool", "content": content, "tool_name": tool_name}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)

    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def recent_history(self, n: int = 10) -> list[dict[str, Any]]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
