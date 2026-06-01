from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conversation:
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 20
    project_root: str = ""
    working_dir: str = ""

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        for k, v in extra.items():
            if k == "tool_calls" and v is not None:
                serialized = []
                for tc in v:
                    if hasattr(tc, "function"):
                        fn = tc.function
                        args = fn.arguments
                        if hasattr(args, "model_dump"):
                            args = args.model_dump()
                        serialized.append({
                            "function": {"name": fn.name, "arguments": args},
                            "id": getattr(tc, "id", None) or fn.name,
                            "type": "function",
                        })
                    else:
                        serialized.append(tc)
                msg["tool_calls"] = serialized
            else:
                msg[k] = v
        self.messages.append(msg)

    def add_tool_result(self, tool_name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "content": content,
        })

    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def recent_history(self, n: int = 10) -> list[dict[str, Any]]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
