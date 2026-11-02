"""One normalised shape for tool calls, with a stable id per call.

Ollama does not give tool calls ids, and the same tool is often called twice in one turn,
so ids are generated here. The id is what later stages use to match a tool result to the
call that produced it (resume after a crash, replay, ledger provenance).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def parse_tool_calls(raw_calls: list[Any] | None) -> list[ToolCall]:
    """Convert the tool calls in a model response (dicts or objects) to ``ToolCall``."""
    parsed: list[ToolCall] = []
    for raw in raw_calls or []:
        if isinstance(raw, dict):
            fn = raw.get("function", {})
            name, args, call_id = fn.get("name", ""), fn.get("arguments", {}), raw.get("id")
        else:
            name, args = raw.function.name, raw.function.arguments
            call_id = getattr(raw, "id", None)
        call = ToolCall(name=name, arguments=_coerce_arguments(args))
        if call_id and call_id != name:
            call.id = str(call_id)
        parsed.append(call)
    return parsed
