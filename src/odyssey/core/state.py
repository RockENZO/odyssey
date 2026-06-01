from typing import Any
from pydantic import BaseModel
from dataclasses import dataclass, field


class Message(BaseModel):
    role: str
    content: str


@dataclass
class AgentState:
    messages: list[dict[str, str]] = field(default_factory=list)
    active_agent: str = "supervisor"
    tool_results: list[str] = field(default_factory=list)
    user_input: str = ""
    final_output: str = ""
    research_results: dict[str, Any] = field(default_factory=dict)
    memory_results: list[dict[str, str]] = field(default_factory=list)
    task_results: list[dict[str, Any]] = field(default_factory=list)
    journal_results: list[dict[str, Any]] = field(default_factory=list)
    briefing_data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
