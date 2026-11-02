"""Structured plan kept in graph state, replacing the old text-echo ``todo_write``."""
from __future__ import annotations

import json
from typing import Any

STATUSES = ("pending", "in_progress", "completed")
_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def parse_todos(items: Any) -> list[dict[str, str]]:
    """Accept a JSON list (of strings or ``{content, status}`` objects) or plain text lines."""
    if isinstance(items, str):
        text = items.strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            items = [line for line in text.splitlines() if line.strip()]
    if not isinstance(items, list):
        items = [items]
    todos: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("title") or item.get("task") or "").strip()
            status = str(item.get("status", "pending")).strip().lower().replace(" ", "_")
        else:
            content, status = str(item).strip().lstrip("-*").strip(), "pending"
            if content[:1].isdigit():
                content = content.lstrip("0123456789").lstrip(".) ").strip()
            for status_name, mark in _MARKS.items():
                if content.startswith(mark):
                    content, status = content[len(mark):].strip(), status_name
                    break
        if not content:
            continue
        todos.append({"content": content, "status": status if status in STATUSES else "pending"})
    return todos


def format_todos(todos: list[dict[str, str]]) -> str:
    if not todos:
        return "(no plan yet)"
    return "\n".join(f"{i}. {_MARKS[t['status']]} {t['content']}" for i, t in enumerate(todos, 1))
