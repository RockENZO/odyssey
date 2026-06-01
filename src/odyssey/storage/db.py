import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Any

from odyssey.config import get_data_dir


_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        db_path = get_data_dir() / "odyssey.db"
        _conn = sqlite3.connect(str(db_path))
        _conn.row_factory = sqlite3.Row
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            priority TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'pending',
            due_date TEXT,
            tags TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            sentiment TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            entry_date TEXT DEFAULT (date('now')),
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            sources TEXT DEFAULT '',
            content TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding_id TEXT,
            tags TEXT DEFAULT '',
            source TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


# ---- Tasks ----

def add_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: str | None = None,
    tags: str = "",
) -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO tasks (title, description, priority, due_date, tags) VALUES (?, ?, ?, ?, ?)",
        (title, description, priority, due_date, tags),
    )
    db.commit()
    return cur.lastrowid


def list_tasks(status: str | None = None) -> list[dict[str, Any]]:
    db = get_db()
    if status:
        rows = db.execute("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC", (status,))
    else:
        rows = db.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    return [dict(r) for r in rows.fetchall()]


def complete_task(task_id: int) -> bool:
    db = get_db()
    cur = db.execute(
        "UPDATE tasks SET status = 'done', updated_at = datetime('now') WHERE id = ?",
        (task_id,),
    )
    db.commit()
    return cur.rowcount > 0


def delete_task(task_id: int) -> bool:
    db = get_db()
    cur = db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    db.commit()
    return cur.rowcount > 0


def get_task(task_id: int) -> dict[str, Any] | None:
    db = get_db()
    row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


# ---- Journal ----

def add_journal_entry(content: str, sentiment: str = "", summary: str = "") -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO journal_entries (content, sentiment, summary) VALUES (?, ?, ?)",
        (content, sentiment, summary),
    )
    db.commit()
    return cur.lastrowid


def get_journal_entries(days: int = 7) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM journal_entries WHERE entry_date >= date('now', ?) ORDER BY created_at DESC",
        (f"-{days} days",),
    )
    return [dict(r) for r in rows.fetchall()]


def get_journal_entry(entry_id: int) -> dict[str, Any] | None:
    db = get_db()
    row = db.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,)).fetchone()
    return dict(row) if row else None


# ---- Research Notes ----

def add_research_note(topic: str, summary: str, sources: str = "", content: str = "", tags: str = "") -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO research_notes (topic, summary, sources, content, tags) VALUES (?, ?, ?, ?, ?)",
        (topic, summary, sources, content, tags),
    )
    db.commit()
    return cur.lastrowid


def list_research_notes(limit: int = 20) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.execute("SELECT * FROM research_notes ORDER BY created_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows.fetchall()]


# ---- Memories ----

def add_memory(content: str, tags: str = "", source: str = "manual") -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO memories (content, tags, source) VALUES (?, ?, ?)",
        (content, tags, source),
    )
    db.commit()
    return cur.lastrowid


def list_memories(limit: int = 20) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows.fetchall()]
