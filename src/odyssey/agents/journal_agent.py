from odyssey.llm.client import generate, get_fast_model
from odyssey.tools.registry import get_tool
from odyssey.storage.db import get_journal_entries


SYSTEM_PROMPT = """You are a thoughtful journaling companion. Help users reflect on their day,
explore their thoughts, and identify patterns in their life. Be empathetic and insightful."""


async def handle_journal(user_input: str) -> str:
    save_tool = get_tool("save_journal")
    summary_tool = get_tool("get_journal_summary")

    write_keywords = ["write", "journal", "today", "reflect", "entry", "note"]
    summary_keywords = ["summary", "summarize", "overview", "show", "list", "recent", "past"]

    is_write = any(kw in user_input.lower() for kw in write_keywords)
    is_summary = any(kw in user_input.lower() for kw in summary_keywords)

    if is_write and not is_summary:
        return await save_tool.run(content=user_input)

    if is_summary:
        return await summary_tool.run(days=7)

    entries = get_journal_entries(days=7)
    if entries:
        context = "\n".join(
            f"[{e['entry_date']}] {e['summary'] or e['content'][:100]}..."
            for e in entries
        )
        prompt = f"""User said: {user_input}

Recent journal entries:
{context}

Respond thoughtfully, referencing their journal entries if relevant."""
    else:
        prompt = f"User said: {user_input}. Respond as a thoughtful journaling companion."

    return await generate(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        model=get_fast_model(),
    )
