from datetime import date

from odyssey.llm.client import generate, get_fast_model
from odyssey.storage.db import list_tasks, get_journal_entries
from odyssey.tools.registry import get_tool


async def generate_briefing() -> str:
    today = date.today().isoformat()
    tasks = list_tasks(status="pending")
    journal_entries = get_journal_entries(days=3)
    search_tool = get_tool("web_search")

    news = ""
    try:
        news = await search_tool.run(query="top news today technology", max_results=3)
    except Exception:
        news = "Could not fetch news."

    sections = [f"# 🌅 Daily Briefing — {today}\n"]

    sections.append("## 📋 Today's Tasks")
    if tasks:
        for t in tasks[:10]:
            due = f" (due: {t['due_date']})" if t["due_date"] else ""
            sections.append(f"- [{'✅' if t['status'] == 'done' else ' '}] {t['title']} [{t['priority']}]{due}")
    else:
        sections.append("- No pending tasks. Enjoy your day!")
    sections.append("")

    sections.append("## 📰 Quick Headlines")
    sections.append(news)
    sections.append("")

    if journal_entries:
        sections.append("## 📓 Recent Journal")
        for e in journal_entries[:3]:
            mood = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(e.get("sentiment", ""), "📝")
            sections.append(f"- {mood} {e['entry_date']}: {e['summary'] or e['content'][:80]}...")
        sections.append("")

    summary = await generate(
        system="You are a morning briefing assistant. Write 1-2 sentences of encouragement or insight for today.",
        prompt=f"Today is {today}. The user has {len(tasks)} pending tasks and {len(journal_entries)} recent journal entries.",
        temperature=0.7,
    )
    sections.append(f"## 💭 Thought of the Day\n{summary}")

    return "\n".join(sections)
