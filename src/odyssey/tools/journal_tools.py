from odyssey.storage.db import add_journal_entry, get_journal_entries
from odyssey.llm.client import generate


async def save_journal_tool(content: str) -> str:
    try:
        summary = await generate(
            system="You are a thoughtful journal assistant. Summarize the following journal entry in 1-2 sentences.",
            prompt=content,
            temperature=0.3,
        )
        sentiment = await generate(
            system="Analyze the sentiment of this journal entry. Respond with one word only: positive, negative, or neutral.",
            prompt=content,
            temperature=0.1,
        )
        sentiment = sentiment.strip().lower()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"
        entry_id = add_journal_entry(content, sentiment=sentiment, summary=summary)
        return f"Journal entry saved (id={entry_id}).\n\nSummary: {summary}\nMood: {sentiment}"
    except Exception as e:
        return f"Failed to save journal entry: {e}"


async def get_journal_summary_tool(days: int = 7) -> str:
    try:
        entries = get_journal_entries(days=days)
        if not entries:
            return "No journal entries found in that period."
        lines = [f"Journal entries from the last {days} days:\n"]
        for e in entries:
            mood_icon = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(e.get("sentiment", ""), "📝")
            lines.append(f"  [{e['id']}] {e['entry_date']} {mood_icon}")
            lines.append(f"       {e['summary'] or e['content'][:100]}...")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to get journal summary: {e}"
