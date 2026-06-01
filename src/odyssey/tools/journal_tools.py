from odyssey.storage.db import add_journal_entry, get_journal_entries


async def _llm_chat(system: str, prompt: str, temperature: float = 0.3) -> str:
    from odyssey.llm.client import chat_completion
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resp = await chat_completion(messages, include_tools=False, temperature=temperature)
    return resp["message"]["content"]


async def save_journal_tool(content: str) -> str:
    try:
        summary = await _llm_chat(
            "You are a thoughtful journal assistant. Summarize the following journal entry in 1-2 sentences.",
            content,
            temperature=0.3,
        )
        sentiment_raw = await _llm_chat(
            "Analyze the sentiment of this journal entry. Respond with one word only: positive, negative, or neutral.",
            content,
            temperature=0.1,
        )
        sentiment = sentiment_raw.strip().lower()
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
