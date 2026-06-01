from odyssey.llm.client import generate, get_fast_model
from odyssey.tools.registry import get_tool
from odyssey.storage.db import list_memories


SYSTEM_PROMPT = """You are a memory management assistant. Help the user save and retrieve information.
When saving, organize the content well. When retrieving, present the information clearly with context."""


async def handle_memory(user_input: str) -> str:
    save_tool = get_tool("save_memory")
    query_tool = get_tool("query_memory")

    save_keywords = ["remember", "save", "store", "note", "remember that"]
    query_keywords = ["recall", "what", "find", "search", "retrieve", "look up"]

    is_save = any(kw in user_input.lower() for kw in save_keywords)
    is_query = any(kw in user_input.lower() for kw in query_keywords)

    if is_save and not is_query:
        content = user_input
        for prefix in ["remember ", "save ", "store ", "note "]:
            if user_input.lower().startswith(prefix):
                content = user_input[len(prefix):].strip()
                break
        return await save_tool.run(content=content)

    if is_query:
        return await query_tool.run(query=user_input)

    memories = list_memories(limit=5)
    if memories:
        context = "\n".join(f"- {m['content'][:200]}" for m in memories)
        prompt = f"""User said: {user_input}

Recent memories:
{context}

Respond helpfully, referencing relevant memories if appropriate."""
    else:
        prompt = f"User said: {user_input}. Respond helpfully."

    return await generate(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        model=get_fast_model(),
    )
