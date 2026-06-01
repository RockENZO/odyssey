from odyssey.llm.client import embed
from odyssey.storage.vector import add_to_collection, query_collection
from odyssey.storage.db import add_memory as db_add_memory


async def save_memory(content: str, tags: str = "") -> str:
    try:
        emb = await embed(content)
        doc_id = add_to_collection(
            "memories",
            texts=[content],
            embeddings=[emb],
            metadatas=[{"tags": tags, "source": "manual"}],
        )[0]
        db_add_memory(content, tags=tags, source="manual")
        return f"Saved to memory (id={doc_id}). I'll remember this."
    except Exception as e:
        return f"Failed to save memory: {e}"


async def query_memory(query: str) -> str:
    try:
        emb = await embed(query)
        results = query_collection("memories", emb, n_results=5)
        if not results:
            return "Nothing found in memory for that query."
        lines = ["Here's what I found in your memory:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['document']}")
            if r["metadata"].get("tags"):
                lines.append(f"   Tags: {r['metadata']['tags']}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to query memory: {e}"
