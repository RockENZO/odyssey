from odyssey.llm.client import generate, get_fast_model
from odyssey.tools.registry import get_tool
from odyssey.storage.db import add_research_note


SYSTEM_PROMPT = """You are a research assistant. Your job is to:
1. Search the web for information on the user's topic
2. Read the most relevant results
3. Synthesize findings into a clear, structured summary
4. Cite your sources

Be thorough but concise. Focus on actionable insights."""


async def research(topic: str) -> str:
    search_tool = get_tool("web_search")
    read_tool = get_tool("read_url")

    search_result = await search_tool.run(query=topic, max_results=5)

    urls = []
    for line in search_result.split("\n"):
        if line.strip().startswith("URL:"):
            url = line.strip()[4:].strip()
            if url not in urls:
                urls.append(url)

    article_contents = []
    for url in urls[:3]:
        content = await read_tool.run(url=url)
        article_contents.append(f"--- Content from {url} ---\n{content}\n")

    combined = f"Search Results:\n{search_result}\n\n"
    combined += "\n".join(article_contents)
    combined += "\n\nSynthesize the above into a well-structured research summary."

    summary = await generate(
        system=SYSTEM_PROMPT,
        prompt=combined,
        model=get_fast_model(),
        temperature=0.3,
    )

    sources_text = "\n".join(urls)
    add_research_note(
        topic=topic,
        summary=summary,
        sources=sources_text,
        content=combined,
    )

    result = f"## Research: {topic}\n\n{summary}\n\n### Sources\n{sources_text}"
    return result
