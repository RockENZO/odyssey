from urllib.parse import quote

import httpx
import re


async def web_search(query: str, max_results: int = 5) -> str:
    results = await _search_duckduckgo(query, max_results)
    if not results:
        results = await _search_fallback(query, max_results)
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results[:max_results], 1):
        lines.append(f"{i}. {r.get('title', 'No title')}")
        lines.append(f"   URL: {r.get('href', r.get('url', 'No URL'))}")
        lines.append(f"   {r.get('body', r.get('snippet', ''))}")
        lines.append("")
    return "\n".join(lines)


async def _search_duckduckgo(query: str, max_results: int) -> list[dict] | None:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results if results else None
    except Exception:
        return None


async def _search_fallback(query: str, max_results: int) -> list[dict]:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; OdysseyAgent/1.0)",
            })
            resp.raise_for_status()
        html = resp.text
        results = []
        for match in re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        ):
            results.append({
                "href": match.group(1),
                "title": re.sub(r'<[^>]+>', '', match.group(2)).strip(),
                "body": re.sub(r'<[^>]+>', '', match.group(3)).strip(),
            })
        return results[:max_results]
    except Exception:
        return []


async def read_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; OdysseyAgent/1.0)",
            })
            resp.raise_for_status()
        text = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"
        return text
    except Exception as e:
        return f"Failed to read URL: {e}"
