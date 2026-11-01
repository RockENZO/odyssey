from urllib.parse import quote

import httpx
import re

# Bound how much of a remote response is copied into memory before the visible
# cap is applied. A page can be arbitrarily large; reading ``resp.text`` first
# would decode and retain the whole body even though only 8,000 characters are
# displayed.
MAX_URL_BYTES = 2_000_000
MAX_SEARCH_BYTES = 512_000
MAX_URL_TEXT_CHARS = 8_000
USER_AGENT = "Mozilla/5.0 (compatible; OdysseyAgent/1.0)"


async def _read_bounded(response, limit):
    """Stream at most ``limit`` bytes and report whether the body was cut."""
    chunks, total = [], 0
    truncated = False
    async for chunk in response.aiter_bytes():
        if total + len(chunk) > limit:
            chunks.append(chunk[:limit - total])
            total = limit
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    encoding = response.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace"), truncated


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
            async with client.stream("GET", url, headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                html, _ = await _read_bounded(resp, MAX_SEARCH_BYTES)
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
            async with client.stream("GET", url, headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                text, download_cut = await _read_bounded(resp, MAX_URL_BYTES)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > MAX_URL_TEXT_CHARS:
            text = text[:MAX_URL_TEXT_CHARS] + "... [truncated]"
        elif download_cut:
            text = (text + " ... [download truncated at 2 MB]") if text else "(page download truncated at 2 MB)"
        return text
    except Exception as e:
        return f"Failed to read URL: {e}"
