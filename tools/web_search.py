# tools/web_search.py
"""
Hybrid web search tool with automatic short-answer summarizer.

Primary: Serper API (if SERPER_API_KEY set)
Fallback: DuckDuckGo HTML scrape

Returns:
{
  "query": "...",
  "provider": "serper" | "duckduckgo_html",
  "results": [ { "title": ..., "url": ..., "snippet": ... }, ... ],
  "short_answer": "Concise one-line answer (if found)"
}
"""
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
# SERPER_API_KEY must be set in .env file for enhanced search
# Falls back to DuckDuckGo if not provided
SERPER_KEY = os.environ.get("SERPER_API_KEY")

def _serper_search(q: str) -> Optional[List[Dict[str, Any]]]:
    if not SERPER_KEY:
        return None
    try:
        url = "https://google.serper.dev/search"
        payload = {"q": q}
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        if res.status_code != 200:
            logger.debug("Serper non-200: %s %s", res.status_code, res.text)
            return None
        data = res.json()
        items = []
        # Serper returns 'organic' list (each with title, link, snippet)
        for it in data.get("organic", []) or []:
            items.append({"title": it.get("title"), "url": it.get("link"), "snippet": it.get("snippet")})
        return items or None
    except Exception as e:
        logger.debug("Serper search failed: %s", e)
        return None

def _duckduckgo_html(q: str) -> List[Dict[str, Any]]:
    try:
        url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(q)
        res = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        out = []
        # Selector targets result blocks in DuckDuckGo HTML
        for r in soup.select(".result"):
            a = r.select_one(".result__a")
            s = r.select_one(".result__snippet")
            if not a:
                continue
            href = a.get("href")
            title = a.get_text(strip=True)
            snippet = s.get_text(strip=True) if s else ""
            out.append({"title": title, "url": href, "snippet": snippet})
            if len(out) >= 10:
                break
        return out
    except Exception as e:
        logger.debug("DuckDuckGo scrape failed: %s", e)
        return []

def _make_short_answer(query: str, results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Simple short-answer extraction without hardcoded logic.
    Returns raw search results for the LLM to interpret.
    """
    if not results:
        return None

    # Return top 3 results as structured data for LLM to parse
    top_results = []
    for r in results[:3]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        if title or snippet:
            entry = f"{title}: {snippet}" if title and snippet else (title or snippet)
            top_results.append(entry.strip())
    
    return "\n\n".join(top_results) if top_results else None

def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Hybrid search: try Serper (if key), fallback to DuckDuckGo HTML.
    Returns structured results and a 'short_answer' when available.
    """
    # Defensive: accept numeric strings from tool invocation and coerce to int
    try:
        max_results = int(max_results)
    except Exception:
        max_results = 5

    # Sanitize bounds
    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50
    # 1) Try Serper first
    results = None
    provider = None
    if SERPER_KEY:
        try:
            s = _serper_search(query)
            if s:
                provider = "serper"
                results = s[:max_results]
        except Exception:
            results = None

    # 2) Fallback to DDG HTML
    if results is None:
        # _duckduckgo_html already limits to 10 internally; still slice defensively
        ddg = _duckduckgo_html(query)
        results = ddg[:max_results]
        provider = "duckduckgo_html"

    short = _make_short_answer(query, results)
    out = {"query": query, "provider": provider, "results": results, "short_answer": short}
    return out

TOOL = {
    "name": "web_search",
    "func": web_search,
    "description": "Hybrid web search (Serper primary, DuckDuckGo HTML fallback). Returns concise short_answer when possible."
}
