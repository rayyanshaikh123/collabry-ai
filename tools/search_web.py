"""
Web Search Tool - Search the internet for current information.

This tool is essential for:
- Finding online courses (Coursera, Udemy, edX, etc.)
- Getting current information not in uploaded documents
- Researching topics, tools, frameworks
- Finding tutorials and learning resources
"""

import os
import httpx
from typing import Optional
from langchain_core.tools import tool


@tool
async def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web for current information about courses, tutorials, or topics.
    
    Use this tool when users ask about:
    - Online courses (Coursera, Udemy, edX, Pluralsight, etc.)
    - Current information not in their documents
    - Tutorials, guides, or learning resources
    - Technology comparisons or recommendations
    
    Args:
        query: Search query (e.g., "Python courses for beginners", "data structures tutorial")
        num_results: Number of results to return (default: 5, max: 10)
    
    Returns:
        Formatted search results with titles, snippets, and links
    
    Examples:
        >>> await search_web("best Python courses on Coursera")
        >>> await search_web("JavaScript tutorial for beginners")
        >>> await search_web("machine learning certification programs")
    """
    # Sanitize inputs
    query = " ".join((query or "").split()).strip()
    if not query:
        return "Please provide a non-empty search query."

    # Keep queries reasonably short for providers that reject long payloads
    if len(query) > 300:
        query = query[:300]

    # Limit results
    num_results = min(max(int(num_results), 1), 10)

    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Try Serper API first (preferred)
    serper_key = (os.getenv("SERPER_API_KEY") or "").strip()
    if serper_key:
        try:
            return await _search_with_serper(query, serper_key, num_results)
        except Exception as e:
            if debug:
                print(f"Serper search failed: {e}, falling back...")
    
    # Try Tavily API
    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if tavily_key:
        try:
            return await _search_with_tavily(query, tavily_key, num_results)
        except Exception as e:
            if debug:
                print(f"Tavily search failed: {e}, falling back...")
    
    # Fallback: Return guidance message
    return _no_api_fallback(query)


async def _search_with_serper(query: str, api_key: str, num_results: int) -> str:
    """Search using Serper API (Google Search)."""
    url = "https://google.serper.dev/search"
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    
    # Format results
    results = []
    
    # Add organic results
    for item in data.get("organic", [])[:num_results]:
        title = item.get("title", "No title")
        snippet = item.get("snippet", "No description")
        link = item.get("link", "")
        
        results.append(f"**{title}**\n{snippet}\n🔗 {link}\n")
    
    if not results:
        return f"No results found for: {query}"
    
    return "\n".join(results)


async def _search_with_tavily(query: str, api_key: str, num_results: int) -> str:
    """Search using Tavily API."""
    url = "https://api.tavily.com/search"
    api_key = (api_key or "").strip()
    
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",
        "include_answer": False
    }

    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    
    # Format results
    results = []
    
    for item in data.get("results", [])[:num_results]:
        title = item.get("title", "No title")
        snippet = item.get("content", "No description")
        link = item.get("url", "")
        
        results.append(f"**{title}**\n{snippet}\n🔗 {link}\n")
    
    if not results:
        return f"No results found for: {query}"
    
    return "\n".join(results)


def _no_api_fallback(query: str) -> str:
    """Fallback when no API keys are configured."""
    
    # Parse query for course-related keywords
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["course", "tutorial", "learn", "certification", "class"]):
        platforms = [
            "🎓 **Coursera** - https://www.coursera.org/search?query=" + query.replace(" ", "+"),
            "🎓 **Udemy** - https://www.udemy.com/courses/search/?q=" + query.replace(" ", "+"),
            "🎓 **edX** - https://www.edx.org/search?q=" + query.replace(" ", "+"),
            "🎓 **Pluralsight** - https://www.pluralsight.com/search?q=" + query.replace(" ", "+"),
            "🎓 **LinkedIn Learning** - https://www.linkedin.com/learning/search?keywords=" + query.replace(" ", "+"),
        ]
        
        return (
            f"I can help you find courses about '{query}'!\n\n"
            "Here are the best platforms to search:\n\n" +
            "\n".join(platforms) +
            "\n\n💡 **Tip:** For real-time course recommendations with ratings and prices, "
            "ask your administrator to add a SERPER_API_KEY or TAVILY_API_KEY to the .env file."
        )
    
    # General web search suggestion
    search_urls = [
        f"🔍 **Google** - https://www.google.com/search?q={query.replace(' ', '+')}",
        f"🔍 **DuckDuckGo** - https://duckduckgo.com/?q={query.replace(' ', '+')}",
    ]
    
    return (
        f"I'd like to search the web for '{query}', but no API key is configured.\n\n"
        "You can search manually here:\n\n" +
        "\n".join(search_urls) +
        "\n\n💡 **For administrators:** Add SERPER_API_KEY or TAVILY_API_KEY to .env for automatic web search."
    )


# Example usage for testing
if __name__ == "__main__":
    result = search_web("Python courses for beginners")
    print(result)
