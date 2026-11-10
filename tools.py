import os
from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MAX_SCRAPED_CHARS_PER_SOURCE = 6500
MAX_SEARCH_SNIPPET_CHARS = 900


class SearchSource(TypedDict):
    title: str
    url: str
    snippet: str
    raw_content: str


def search_sources(query: str) -> list[SearchSource]:
    """Return deduplicated search results as structured source data."""
    results = tavily.search(
        query=query,
        max_results=3,
        include_raw_content="text",
    )
    sources: list[SearchSource] = []
    seen_urls = set()

    for result in results.get("results", []):
        url = result.get("url", "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = " ".join(result.get("title", "Untitled source").split())
        snippet = " ".join(result.get("content", "").split())
        if len(snippet) > MAX_SEARCH_SNIPPET_CHARS:
            snippet = (
                snippet[: MAX_SEARCH_SNIPPET_CHARS - 3].rsplit(" ", 1)[0] + "..."
            )
        raw_content = result.get("raw_content") or ""
        if not isinstance(raw_content, str):
            raw_content = ""

        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "raw_content": raw_content[:MAX_SCRAPED_CHARS_PER_SOURCE],
            }
        )

    return sources


def format_search_results(sources: list[SearchSource]) -> str:
    """Format structured search results for display and the research prompt."""
    entries = [
        f"{index}. {source['title']}\n"
        f"   URL: {source['url']}\n"
        f"   {source['snippet']}"
        for index, source in enumerate(sources, start=1)
    ]
    return "\n\n".join(entries) if entries else "No search results found."


def web_search(query: str) -> str:
    """Return formatted search results without using an LLM agent."""
    return format_search_results(search_sources(query))


def scrape_url(url: str) -> str:
    """Scrape a single page and return a bounded amount of clean text."""
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Scraping failed for {url}: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:MAX_SCRAPED_CHARS_PER_SOURCE]
