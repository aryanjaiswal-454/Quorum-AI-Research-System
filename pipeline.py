import sys
from typing import TypedDict

from agents import (
    MAX_COMPLETION_TOKENS,
    MAX_PROMPT_TOKENS,
    GROQ_TPM_LIMIT,
    TPM_SAFETY_MARGIN,
    count_prompt_tokens,
    generate_report_and_feedback,
)
from tools import (
    MAX_SCRAPED_CHARS_PER_SOURCE,
    format_search_results,
    scrape_url,
    search_sources,
    SearchSource,
)

SEARCH_CONTEXT_LIMIT = 1800
TOPIC_CHAR_LIMIT = 300


class ScrapedSource(TypedDict):
    title: str
    url: str
    content: str
    status: str


class PipelineResult(TypedDict):
    search_results: str
    scraped_url: str
    scraped_sources: list[ScrapedSource]
    scraped_content: str
    estimated_request_tokens: int
    report: str
    feedback: str


def _print_console(*values: object, sep: str = " ", end: str = "\n") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    text = sep.join(map(str, values)) + end
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding))


def _build_research_context(
    search_results: str,
    scraped_sources: list[ScrapedSource],
    chars_per_source: int,
) -> str:
    scraped_research = "\n\n".join(
        f"Source: {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Content status: {source['status']}\n"
        f"Page text: {source['content'][:chars_per_source]}"
        for source in scraped_sources
    ) or "No source URLs were returned by web search."
    return (
        f"SEARCH RESULTS:\n{search_results[:SEARCH_CONTEXT_LIMIT]}\n\n"
        f"SCRAPED CONTENT:\n{scraped_research}"
    )


def _fit_research_context(
    topic: str,
    search_results: str,
    scraped_sources: list[ScrapedSource],
) -> str:
    low = 0
    high = MAX_SCRAPED_CHARS_PER_SOURCE
    while low < high:
        chars_per_source = (low + high + 1) // 2
        candidate = _build_research_context(
            search_results,
            scraped_sources,
            chars_per_source,
        )
        if count_prompt_tokens(topic, candidate) <= MAX_PROMPT_TOKENS:
            low = chars_per_source
        else:
            high = chars_per_source - 1

    research = _build_research_context(search_results, scraped_sources, low)
    prompt_tokens = count_prompt_tokens(topic, research)
    if prompt_tokens > MAX_PROMPT_TOKENS:
        raise ValueError(
            "The topic and source metadata exceed the prompt token budget; "
            "try a shorter topic."
        )
    return research


def _read_source(source: SearchSource) -> ScrapedSource:
    tavily_content = source["raw_content"].strip()
    if tavily_content:
        return {
            "title": source["title"],
            "url": source["url"],
            "content": tavily_content,
            "status": "Page text returned by Tavily",
        }

    fetched_content = scrape_url(source["url"])
    if fetched_content.startswith(f"Scraping failed for {source['url']}:"):
        if source["snippet"]:
            return {
                "title": source["title"],
                "url": source["url"],
                "content": source["snippet"],
                "status": "Search excerpt only; direct page fetch failed",
            }
        return {
            "title": source["title"],
            "url": source["url"],
            "content": fetched_content,
            "status": "Page text unavailable",
        }

    return {
        "title": source["title"],
        "url": source["url"],
        "content": fetched_content[:MAX_SCRAPED_CHARS_PER_SOURCE],
        "status": "Full text extracted directly",
    }


def run_research_pipeline(topic: str) -> PipelineResult:
    topic = topic.strip()
    if not topic:
        raise ValueError("Research topic cannot be empty.")
    if len(topic) > TOPIC_CHAR_LIMIT:
        raise ValueError(f"Research topic must be {TOPIC_CHAR_LIMIT} characters or fewer.")

    print("\n" + " =" * 50)
    print("step 1 - searching the web ...")
    print("=" * 50)
    sources = search_sources(topic)
    search_results = format_search_results(sources)
    _print_console("\nsearch results:\n", search_results)

    print("\n" + " =" * 50)
    print("step 2 - scraping all search results ...")
    print("=" * 50)
    scraped_sources = [_read_source(source) for source in sources]
    scraped_url = (
        scraped_sources[0]["url"] if scraped_sources else ""
    )
    scraped_content = "\n\n".join(
        f"Source: {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Status: {source['status']}\n"
        f"{source['content']}"
        for source in scraped_sources
    ) or "No source URLs were returned by web search."
    _print_console("\nscraped content:\n", scraped_content)

    research = _fit_research_context(topic, search_results, scraped_sources)
    prompt_tokens = count_prompt_tokens(topic, research)
    estimated_request_tokens = (
        prompt_tokens + MAX_COMPLETION_TOKENS + TPM_SAFETY_MARGIN
    )
    if estimated_request_tokens > GROQ_TPM_LIMIT:
        raise ValueError(
            "The estimated Groq request exceeds the 8,000 TPM budget; "
            "reduce the topic or source content."
        )
    print(
        f"\nEstimated Groq budget: {prompt_tokens:,} prompt + "
        f"{MAX_COMPLETION_TOKENS:,} completion + "
        f"{TPM_SAFETY_MARGIN:,} safety = {estimated_request_tokens:,} of "
        f"{GROQ_TPM_LIMIT:,} TPM tokens"
    )

    print("\n" + " =" * 50)
    print("step 3 - drafting and reviewing the report in one Groq request ...")
    print("=" * 50)
    report, feedback = generate_report_and_feedback(topic, research)

    _print_console("\nfinal report:\n", report)
    _print_console("\nreview:\n", feedback)
    state: PipelineResult = {
        "search_results": search_results,
        "scraped_url": scraped_url,
        "scraped_sources": scraped_sources,
        "scraped_content": scraped_content,
        "estimated_request_tokens": estimated_request_tokens,
        "report": report,
        "feedback": feedback,
    }
    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)
