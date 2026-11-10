import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import agents
import pipeline
import tools


class ReportParsingTests(unittest.TestCase):
    def test_splits_exact_section_markers(self) -> None:
        report, critique = agents._parse_report_and_feedback(
            "===== REPORT =====\nEvidence-based report.\n"
            "===== CRITIQUE =====\nScore: 8/10."
        )

        self.assertEqual(report, "Evidence-based report.")
        self.assertEqual(critique, "Score: 8/10.")

    def test_accepts_markdown_section_headings(self) -> None:
        report, critique = agents._parse_report_and_feedback(
            "## Report\nFindings from the sources.\n"
            "### Evidence review\nScore: 7/10."
        )

        self.assertEqual(report, "Findings from the sources.")
        self.assertEqual(critique, "Score: 7/10.")

    def test_preserves_report_when_critique_is_missing(self) -> None:
        report, critique = agents._parse_report_and_feedback(
            "A valid report without a separate review."
        )

        self.assertEqual(report, "A valid report without a separate review.")
        self.assertEqual(critique, agents.MISSING_CRITIQUE)

    def test_rejects_an_empty_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty research report"):
            agents._parse_report_and_feedback(
                "===== REPORT =====\n===== CRITIQUE =====\nScore: 4/10."
            )

    def test_reports_when_groq_reaches_output_limit(self) -> None:
        message = AIMessage(
            content="A valid report.",
            response_metadata={"finish_reason": "length"},
        )

        with patch.object(agents, "research_chain") as research_chain:
            research_chain.invoke.return_value = message
            report, feedback = agents.generate_report_and_feedback("topic", "evidence")

        self.assertEqual(report, "A valid report.")
        self.assertIn("reached its output limit", feedback)


class SourceRetrievalTests(unittest.TestCase):
    def test_search_requests_and_keeps_extracted_text(self) -> None:
        with patch.object(
            tools.tavily,
            "search",
            return_value={
                "results": [
                    {
                        "title": "Source title",
                        "url": "https://example.com/article",
                        "content": "Search excerpt",
                        "raw_content": "Full extracted article text",
                    }
                ]
            },
        ) as search:
            sources = tools.search_sources("research question")

        search.assert_called_once_with(
            query="research question",
            max_results=3,
            include_raw_content="text",
        )
        self.assertEqual(sources[0]["raw_content"], "Full extracted article text")

    def test_prefers_tavily_text_without_fetching_page_again(self) -> None:
        source = {
            "title": "Source title",
            "url": "https://example.com/article",
            "snippet": "Search excerpt",
            "raw_content": "Extracted article text",
        }

        with patch.object(pipeline, "scrape_url") as scrape:
            result = pipeline._read_source(source)

        scrape.assert_not_called()
        self.assertEqual(result["content"], "Extracted article text")
        self.assertEqual(result["status"], "Page text returned by Tavily")

    def test_uses_search_excerpt_when_direct_fetch_fails(self) -> None:
        source = {
            "title": "Blocked source",
            "url": "https://example.com/blocked",
            "snippet": "Relevant search excerpt",
            "raw_content": "",
        }

        with patch.object(
            pipeline,
            "scrape_url",
            return_value="Scraping failed for https://example.com/blocked: 403 Forbidden",
        ):
            result = pipeline._read_source(source)

        self.assertEqual(result["content"], "Relevant search excerpt")
        self.assertEqual(
            result["status"],
            "Search excerpt only; direct page fetch failed",
        )


if __name__ == "__main__":
    unittest.main()
