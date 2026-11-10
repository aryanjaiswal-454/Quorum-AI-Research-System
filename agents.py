import os
import re
import tiktoken

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GROQ_TPM_LIMIT = 8000
MAX_PROMPT_TOKENS = 4000
MAX_COMPLETION_TOKENS = 3000
TPM_SAFETY_MARGIN = 800
TOKEN_ENCODING = tiktoken.get_encoding("o200k_harmony")

if MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS + TPM_SAFETY_MARGIN > GROQ_TPM_LIMIT:
    raise ValueError("Configured Groq request budget exceeds the 8,000 TPM limit.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=MAX_COMPLETION_TOKENS,
    max_retries=0,
    reasoning_effort="high",
)

research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful research analyst. Reason through the evidence before writing. "
            "Use only the supplied sources, distinguish source claims from established facts, "
            "compare agreement and disagreement across sources, and explicitly note uncertainty. "
            "Do not invent facts, dates, statistics, quotations, or citations.",
        ),
        (
            "human",
            """Research question: {topic}

            Evidence gathered from web search and scraped pages:
            {research}

            Return exactly these two sections, in this order:
            ===== REPORT =====
            Write a concise, well-reasoned report with:
            - A short introduction defining the scope of the question.
            - Up to three findings supported by the supplied source text. Identify each source
              and distinguish direct evidence from interpretation.
            - A synthesis of agreement, disagreement, and evidence gaps.
            - A conclusion and source list using only the supplied URLs.
            Treat search excerpts as limited evidence, not as full-page content. Never infer
            article details from titles or URLs.
            ===== CRITIQUE =====
            Score the report out of 10. Briefly identify its strongest evidence, one material
            limitation, and one specific follow-up question or source that would improve confidence.
            Keep the complete report and critique concise enough to finish within the output limit.""",
        ),
    ]
)

research_chain = research_prompt | llm
SECTION_HEADER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:={3,}\s*)?(?:\*{1,2}\s*)?"
    r"(REPORT|CRITIQUE|EVIDENCE REVIEW)"
    r"(?:\s*\*{1,2})?(?:\s*={3,})?\s*:?\s*$"
)
MISSING_CRITIQUE = (
    "The model did not return a separate evidence review. Verify the report's "
    "claims against the source text shown below."
)


def count_prompt_tokens(topic: str, research: str) -> int:
    """Conservatively estimate prompt tokens using GPT-OSS's Harmony encoding."""
    messages = research_prompt.format_messages(topic=topic, research=research)
    return sum(
        len(TOKEN_ENCODING.encode(message.content, disallowed_special=())) + 16
        for message in messages
    ) + 8


def _parse_report_and_feedback(response: str) -> tuple[str, str]:
    """Split common report/review heading formats without discarding a valid report."""
    headers = list(SECTION_HEADER.finditer(response))
    report_header = next(
        (header for header in headers if header.group(1).upper() == "REPORT"),
        None,
    )
    critique_header = next(
        (
            header
            for header in headers
            if header.group(1).upper() in {"CRITIQUE", "EVIDENCE REVIEW"}
            and (report_header is None or header.start() > report_header.start())
        ),
        None,
    )

    if report_header is not None:
        report_start = report_header.end()
    else:
        report_start = 0

    report_end = (
        critique_header.start()
        if critique_header is not None
        else len(response)
    )
    report = response[report_start:report_end].strip()
    feedback = (
        response[critique_header.end():].strip()
        if critique_header is not None
        else MISSING_CRITIQUE
    )

    if not report:
        raise ValueError("Groq returned an empty research report.")
    return report, feedback or MISSING_CRITIQUE


def generate_report_and_feedback(topic: str, research: str) -> tuple[str, str]:
    """Generate the report and its review in one bounded Groq request."""
    message = research_chain.invoke({"topic": topic, "research": research})
    response = StrOutputParser().invoke(message)
    report, feedback = _parse_report_and_feedback(response)

    if getattr(message, "response_metadata", {}).get("finish_reason") == "length":
        feedback = (
            "The model reached its output limit; the report or review may be incomplete.\n\n"
            f"{feedback}"
        )
    return report, feedback
