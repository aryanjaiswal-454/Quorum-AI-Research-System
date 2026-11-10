import math
import time

import streamlit as st
from groq import APIError, RateLimitError

from pipeline import run_research_pipeline

RETRY_COOLDOWN_SECONDS = 60

st.set_page_config(
    page_title="Quorum | Research workspace",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


EXAMPLE_TOPICS = (
    "How war is impacting stock market",
    "Recent advances in fusion energy",
    "How AI is changing medical research",
)
EXAMPLE_PLACEHOLDER = "Choose an example..."


def apply_example_topic() -> None:
    selected_topic = st.session_state.sample_topic
    if selected_topic != EXAMPLE_PLACEHOLDER:
        st.session_state.topic_input = selected_topic


def seconds_until_retry(retry_at: float, now: float | None = None) -> int:
    current_time = time.time() if now is None else now
    return max(0, math.ceil(retry_at - current_time))


def start_research() -> None:
    st.session_state.results = {}
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.rate_limit_retry_at = None


@st.fragment(run_every="1s")
def render_rate_limit_countdown(retry_at: float) -> None:
    remaining = seconds_until_retry(retry_at)
    if remaining == 0:
        st.rerun()
    st.info(
        f"Groq is temporarily rate-limited. Try again in **{remaining} seconds**."
    )


for key, default in (("results", {}), ("running", False), ("done", False)):
    if key not in st.session_state:
        st.session_state[key] = default
st.session_state.setdefault("rate_limit_retry_at", None)

st.header(":material/hub: QUORUM")
st.caption("RESEARCH WORKSPACE")
st.title("Turn a question into a clearer picture.")
st.write(
    "Quorum finds relevant sources, retrieves available page text, and brings "
    "the evidence together in a concise report."
)

st.space("small")
input_column, process_column = st.columns([1.25, 0.75], gap="large")

with input_column:
    with st.container(border=True):
        st.subheader("Start with a question")
        st.caption("Be specific for more focused sources and findings.")
        st.selectbox(
            "Need a starting point?",
            (EXAMPLE_PLACEHOLDER, *EXAMPLE_TOPICS),
            key="sample_topic",
            on_change=apply_example_topic,
        )

        with st.form("research_form"):
            st.text_input(
                "Research topic",
                placeholder="e.g. What are the latest developments in cricket?",
                key="topic_input",
                max_chars=300,
            )
            run_button = st.form_submit_button(
                "Run research",
                type="primary",
                icon=":material/arrow_forward:",
                width="stretch",
                disabled=(
                    st.session_state.rate_limit_retry_at is not None
                    and seconds_until_retry(st.session_state.rate_limit_retry_at) > 0
                ),
            )

        if run_button:
            topic = st.session_state.topic_input.strip()
            if not topic:
                st.warning("Enter a research topic to get started.", icon=":material/info:")
            else:
                start_research()
                st.rerun()

with process_column:
    with st.container(border=True):
        st.subheader("A focused workflow")
        st.caption("Evidence-led synthesis in one TPM-budgeted Groq request.")
        steps = (
            ("travel_explore", "Find sources", "Searches for up to three relevant results."),
            ("article", "Read for context", "Uses Tavily page text, with direct-fetch fallback."),
            ("summarize", "Synthesize evidence", "Compares sources using balanced reasoning in one Groq request."),
        )
        for icon, title, description in steps:
            st.markdown(f":material/{icon}: **{title}**")
            st.caption(description)

        if st.session_state.running:
            st.badge("Research in progress", icon=":material/pending:", color="orange")
        elif st.session_state.done and st.session_state.results:
            st.badge("Research complete", icon=":material/check_circle:", color="green")
        else:
            st.badge("Ready when you are", icon=":material/radio_button_unchecked:", color="blue")

if st.session_state.running and not st.session_state.done:
    pipeline_succeeded = False
    try:
        with st.status(
            "Quorum is gathering and reviewing evidence...",
            expanded=True,
        ) as progress:
            progress.write("Searching for sources, reading each result, then drafting a report.")
            pipeline_result = run_research_pipeline(st.session_state.topic_input)

        st.session_state.results = {
            "search": pipeline_result["search_results"],
            "scraped_url": pipeline_result.get("scraped_url", ""),
            "scraped_sources": pipeline_result.get("scraped_sources", []),
            "reader": pipeline_result["scraped_content"],
            "estimated_request_tokens": pipeline_result["estimated_request_tokens"],
            "writer": pipeline_result["report"],
            "critic": pipeline_result["feedback"],
        }
        pipeline_succeeded = True
    except RateLimitError:
        st.session_state.rate_limit_retry_at = (
            time.time() + RETRY_COOLDOWN_SECONDS
        )
    except (APIError, ValueError) as exc:
        st.error(f"Research failed: {exc}", icon=":material/error:")
    finally:
        st.session_state.running = False
        st.session_state.done = True

    if pipeline_succeeded:
        st.rerun()

retry_at = st.session_state.rate_limit_retry_at
if retry_at is not None:
    remaining = seconds_until_retry(retry_at)
    if remaining > 0:
        render_rate_limit_countdown(retry_at)
    else:
        st.success("The wait is over. You can try your research again.")
        if st.button(
            "Try again",
            type="primary",
            icon=":material/refresh:",
            key="retry_after_rate_limit",
        ):
            start_research()
            st.rerun()

results = st.session_state.results
if results:
    st.space("large")
    st.header("Research brief")
    if "estimated_request_tokens" in results:
        st.caption(
            "Estimated request budget: "
            f"{results['estimated_request_tokens']:,} tokens "
            "(prompt + completion allowance + safety margin; not actual usage)."
        )

    with st.container(border=True):
        st.markdown("#### :material/description: Findings")
        st.markdown(results["writer"])
        st.download_button(
            "Download report",
            data=results["writer"],
            file_name=f"quorum_research_report_{int(time.time())}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="content",
        )

    with st.container(border=True):
        st.markdown("#### :material/rate_review: Evidence review")
        st.markdown(results["critic"])

    with st.expander("Sources and page content"):
        st.markdown("**Search results**")
        st.text(results["search"])
        scraped_sources = results.get("scraped_sources", [])
        if scraped_sources:
            for index, source in enumerate(scraped_sources, start=1):
                st.markdown(f"**Source {index}: {source['title']}**")
                st.link_button(
                    "Open source",
                    source["url"],
                    icon=":material/open_in_new:",
                    key=f"scraped_source_{index}",
                )
                if source.get("status"):
                    st.caption(source["status"])
                st.text(source["content"])
        elif results.get("scraped_url"):
            st.markdown("**Scraped source**")
            st.link_button(
                "Open scraped source",
                results["scraped_url"],
                icon=":material/open_in_new:",
            )
            st.text(results["reader"])
        else:
            st.markdown("**Scraped sources**")
            st.warning(
                "The search results did not provide any source URLs to scrape.",
                icon=":material/link_off:",
            )
            st.text(results["reader"])

st.caption("Quorum · Evidence-led research, clearly presented.")
