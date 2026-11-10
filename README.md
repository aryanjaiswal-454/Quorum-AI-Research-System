# Quorum

Quorum is a lightweight AI-assisted research app. It searches the web for up to
three sources, retrieves available page text, and uses Groq to produce a concise
report and evidence review.

## Features

- Web search through Tavily, with duplicate URLs removed.
- Retrieves page text for every result (up to three), using Tavily's extracted
  text when available and a direct-fetch fallback; labels search-excerpt-only
  results when a page cannot be fetched.
- Generates the report and critique in a single Groq model request.
- Includes a Streamlit interface and a command-line runner.
- Limits the research topic to 300 characters and bounds the text sent to the
  model.

## Requirements

- Python 3.14 or newer.
- A [Groq API key](https://console.groq.com/keys).
- A [Tavily API key](https://app.tavily.com/home).

## Setup

Run these commands from the project directory.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your keys:

```dotenv
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Keep `.env` private; it is excluded from version control.

If PowerShell prevents environment activation, allow scripts for the current
session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the Streamlit app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Without activating the environment, run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open the local URL printed by Streamlit (usually
<http://localhost:8501>). Enter a topic or choose an example, then select
**Run research**. The report, evidence review, source links, and scraped text
appear on the page. Use **Download report** to save the report as Markdown.

## Run from the command line

```powershell
.\.venv\Scripts\Activate.ps1
python pipeline.py
```

Without activating the environment, run:

```powershell
.\.venv\Scripts\python.exe pipeline.py
```

Enter a research topic when prompted. The pipeline prints search results,
scraped content, the report, and the critique to the terminal.

## How a research run works

1. Tavily searches the topic and returns up to three unique sources, requesting
   cleaned page text with the same search call.
2. When Tavily does not return page text, Quorum tries a direct fetch with an
   eight-second timeout. Text is limited to 6,500 characters per source, and a
   search excerpt is used and labeled if the page blocks direct fetching.
3. Quorum uses the GPT-OSS Harmony tokenizer to fit the topic, search results,
   and extracted page text into a prompt budget of up to 3,500 tokens. It trims
   page excerpts evenly across sources when needed.
4. Groq generates a report and critique in one request. The configured model
   is `openai/gpt-oss-120b`, with medium reasoning effort, a 3,200-token
   completion cap, and automatic retries disabled.

The app budgets each request as at most 3,500 prompt tokens, up to 3,200
completion tokens, and an 800-token safety margin: a 7,500-token estimate
against the 8,000 TPM limit shown for the model. Quorum displays this budget
estimate for each completed run; it is not the provider's actual usage. The
margin helps account for message formatting and token-estimation differences.
Other requests using the same account during the minute can still cause a rate
limit error. The dashboard shown for this Groq account also lists a 200,000 TPD
limit. That daily quota is shared with other account usage; Quorum does not
track or reserve it, so monitor usage in the Groq dashboard.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface |
| `pipeline.py` | Shared command-line and UI research workflow |
| `tools.py` | Tavily search, result formatting, and page scraping |
| `agents.py` | Groq model and report/critique prompt |
| `.streamlit/config.toml` | Quorum's Streamlit theme |
| `requirements.txt` | Python dependencies for pip |
| `pyproject.toml` / `uv.lock` | Project metadata and uv dependency lock |

## Notes

- Search and scraping require internet access and valid API keys.
- Some websites may block direct requests, return little text, or time out.
  Quorum uses Tavily's search excerpt when available and labels that limited
  evidence in the source list and report context.
- If Groq reaches the configured output limit, Quorum preserves the generated
  report and warns that the report or review may be incomplete.
- If Groq rate-limits a request, Quorum displays a 60-second countdown and
  enables a manual retry when the wait is over.
- AI-generated reports can contain errors or unsupported claims. Check the
  linked sources before relying on the findings.
- Never commit API keys or paste them into source files.