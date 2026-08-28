# Multi-Agent RAG Chatbot

A Streamlit chatbot that routes questions to specialist AI agents and combines their answers — built as portfolio project #6 in a LangChain/LangGraph learning series.

## What it does

Instead of one AI trying to answer everything, this app has a **supervisor** that reads your question, decides which specialist agent(s) are needed, and merges their answers into one response.

Example: *"What was Lionsgate's revenue this quarter, and how many horror movies came out in 2020?"*
→ Supervisor calls **two** agents (earnings PDF + movie database), then combines both answers into one.

## The three specialist agents

| Agent | What it knows |
|---|---|
| `pdf_rag_tool` | Lionsgate quarterly earnings reports (via Supabase vector search) |
| `sql_agent_tool` | IMDb movie database — ratings, votes, genres, years |
| `web_search_tool` | Live web search for current/recent news (via Tavily) |

## How the supervisor decides

This uses a **plan-then-execute** pattern:
1. The supervisor LLM reads the question once and outputs a JSON list of which tool(s) are needed
2. Each needed tool runs and returns its own answer
3. If more than one tool was used, a synthesis step combines the answers into one clear response (and flags it if sources disagree)

## Tech stack
- LangChain + LangGraph
- Groq (`openai/gpt-oss-120b`) for LLM calls
- Supabase pgvector for PDF retrieval
- SQLite + SQLDatabaseToolkit for movie data
- Tavily for web search
- Streamlit for the chat UI

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py


## Part of a larger series

This project builds on three earlier standalone agents (search, SQL, PDF-RAG) and a LangGraph router — this project reuses all three as specialists under one supervisor instead of a single-path router.
```

