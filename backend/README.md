# Research Agent Backend

Backend for the Research Agent app.

## Current Capabilities

- FastAPI API for health, paper management, chat, style profile, and retrieval preview
- LangGraph runtime pipeline: `prepare_mode -> retrieve -> rerank -> draft_answer -> validate_answer -> finalize_answer`
- Hybrid retrieval with reranking
- Reviewer Claim Trial Engine with persisted per-session debate state
- Comparator claim-matrix generation with structured section output
- Runtime-safe structured fallbacks when model providers fail

## Modes

- `Local`: strict grounded paper QA
- `Global`: open response style with optional paper grounding
- `Writer`: style-aware drafting
- `Reviewer`: Skeptic/Advocate/Judge/Rewrite workflow
- `Comparator`: multi-paper claim and benchmark comparison

## Provider Split

- Pinecone: dense vector store
- Embeddings: local hash (default) or Gemini
- Generation: Groq / Gemini / OpenRouter via router + fallback order

## Environment

Copy `backend/.env.example` to `backend/.env` and set relevant keys:

- `PINECONE_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `GENERATION_PROVIDER` (`auto` | `groq` | `gemini` | `openrouter`)
- `GENERATION_FALLBACK_ORDER` (default `gemini,openrouter,groq`)
- `GENERATION_PROVIDER_COOLDOWN_SECONDS` (default `600`)
- `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`)
- `EMBEDDING_PROVIDER` (`local` | `auto` | `gemini`; default `local`)

Optional tracing:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=research-agent`
- `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`

## Run

From `backend/`:

```powershell
uvicorn research_agent.api:app --reload --app-dir src
```

From workspace root:

```powershell
$env:PYTHONPATH='backend\src'
.\.venv\Scripts\python.exe -m uvicorn research_agent.api:app --port 8010
```

The same FastAPI server also serves the frontend UI at `/`, so once the backend is up you can
open `http://127.0.0.1:8010/` directly.

## Stress Script

```powershell
python .\backend\stress_test_outputs.py
```

This script emits JSON with reviewer/local/global/comparator stress metrics.

## Docs

- [`../README.md`](../README.md)
- [`../docs/architecture_walkthrough.md`](../docs/architecture_walkthrough.md)
- [`../docs/graph_state.md`](../docs/graph_state.md)
