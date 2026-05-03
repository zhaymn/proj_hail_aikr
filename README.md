# Research Agent — AI-Powered Research Paper Assistant

## Table of Contents

1. [Project Description](#project-description)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Installation Steps](#installation-steps)
8. [How to Run the Project](#how-to-run-the-project)
9. [Example Input / Output](#example-input--output)
10. [API Reference](#api-reference)
11. [Configuration Reference](#configuration-reference)

---

## Project Description

**Research Agent** is a full-stack, AI-powered research paper assistant designed to streamline academic workflows. Users upload research PDFs through a modern web dashboard and then interact with the papers through multiple specialised AI modes — each targeting a different stage of the research process.

The system ingests PDFs via semantic chunking, indexes them in a Pinecone vector database using hybrid (dense + sparse) retrieval with reranking, and orchestrates multi-step reasoning through LangGraph state machines. Answers are always grounded in the uploaded papers and include inline citations, making the tool suitable for evidence-based academic work.

The project is composed of two main components:

| Component | Technology | Entry Point |
|-----------|-----------|-------------|
| **Frontend** | React (JSX) served as a static bundle | `research_agent.jsx` |
| **Backend** | Python 3.11 + FastAPI + LangGraph | `backend/src/research_agent/api.py` |

---

## Key Features

| Mode | Purpose |
|------|---------|
| **Local Brain** | Strict paper-grounded question answering with inline citations. Only answers from the content of uploaded papers. |
| **Global Brain** | Open-ended answers that optionally draw on uploaded paper context for richer responses. |
| **Paper Writer** | Style-aware drafting and rewriting. Learns the user's writing style from uploaded papers and produces text in that voice. |
| **Reviewer** | A structured **Claim Trial Engine** — a Skeptic argues against claims, an Advocate defends them, and a Judge delivers an evidence-only verdict with a rewrite recommendation card. |
| **Comparator** | Claim-level comparison across 2–3 selected papers. Produces a Claim Matrix, Conflict Map, Benchmark Verdict Matrix, Method Trade-offs analysis, and Decision-by-Use-Case summary. |

Additional capabilities:

- **Hybrid Retrieval**: Combines dense (embedding-based) and sparse (keyword-based) retrieval with Reciprocal Rank Fusion (RRF) and neural reranking.
- **Semantic Chunking**: Splits PDFs into semantically coherent chunks, preserving paragraph-level meaning.
- **Multi-Provider LLM Fallback**: Automatically rotates between LLM providers when rate limits or failures occur, ensuring uninterrupted service.
- **Retrieval-Only Fallback**: If all LLM providers fail, the system returns structured retrieval results so the user is never left without an answer.

---

## System Architecture

### High-Level Overview

The application follows a three-tier architecture: a React frontend communicates with a FastAPI backend, which orchestrates AI reasoning through a LangGraph state machine.

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        USER (Web Browser)                          │
 └────────────────────────────┬────────────────────────────────────────┘
                              │  Upload PDFs / Ask Questions
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                     REACT FRONTEND                                  │
 │  research_agent.jsx                                                 │
 │  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌────────────────┐   │
 │  │ PDF      │  │ Mode       │  │ Chat      │  │ Citation       │   │
 │  │ Uploader │  │ Selector   │  │ Interface │  │ Viewer         │   │
 │  └──────────┘  └────────────┘  └───────────┘  └────────────────┘   │
 └────────────────────────────┬────────────────────────────────────────┘
                              │  HTTP (REST API)
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                     FASTAPI BACKEND                                 │
 │  api.py → runtime.py                                                │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
 │  │ Paper        │  │ Chat         │  │ Style Memory             │  │
 │  │ Management   │  │ Orchestration│  │ Service                  │  │
 │  └──────────────┘  └──────┬───────┘  └──────────────────────────┘  │
 │                           │                                         │
 │                           ▼                                         │
 │               ┌───────────────────────┐                             │
 │               │   LangGraph Engine    │                             │
 │               │   (State Machine)     │                             │
 │               └───────────┬───────────┘                             │
 │                           │                                         │
 │            ┌──────────────┼──────────────┐                          │
 │            ▼              ▼              ▼                          │
 │     ┌────────────┐ ┌───────────┐ ┌─────────────┐                   │
 │     │ Retrieval  │ │ Text Gen  │ │ Semantic    │                   │
 │     │ (Dense +   │ │ Service   │ │ Chunker     │                   │
 │     │  Sparse)   │ │ (Groq /   │ │             │                   │
 │     │            │ │  Gemini)  │ │             │                   │
 │     └─────┬──────┘ └───────────┘ └─────────────┘                   │
 │           │                                                         │
 │           ▼                                                         │
 │     ┌────────────┐                                                  │
 │     │ Pinecone   │                                                  │
 │     │ Vector DB  │                                                  │
 │     └────────────┘                                                  │
 └─────────────────────────────────────────────────────────────────────┘
```

### LangGraph Processing Pipeline

Every user query passes through a six-node LangGraph state machine. The graph is linear — each node feeds into the next, ensuring structured and validated outputs.

```
 ┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────┐
 │  START  │───>│ prepare_mode │───>│ retrieve │───>│ rerank  │───>│ draft_answer │───>│   validate   │───>│ END │
 └─────────┘    └──────────────┘    └──────────┘    └─────────┘    └──────────────┘    │   + finalize │    └─────┘
                       │                  │              │                │             └──────────────┘
                       │                  │              │                │
                 Set system         Query Pinecone   Score and      Generate answer
                 prompt based       using hybrid     filter top     with LLM using
                 on selected        dense + sparse   chunks by      retrieved context
                 mode (local,       retrieval with   relevance      and mode-specific
                 global, writer,    sub-queries      to user        instructions,
                 reviewer,          per mode         query          then validate
                 comparator)                                        citations and
                                                                    format output
```

### Reviewer Pipeline (Claim Trial Engine)

The Reviewer mode implements a multi-turn adversarial debate system for rigorous paper evaluation:

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │                     REVIEWER PIPELINE                                 │
 │                                                                       │
 │  User selects a paper + review focus (e.g. "novelty", "methodology") │
 │                              │                                        │
 │                              ▼                                        │
 │                    ┌───────────────────┐                               │
 │                    │  Load/Resume      │  Retrieves session state      │
 │                    │  Session State    │  (persisted across turns)     │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │                             ▼                                         │
 │                    ┌───────────────────┐                               │
 │                    │  Retrieve + Rank  │  Targeted sub-queries for     │
 │                    │  Evidence Chunks  │  novelty, method, evaluation  │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │                             ▼                                         │
 │                    ┌───────────────────┐                               │
 │                    │  Generate Attack  │  Identify weak claims in      │
 │                    │  Vectors          │  the paper to challenge       │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │              ┌──────────────┴──────────────┐                          │
 │              │    FOR EACH ATTACK VECTOR    │                          │
 │              │                              │                          │
 │              │  ┌────────────────────────┐  │                          │
 │              │  │ SKEPTIC argues claim   │  │                          │
 │              │  │ is weak/unsupported    │──┤  Multi-turn debate       │
 │              │  └────────────────────────┘  │  continues until         │
 │              │              │               │  resolution or           │
 │              │              ▼               │  turn cap reached        │
 │              │  ┌────────────────────────┐  │                          │
 │              │  │ ADVOCATE defends with  │  │                          │
 │              │  │ paper evidence         │──┘                          │
 │              │  └────────────────────────┘                             │
 │              │              │                                          │
 │              │              ▼                                          │
 │              │  ┌────────────────────────┐                             │
 │              │  │ JUDGE evaluates only   │  Evidence-only verdict:     │
 │              │  │ grounded evidence      │  upheld / partially /       │
 │              │  └────────────────────────┘  overturned                 │
 │              │              │                                          │
 │              │              ▼                                          │
 │              │  ┌────────────────────────┐                             │
 │              │  │ REWRITE CARD           │  Actionable fix             │
 │              │  │ compiled for author    │  recommendation             │
 │              │  └────────────────────────┘                             │
 │              └─────────────────────────────                            │
 │                             │                                          │
 │                             ▼                                          │
 │                    ┌───────────────────┐                               │
 │                    │  Final Reviewer   │  Aggregated report with       │
 │                    │  Report           │  all verdicts + rewrite cards │
 │                    └───────────────────┘                               │
 └───────────────────────────────────────────────────────────────────────┘
```

### Comparator Pipeline (Claim Matrix Lab)

The Comparator mode performs claim-level analysis across 2–3 papers:

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │                    COMPARATOR PIPELINE                                │
 │                                                                       │
 │  User selects 2–3 papers + comparison question                       │
 │                              │                                        │
 │                              ▼                                        │
 │                    ┌───────────────────┐                               │
 │                    │  Per-Paper        │  5 sub-queries per paper:     │
 │                    │  Retrieval Merge  │  scope, method, evaluation,  │
 │                    │                   │  limitations, contributions  │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │                             ▼                                         │
 │                    ┌───────────────────┐                               │
 │                    │  Rerank + Dedupe  │  Remove duplicates, filter   │
 │                    │                   │  references & boilerplate    │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │                             ▼                                         │
 │                    ┌───────────────────┐                               │
 │                    │  Generate         │  LLM produces structured     │
 │                    │  Comparison       │  multi-section output:       │
 │                    │  Report           │                               │
 │                    └────────┬──────────┘                               │
 │                             │                                         │
 │                ┌────────────┼────────────────┐                        │
 │                ▼            ▼                ▼                        │
 │         ┌───────────┐ ┌──────────┐ ┌──────────────────┐              │
 │         │ Claim     │ │ Conflict │ │ Benchmark Verdict│              │
 │         │ Matrix    │ │ Map      │ │ Matrix           │              │
 │         └───────────┘ └──────────┘ └──────────────────┘              │
 │                ▼            ▼                ▼                        │
 │         ┌───────────┐ ┌──────────┐ ┌──────────────────┐              │
 │         │ Method    │ │Synthesis │ │ Decision by      │              │
 │         │ Trade-offs│ │Blueprint │ │ Use Case         │              │
 │         └───────────┘ └──────────┘ └──────────────────┘              │
 │                             │                                         │
 │                             ▼                                         │
 │                    ┌───────────────────┐                               │
 │                    │  Validate +       │  Citation reindexing +       │
 │                    │  Finalize         │  quality check               │
 │                    └───────────────────┘                               │
 └───────────────────────────────────────────────────────────────────────┘
```

### Hybrid Retrieval System

The retrieval layer combines two complementary search strategies to maximise recall:

```
                         User Query
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌───────────────┐        ┌───────────────┐
       │ Dense Search   │        │ Sparse Search  │
       │ (Embedding     │        │ (Keyword /     │
       │  Similarity)   │        │  BM25 Match)   │
       └───────┬───────┘        └───────┬───────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Reciprocal Rank   │
                  │ Fusion (RRF)      │
                  │ Merges both       │
                  │ ranked lists      │
                  └────────┬──────────┘
                           │
                           ▼
                  ┌───────────────────┐
                  │ Neural Reranker   │
                  │ Scores final      │
                  │ relevance         │
                  └────────┬──────────┘
                           │
                           ▼
                  Top-K relevant chunks
                  passed to LLM for
                  answer generation
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18 (JSX), HTML5, CSS3 |
| **Backend Framework** | FastAPI, Uvicorn |
| **AI Orchestration** | LangGraph, LangChain |
| **LLM Providers** | Groq (Llama 3.3 70B), Google Gemini 2.0 Flash |
| **Embeddings** | Local hash-based embeddings (default) or Gemini embeddings |
| **Vector Database** | Pinecone (serverless) |
| **PDF Processing** | PyPDF + custom semantic chunker |
| **Language** | Python 3.11+, JavaScript (ES2022) |

---

## Repository Structure

```
research_agent-main/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependency list (pip-installable)
│
├── index.html                       # Application entry point
├── landing.html                     # Landing page
├── dashboard.html                   # Main application shell
├── research_agent.jsx               # React frontend (all UI components)
│
├── backend/
│   ├── pyproject.toml               # Python project metadata & dependencies
│   ├── .env.example                 # Template for environment variables
│   └── src/research_agent/
│       ├── api.py                   # FastAPI application & route definitions
│       ├── runtime.py               # Core runtime: paper mgmt, chat orchestration
│       ├── schemas.py               # Pydantic request/response models
│       ├── config.py                # Settings (env vars, defaults, validation)
│       ├── graph/
│       │   ├── state.py             # LangGraph state definitions
│       │   └── builder.py           # Graph construction (6-node pipeline)
│       ├── retrieval/
│       │   ├── ingestion.py         # PDF upload, storage, text extraction
│       │   ├── chunking.py          # Semantic chunking logic
│       │   ├── dense.py             # Dense (embedding) retrieval + Pinecone
│       │   └── sparse.py            # Sparse (keyword/BM25) retrieval
│       └── services/
│           ├── text_generation.py   # Multi-provider LLM generation with fallback
│           ├── groq_text.py         # Groq provider adapter
│           ├── gemini_text.py       # Gemini provider adapter
│           └── style_memory.py      # Writing-style profile extraction
│
└── docs/
    ├── architecture.md              # High-level architecture document
    ├── architecture_walkthrough.md  # Detailed implementation walkthrough
    └── graph_state.md               # LangGraph state reference
```

---

## Prerequisites

Before installing, ensure you have the following:

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| **Python** | 3.11 | `python --version` |
| **pip** | 23.0+ | `pip --version` |
| **Git** | Any | `git --version` |

You will also need API keys for the following services:

| Service | Purpose | Required? |
|---------|---------|-----------|
| **Gemini** (`GEMINI_API_KEY`) | LLM generation + optional embeddings | Yes |
| **Groq** (`GROQ_API_KEY`) | Fast LLM generation via Llama 3.3 70B | Recommended |
| **Pinecone** (`PINECONE_API_KEY`) | Vector database for dense retrieval | Yes |

---

## Installation Steps

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/research_agent-main.git
cd research_agent-main
```

### Step 2 — Create a Python Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install -e ./backend
```

This installs all required packages defined in `backend/pyproject.toml`, including FastAPI, LangGraph, LangChain, Pinecone, PyPDF, and the Gemini/Groq client libraries.

Alternatively, install from the flat requirements file:

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
# Windows (PowerShell)
Copy-Item .\backend\.env.example .\backend\.env

# macOS / Linux
cp ./backend/.env.example ./backend/.env
```

Open `backend/.env` in a text editor and fill in your API keys:

```env
GEMINI_API_KEY=your-gemini-key-here
PINECONE_API_KEY=your-pinecone-key-here
GROQ_API_KEY=your-groq-key-here
GENERATION_PROVIDER=auto
EMBEDDING_PROVIDER=local
```

---

## How to Run the Project

### 1. Start the Backend Server

From the repository root, with your virtual environment activated:

```bash
# Windows (PowerShell)
.\.venv\Scripts\python.exe -m uvicorn research_agent.api:app --app-dir backend\src --host 127.0.0.1 --port 8010

# macOS / Linux
python -m uvicorn research_agent.api:app --app-dir backend/src --host 127.0.0.1 --port 8010
```

You should see output similar to:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8010 (Press CTRL+C to quit)
```

### 2. Verify the Server Is Running

```bash
curl http://127.0.0.1:8010/health
```

Expected response:

```json
{
  "status": "ok",
  "app_name": "Research Agent",
  "environment": "development",
  "graph_ready": true,
  "llm_available": true,
  "indexed_papers": 0
}
```

### 3. Open the Application

Open your web browser and navigate to:

| URL | Page |
|-----|------|
| `http://127.0.0.1:8010/` | Landing page |
| `http://127.0.0.1:8010/dashboard` | Main application dashboard |

---

## Example Input / Output

### Example 1: Uploading a Paper

**Input:** Upload a PDF file (e.g., `attention_is_all_you_need.pdf`) via the dashboard's upload panel or via the API:

```bash
curl -X POST http://127.0.0.1:8010/api/papers/upload \
  -F "files=@attention_is_all_you_need.pdf"
```

**Output:**

```json
{
  "papers": [
    {
      "paper_id": "a1b2c3d4",
      "filename": "attention_is_all_you_need.pdf",
      "stored_path": "backend/storage/papers/a1b2c3d4.pdf",
      "text_path": "backend/storage/text/a1b2c3d4.txt",
      "chunk_count": 47,
      "char_count": 52340,
      "uploaded_at": "2026-05-03T12:00:00Z"
    }
  ]
}
```

### Example 2: Local Brain — Paper-Grounded QA

**Input (via dashboard chat or API):**

```json
{
  "session_id": "session-001",
  "mode": "local",
  "message": "What is the main contribution of this paper?",
  "paper_ids": ["a1b2c3d4"]
}
```

**Output:**

```json
{
  "session_id": "session-001",
  "mode": "local",
  "answer": "The main contribution of this paper is the Transformer architecture, which relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely. The authors demonstrate that this model achieves superior performance on machine translation benchmarks while being significantly more parallelizable and requiring less training time [1].",
  "citations": [
    {
      "paper_id": "a1b2c3d4",
      "filename": "attention_is_all_you_need.pdf",
      "snippet": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...",
      "page": 1
    }
  ]
}
```

### Example 3: Global Brain — Open-Ended Query

**Input:**

```json
{
  "session_id": "session-002",
  "mode": "global",
  "message": "How does self-attention compare to recurrent architectures for sequence modelling?",
  "paper_ids": ["a1b2c3d4"]
}
```

**Output:**

```json
{
  "session_id": "session-002",
  "mode": "global",
  "answer": "Self-attention offers several advantages over recurrent architectures:\n\n1. Parallelisation: Unlike RNNs which process tokens sequentially, self-attention computes all positions simultaneously [1].\n\n2. Long-range dependencies: Self-attention connects any two positions with O(1) operations, compared to O(n) for RNNs [1].\n\n3. Interpretability: Attention weights show which tokens the model focuses on.\n\nHowever, self-attention has O(n²) memory complexity, which can limit very long sequences.",
  "citations": [
    {
      "paper_id": "a1b2c3d4",
      "filename": "attention_is_all_you_need.pdf",
      "snippet": "Self-attention is an attention mechanism relating different positions of a single sequence...",
      "page": 3
    }
  ]
}
```

### Example 4: Comparator — Multi-Paper Comparison

**Input:**

```json
{
  "session_id": "session-003",
  "mode": "comparator",
  "message": "Compare the training methodology and computational requirements.",
  "paper_ids": ["a1b2c3d4", "e5f6g7h8"]
}
```

**Output (abbreviated):**

```json
{
  "session_id": "session-003",
  "mode": "comparator",
  "answer": "## Papers Compared\n| Paper | Key Focus |\n|-------|-----------|\n| attention_is_all_you_need.pdf | Transformer architecture |\n| bert_pretraining.pdf | Bidirectional pre-training |\n\n## Claim Matrix\n| Claim | Paper 1 | Paper 2 | Agreement |\n|-------|---------|---------|-----------|\n| Self-attention is sufficient | Supported | Builds upon | Aligned |\n| Pre-training improves downstream | Not addressed | Core claim | N/A |\n\n## Method Trade-offs\n...\n\n## Decision by Use Case\n...",
  "citations": [ ... ]
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check and status |
| `GET` | `/api/papers` | List all uploaded papers |
| `POST` | `/api/papers/upload` | Upload one or more PDF files |
| `DELETE` | `/api/papers` | Delete all papers and clear index |
| `DELETE` | `/api/papers/{paper_id}` | Delete a specific paper |
| `POST` | `/api/papers/{paper_id}/re-ingest` | Re-process a previously uploaded paper |
| `GET` | `/api/style-profile` | View the current writing-style profile |
| `POST` | `/api/chat` | Send a query in any mode (local, global, writer, reviewer, comparator) |
| `POST` | `/api/retrieval/preview` | Preview retrieval results for a query |

---

## Configuration Reference

All configuration is managed via environment variables in `backend/.env`.

### LLM Provider Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_PROVIDER` | `auto` | Active provider: `auto`, `groq`, `gemini` |
| `GENERATION_FALLBACK_ORDER` | `gemini,groq` | Fallback chain when `auto` is selected |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model identifier for Groq |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model identifier for Gemini |
| `EMBEDDING_PROVIDER` | `local` | Embedding source: `local` (hash-based) or `gemini` |

### Retrieval Tuning

| Variable | Description |
|----------|-------------|
| `retrieval_top_k` | Number of chunks to retrieve before reranking |
| `rerank_top_n` | Number of chunks kept after reranking |
| `hybrid_dense_weight` / `hybrid_sparse_weight` | Blending weights for dense vs. sparse retrieval |
| `chunk_size` / `chunk_overlap` | Chunking parameters for PDF ingestion |

### Reviewer Tuning

| Variable | Description |
|----------|-------------|
| `reviewer_attack_vector_count` | Number of claim vectors to evaluate |
| `reviewer_max_turns` | Maximum debate rounds per vector |
| `reviewer_warning_turn` | Turn at which resolution pressure is applied |
