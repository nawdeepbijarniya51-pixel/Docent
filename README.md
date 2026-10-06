# Docent — Talk to any PDF

A production-style, async, hybrid-retrieval RAG chatbot. Upload any PDF, and ask it questions — grounded strictly in the document's content, with real (not simulated) pipeline progress and cited source snippets.

> **Stack:** FastAPI · LangChain · Groq (Qwen models) · Hugging Face embeddings · Qdrant (hybrid dense + sparse) · Cohere Rerank · vanilla JS frontend

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The RAG Pipeline in Detail](#the-rag-pipeline-in-detail)
  - [1. Ingestion Pipeline](#1-ingestion-pipeline)
  - [2. Query Pipeline](#2-query-pipeline)
  - [3. Query Routing](#3-query-routing)
  - [4. Hybrid Retrieval](#4-hybrid-retrieval)
  - [5. Reranking](#5-reranking)
  - [6. Answer Generation](#6-answer-generation)
- [Async Job Architecture](#async-job-architecture)
- [Model Configuration](#model-configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)

---

## Overview

Most "chat with your PDF" demos do naive single-vector similarity search and call it RAG. This project instead implements a **multi-stage retrieval pipeline** that mirrors how production RAG systems work in practice.

- **Query routing** — every question is classified before retrieval even starts, so vague or compound questions get rewritten/decomposed instead of silently returning bad results.
- **Hybrid search** — dense (semantic) + sparse (BM25/keyword) retrieval combined, not just cosine similarity on embeddings alone.
- **Cross-encoder reranking** — a second, more expensive but more accurate model re-scores retrieved candidates before they ever reach the LLM's context window.
- **Grounded generation** — the answer model is explicitly instructed to refuse to answer beyond what's in the retrieved context, with source attribution back to page numbers.
- **Real, staged progress reporting** — both ingestion and querying report actual pipeline stage/progress to the frontend (not a fake timed animation), via a polling job-status API.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion["📄 Ingestion Pipeline (/upload)"]
        A[PDF Upload] --> B[Parse: PyPDFLoader]
        B --> C[Chunk: RecursiveCharacterTextSplitter<br/>1000 chars, 100 overlap]
        C --> D{Collection exists<br/>for this content hash?}
        D -- Yes --> E[Reuse existing Qdrant collection]
        D -- No --> F[Embed chunks in batches<br/>Dense: Hugging Face BGE-M3<br/>Sparse: Qdrant/bm25]
        F --> G[Upsert into Qdrant<br/>Hybrid collection]
        E --> H[Session created]
        G --> H
    end

    subgraph Query["💬 Query Pipeline (/chat)"]
        Q[User question] --> R[Route: NoChange / Rephrase / MultiQuery<br/>Groq Qwen model]
        R -->|NoChange| S1[Use query as-is]
        R -->|Rephrase| S2[Rewrite: Groq Qwen model]
        R -->|MultiQuery| S3[Decompose into 2-4 sub-queries]
        S1 --> T[Hybrid Retrieval<br/>top-20 per query]
        S2 --> T
        S3 --> T
        T --> U[Rerank: Cohere rerank-v4.0-fast<br/>top-5]
        U --> V[Generate Answer: Groq Qwen model<br/>context-grounded]
        V --> W[Answer + Sources]
    end

    H -.retriever bundle.-> T
```

**Both pipelines are exposed as async, pollable background jobs** — `POST /upload` and `POST /chat` return a `job_id` immediately, and the frontend polls `/upload/status/{job_id}` or `/chat/status/{job_id}`.

---

## The RAG Pipeline in Detail

### 1. Ingestion Pipeline

| Stage | What happens | Reported progress |
|---|---|---|
| `parsing` | `PyPDFLoader` extracts text per page | 5% |
| `chunking` | `RecursiveCharacterTextSplitter` splits pages into overlapping chunks (`chunk_size=1000`, `chunk_overlap=100`) | 15–25% |
| `embedding` | Chunks embedded in ~10 batches (dense + sparse) and upserted into Qdrant incrementally, so progress reflects real work done | 25–90% |
| `storing` | Vector index finalized | 90–100% |
| `done` | Retriever built, chat session created | 100% |

**Deduplication by content hash.** Every uploaded PDF is SHA-256 hashed; the Qdrant collection is named `rag_{content_hash}`. If the exact same file content has already been indexed, the pipeline reuses the collection instead of re-embedding.

### 2. Query Pipeline

Every chat message runs through five distinct, independently-reported stages:

```
routing → branching → retrieving → reranking → answering
```

Rather than one opaque LLM call, each stage is a separate, inspectable step — which is also what makes the real-time progress UI possible.

### 3. Query Routing

Before any retrieval happens, a lightweight router model (Groq-hosted Qwen) classifies the incoming question into exactly one of three categories:

| Label | Meaning | Action taken |
|---|---|---|
| `NoChange` | Query is already clear, specific, well-formed | Used as-is |
| `Rephrase` | Query is vague, too short, or missing keywords | Rewritten into a single clearer, keyword-rich query |
| `MultiQuery` | Query contains multiple sub-questions or a comparison | Decomposed into 2–4 diverse sub-queries |

This matters because a single embedding of a vague or compound question is a genuinely bad retrieval query — routing catches this *before* it silently degrades answer quality.

All routing/rewriting/decomposition steps enforce structured output via Pydantic output parsers (`RouteOutput`, `ConditionalChainOutput`) — the LLM is constrained to return valid JSON matching a fixed schema.

### 4. Hybrid Retrieval

Each (possibly rewritten/decomposed) query is retrieved against Qdrant in **hybrid mode**:

- **Dense vectors** — Hugging Face embeddings (e.g. `BAAI/bge-m3`), capturing semantic similarity
- **Sparse vectors** — `Qdrant/bm25` (via FastEmbed), capturing exact keyword/term overlap

This combination is deliberate: dense embeddings alone often miss exact terminology (model numbers, clause names, proper nouns) that a user's question quotes directly, while pure keyword search misses semantic intent.

### 5. Reranking

The top-20 hybrid candidates are **not** sent directly to the LLM. Instead, they're passed through `CohereRerank` (`rerank-v4.0-fast`), a cross-encoder model that jointly scores each (query, passage) pair and keeps only the most relevant top-5.

### 6. Answer Generation

The final context is assembled from reranked passages and passed to a Groq-hosted Qwen model under a strict system prompt that enforces:

- Answer **only** from the provided context — no outside knowledge
- Explicit refusal (`"I don't have enough information in the provided context to answer this question."`) rather than guessing when context is insufficient
- Flag conflicting information across chunks instead of silently picking one
- Markdown-formatted output (bold key terms, tables for comparisons, code blocks where relevant)
- Never fabricate sources, numbers, or citations

**Multi-query answers** (from the `MultiQuery` route) run this generation step once per sub-query, and the results are joined with `---` separators — so a compound question gets a structurally clear answer.

**Source attribution:** every reranked chunk that contributed to an answer is returned with its page number, rerank relevance score, and a text snippet, letting the frontend show exactly where each answer came from.

**Chat history:** prior turns are trimmed (`trim_messages`, max 7000 tokens, keeps most recent) before being passed into the router — so routing decisions are context-aware without unbounded prompt growth.

---

## Async Job Architecture

Both `/upload` and `/chat` are long-running operations (embedding calls, multiple LLM round-trips, reranking) — too slow to hold open a single HTTP request for. Instead:

1. `POST /upload` / `POST /chat` validate the request, spin up a **background thread**, and return `{job_id}` immediately.
2. The frontend polls `GET /upload/status/{job_id}` or `GET /chat/status/{job_id}` every 300–400ms.
3. Each pipeline stage reports its own progress via an `on_progress`/`on_stage` callback threaded all the way through `rag_pipeline.py`, giving the frontend **real** stage/percentage data — not a fake animation.
4. The job only transitions to `stage: "done"` once *all* post-processing (session creation, chat history append) is complete.

```python
JOBS: Dict[str, dict]              # job_id -> {stage, progress, message, result, error}
SESSIONS: Dict[str, dict]          # session_id -> {pdf_hash, filename, chat_history}
RETRIEVER_CACHE: Dict[str, ...]    # pdf_hash -> RetrieverBundle (base retriever + reranker)
```

---

## Model Configuration

| Role | Model | Purpose |
|---|---|---|
| Dense embeddings | Hugging Face `BAAI/bge-m3` | Semantic vector search |
| Sparse embeddings | `Qdrant/bm25` (FastEmbed) | Keyword/term-overlap search |
| Router | Groq-hosted Qwen model | Fast query classification |
| Query rewriter | Groq-hosted Qwen model | Rephrase / multi-query generation |
| Answer generation | Groq-hosted Qwen model | Final grounded answer synthesis |
| Reranker | `rerank-v4.0-fast` (Cohere) | Cross-encoder relevance scoring |
| Vector store | Qdrant (hybrid mode) | Dense + sparse index |

All of these are surfaced live to the frontend via `GET /config`, so the UI's "model configuration" panel is never hardcoded — it reflects whatever's actually running server-side.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks required env vars are set |
| `/config` | GET | Returns actual model identifiers in use |
| `/upload` | POST | Starts PDF ingestion, returns `{job_id}` |
| `/upload/status/{job_id}` | GET | Poll ingestion progress (`parsing → chunking → embedding → storing → done`) |
| `/chat` | POST | Starts answering a question, returns `{job_id}` |
| `/chat/status/{job_id}` | GET | Poll answer progress (`routing → branching → retrieving → reranking → answering → done`) |
| `/sessions/{id}/history` | GET | Fetch a session's chat history |
| `/sessions/{id}` | DELETE | Drop a session's in-memory state |

---

## Project Structure

```
.
├── main.py            # FastAPI app: routes, async job orchestration, CORS
├── rag_pipeline.py    # Core RAG logic: ingestion, routing, retrieval, rerank, generation
├── index.html         # Self-contained frontend (no build step)
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLM calls for routing, rewriting, answer generation |
| `HF_API_TOKEN` | Yes | Hugging Face embeddings access |
| `COHERE_API_KEY` | Yes | Reranking |
| `QDRANT_API_KEY` | Yes | Vector store auth |
| `QDRANT_URL` | No | Defaults to a preconfigured Qdrant Cloud cluster |
| `CORS_ORIGINS` | No | Comma-separated allowed frontend origins |

See `.env.example`.

---

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn main:app --reload --port 8000
```

Then open `index.html` in a local static server (e.g. VS Code Live Server) — it defaults to `http://127.0.0.1:8000` as the API base for local dev.

---

## Deployment

- **Backend:** Render (`uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`)
- **Frontend:** GitHub Pages (static, no build step)

> **Important:** the backend must run with a **single worker**. `JOBS`, `SESSIONS`, and `RETRIEVER_CACHE` are in-memory dicts — multiple worker processes would each have their own copy, causing inconsistent state.

---

## Known Limitations

- **In-memory state** — sessions, jobs, and the retriever cache do not survive a process restart, and won't scale correctly across multiple worker processes or instances.
- **Ephemeral disk** — uploaded PDFs are stored on local disk; on platforms like Render's free tier, this is wiped on every restart/redeploy.
- **Cold starts** — all models and the Qdrant client are initialized at module import time; the first request after an idle period (e.g. Render free tier spin-down) will be noticeably slower.
- **No auth** — there is no user authentication or per-user isolation; any session ID is retrievable by anyone who has it.
