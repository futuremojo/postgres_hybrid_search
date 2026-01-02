# Hybrid Search Demo Plan

## Intro
Build a single‑flow demo for hybrid search in Postgres 17 using pg_textsearch (BM25) + pgvector (embeddings). Goal: show real ingest, rank fusion, rerank, and query explainability in one Streamlit UI so behavior is visible, tunable, and reproducible.

## Why
Hybrid search boosts recall + relevance vs text‑only or vector‑only. This demo proves Postgres can do state‑of‑the‑art ranking with minimal infra and clear trade‑offs.

## Phases
- [x] Phase 0: Confirm pg_textsearch API
  - [x] Pull pg_textsearch docs/source, verify `<@>` operator, BM25 params, index type
  - [x] Update DB image to install official pg_textsearch (drop stub)

- [x] Phase 1: DB schema + indexes
  - [x] Tables: documents (hash dedup), chunks (text + embedding + metadata)
  - [x] Text index per pg_textsearch BM25
  - [x] Vector indexes: HNSW + IVFFlat, tunable params
  - [x] Helper views for explain and diagnostics

- [x] Phase 2: Ingest pipeline
  - [x] Parsers: PDF, DOCX, MD
  - [x] Chunking defaults (token size + overlap), english config
  - [x] OpenAI embeddings (default text‑embedding‑3‑small), env override
  - [x] Skip if hash exists; insert chunks

- [x] Phase 3: Query + hybrid
  - [x] Text search (BM25 via pg_textsearch)
  - [x] Vector search (pgvector)
  - [x] RRF merge w tunable k + weights
  - [x] Optional Cohere rerank (checkbox)
  - [x] Collect timing + rank positions

- [x] Phase 4: Streamlit UI
  - [x] Single flow: upload → process → query
  - [x] Tabs: RRF merged, Text fragments, Embedding hits
  - [x] Controls: index type, params, RRF k/weights, rerank toggle
  - [x] Explain plan + score breakdowns

- [x] Phase 5: Ops + docs
  - [x] Add deps to pyproject.toml
  - [x] Env vars: OPENAI_API_KEY, COHERE_API_KEY, model names
  - [x] README run steps + notes

- [x] Phase 6: UX/UI enhancements
  - [x] Ingestion should start immediately when the user uploads a file (or return an error on invalid format or let the user know it's already ingested)
  - [x] Add an ingestion progress bar
  - [x] Search should execute when the user hits entry in the query input
  - [x] The search button should be right beside the search box
  - [x] The use cohere rerank checkbox should be under the search box
