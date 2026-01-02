# pg-hybrid-search-demo

Postgres 17 + pg_textsearch (BM25) + pgvector, single‑doc hybrid search UI in Streamlit.

## What it does
- Upload pdf/docx/markdown
- Chunk + embed with OpenAI
- BM25 + vector search, RRF merge, optional Cohere rerank
- Explain plans, score panes, overlap stats

## Run
1) Create `.env` from `.env.example`
2) Set keys:
   - `OPENAI_API_KEY`
   - `COHERE_API_KEY` (optional; required only if rerank enabled)
3) `docker compose down -v` (first run or schema changes)
4) `docker compose up --build`
5) Open `http://localhost:8501`

## Env vars
- `OPENAI_API_KEY` (required for ingest + search)
- `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`)
- `OPENAI_EMBED_DIM` (default `1536`)
- `OPENAI_EMBED_BATCH_SIZE` (default `64`)
- `CHUNK_TOKENS` (default `800`)
- `CHUNK_OVERLAP` (default `120`)
- `COHERE_API_KEY` (optional; required for rerank)
- `COHERE_RERANK_MODEL` (default `rerank-english-v3.0`)
- `APP_AUTH_USER` (optional; enable simple login)
- `APP_AUTH_PASSWORD` (optional; enable simple login)

## Notes
- pg_textsearch is prerelease; index format can change.
- HNSW supports max 2000 dims, so embedding dim is 1536.
- If Docker runs out of disk: `docker system prune -af --volumes`.
- Tour uses driver.js from jsdelivr CDN (browser needs net access).

---

## How It Works (Technical Deep Dive)

This section explains how the codebase works under the hood. You should know that hybrid search combines keyword search (lexical) with meaning-based search (semantic). This guide covers how that happens in PostgreSQL.

### Architecture Overview

The system has two main parts:

1. **Database** (PostgreSQL 17): Stores documents and runs both search types
2. **Application** (Python/Streamlit): Handles file uploads, calls the database, and shows results

Everything runs in Docker containers. The database has two special extensions:
- `pgvector`: Lets PostgreSQL store and search vector embeddings
- `pg_textsearch`: Adds BM25 text search (a smarter keyword search)

### The Two Types of Search

#### Lexical Search (BM25)

BM25 is a ranking algorithm that scores documents based on keyword matches. It improves on simple keyword search by considering:

- **Term frequency**: How often a word appears in a chunk
- **Document length**: Shorter documents with the same matches rank higher
- **Inverse document frequency**: Rare words matter more than common ones

The database uses the `<@>` operator to compute BM25 scores. Lower scores mean better matches (it returns a distance, not a similarity).

#### Semantic Search (Vector Embeddings)

Vector search finds text with similar meaning, even if the words are different. Here's how it works:

1. **Embedding**: Text is converted to a list of 1536 numbers (a vector) using OpenAI's API
2. **Storage**: Vectors are stored in the database alongside the text
3. **Similarity**: When you search, your query becomes a vector too, and the database finds chunks with similar vectors

The database uses the `<=>` operator for cosine distance. Lower values mean more similar.

### Database Schema

The database has two tables:

**documents**: Stores metadata about uploaded files
- `doc_hash`: SHA-256 hash to prevent duplicate uploads
- `filename`, `file_type`, `size_bytes`: Basic file info

**chunks**: Stores pieces of each document
- `content`: The text of this chunk
- `embedding`: A 1536-dimension vector representing the meaning
- `chunk_index`: Position in the original document
- `meta`: Extra info like token positions

### Indexes Explained

Indexes make searches fast. Without them, the database would scan every row.

#### BM25 Index

    CREATE INDEX chunks_bm25_idx
        ON chunks USING bm25 (content)
        WITH (text_config = 'english', k1 = 1.2, b = 0.75);

- `text_config = 'english'`: Uses English stemming (so "running" matches "run")
- `k1 = 1.2`: Controls term frequency saturation (higher = more credit for repeated words)
- `b = 0.75`: Controls document length normalization (higher = penalize longer documents more)

#### HNSW Index (Vector Search)

    CREATE INDEX chunks_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

HNSW (Hierarchical Navigable Small World) builds a graph of vectors for fast approximate search.

- `m = 16`: Each node connects to 16 neighbors (more = better recall, larger index)
- `ef_construction = 64`: How thoroughly the graph is built (more = better quality, slower build)

At query time, you can tune `ef_search` (how many nodes to explore). Higher values give better results but slower queries.

#### IVFFlat Index (Alternative Vector Search)

    CREATE INDEX chunks_ivfflat_idx
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

IVFFlat divides vectors into 100 clusters. At query time:
- `probes`: How many clusters to search (more = better recall, slower)

IVFFlat is faster to build than HNSW but usually has lower recall at the same speed.

### Ingestion Pipeline

When you upload a file:

1. **Hash check**: Compute SHA-256 hash. If it exists, skip (no duplicates).
2. **Extract text**: Parse PDF, DOCX, or Markdown into plain text.
3. **Chunk**: Split text into pieces of ~800 tokens with 120-token overlap.
   - Overlap helps preserve context at chunk boundaries.
4. **Embed**: Send chunks to OpenAI in batches of 64. Each returns a 1536-dim vector.
5. **Store**: Insert document metadata and all chunks with their embeddings.

### Search Pipeline

When you search:

1. **Text search**: Run BM25 query, get top K results with scores
2. **Embed query**: Convert your query to a vector via OpenAI
3. **Vector search**: Find top K closest vectors in the database
4. **Merge with RRF**: Combine the two result lists (see below)
5. **Optional rerank**: Use Cohere to re-score results based on relevance

### RRF (Reciprocal Rank Fusion)

RRF merges two ranked lists without needing to normalize scores. The formula:

    RRF_score = weight / (k + rank)

For each result:
- If it's ranked #1 in text search: `text_weight / (k + 1)`
- If it's ranked #3 in vector search: `vector_weight / (k + 3)`
- Add both scores if it appears in both lists

**Parameters**:
- `rrf_k`: Dampening constant (default 60). Higher values reduce the gap between ranks.
- `weight_text` and `weight_vector`: How much to favor each search type (default 1.0 each).

Results are sorted by total RRF score. Items found by both searches get higher scores.

### Cohere Reranking (Optional)

RRF is simple but doesn't understand the text. Cohere reranking uses a neural model to re-score results based on actual relevance to the query. This often improves results but adds latency and cost.

### Tunable Parameters in the UI

| Parameter | What it does |
|-----------|--------------|
| **Top K** | How many results to fetch from each search |
| **RRF K** | Dampening constant for rank fusion (higher = ranks matter less) |
| **Text weight** | How much to favor BM25 results |
| **Vector weight** | How much to favor embedding results |
| **Vector index** | Choose HNSW or IVFFlat |
| **ef_search** | HNSW search exploration (higher = better recall, slower) |
| **probes** | IVFFlat clusters to search (higher = better recall, slower) |
| **Force index** | Disable sequential scans (for testing index performance) |
| **Use Cohere rerank** | Re-score results with a neural model |

### Understanding the Results

The UI shows several views:

- **Score panes**: Side-by-side rankings from text, vector, and merged results
- **Overlap@k**: What fraction of text/vector results appear in the final merged list
- **Timing**: Milliseconds spent on each step (embed, text search, vector search, RRF, rerank)
- **Explain plan**: PostgreSQL's query plan showing whether indexes were used

### Why PostgreSQL?

Most hybrid search systems use separate databases (Elasticsearch for text, Pinecone for vectors). This demo shows PostgreSQL can do both:

- Single database = simpler operations
- Transactional consistency between text and vectors
- SQL for all queries
- pg_textsearch brings state-of-the-art BM25 to Postgres
- pgvector is mature and production-ready

The trade-off is scale. Dedicated vector databases may perform better at billions of vectors.

### File Structure

    pg_hybrid_search_demo/
    ├── app.py              # Main application (ingestion, search, UI)
    ├── pyproject.toml      # Python dependencies
    ├── docker-compose.yml  # Runs app + database
    ├── Dockerfile          # Python container
    ├── db/
    │   ├── Dockerfile      # Postgres 17 + extensions
    │   └── init/
    │       ├── 01-extensions.sql  # Enable pgvector, pg_textsearch
    │       └── 02-schema.sql      # Tables and indexes
    └── README.md           # This file
