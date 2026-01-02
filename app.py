"""Hybrid search demo app."""

import hashlib
import io
import json
import logging
import os
import time
from typing import Callable, Iterable

import cohere
import psycopg
from psycopg.rows import dict_row
import streamlit as st
import tiktoken
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("pg_hybrid_search_demo")

DB_URL = os.getenv("DATABASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("OPENAI_EMBED_DIM", "1536"))
EMBED_BATCH = int(os.getenv("OPENAI_EMBED_BATCH_SIZE", "64"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
COHERE_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")


def check_db():
    """Check required extensions."""
    if not DB_URL:
        return "DATABASE_URL missing", False
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT extname
                    FROM pg_extension
                    WHERE extname IN ('vector', 'pg_textsearch')
                    ORDER BY extname;
                    """
                )
                ext = {row[0] for row in cur.fetchall()}
        missing = [name for name in ("vector", "pg_textsearch") if name not in ext]
        if missing:
            return f"missing extensions: {', '.join(missing)}", False
        return "extensions ok: pg_vector (vector), pg_textsearch", True
    except Exception as exc:
        return f"db error: {exc}", False


def detect_file_type(filename: str) -> str:
    """Infer supported file type from filename."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in (".md", ".markdown"):
        return "markdown"
    raise ValueError(f"unsupported file type: {ext or 'unknown'}")


def read_pdf(data: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n\n".join(parts)


def read_docx(data: bytes) -> str:
    """Extract text from DOCX bytes."""
    doc = Document(io.BytesIO(data))
    parts = [para.text for para in doc.paragraphs if para.text]
    return "\n\n".join(parts)


def read_markdown(data: bytes) -> str:
    """Decode markdown bytes."""
    return data.decode("utf-8", errors="replace")


def extract_text(filename: str, data: bytes) -> str:
    """Extract text based on file type."""
    file_type = detect_file_type(filename)
    if file_type == "pdf":
        text = read_pdf(data)
    elif file_type == "docx":
        text = read_docx(data)
    else:
        text = read_markdown(data)
    return text.replace("\x00", " ").strip()


def chunk_text(text: str) -> list[dict]:
    """Chunk text into token windows."""
    if CHUNK_TOKENS <= CHUNK_OVERLAP:
        raise ValueError("CHUNK_TOKENS must be > CHUNK_OVERLAP")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        if not chunk_text:
            continue
        chunks.append(
            {
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                "token_count": len(chunk_tokens),
            }
        )
    return chunks


def embed_texts(
    texts: list[str],
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed texts with OpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")
    LOGGER.debug("embed start count=%d model=%s", len(texts), OPENAI_MODEL)
    client = OpenAI()
    embeddings: list[list[float]] = []
    total = len(texts)
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        data = sorted(resp.data, key=lambda item: item.index)
        embeddings.extend([item.embedding for item in data])
        if progress_cb:
            progress_cb(min(i + len(batch), total), total)
    if embeddings and len(embeddings[0]) != EMBED_DIM:
        raise RuntimeError(f"embedding dim mismatch: {len(embeddings[0])} != {EMBED_DIM}")
    LOGGER.debug("embed done count=%d", len(embeddings))
    return embeddings


def vector_literal(vec: Iterable[float]) -> str:
    """Format vector for SQL."""
    return "[" + ",".join(f"{value:.8f}" for value in vec) + "]"




def format_snippet(text: str, max_chars: int = 140) -> str:
    """Shorten text for previews."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def apply_vector_settings(cur, index_mode: str, ef_search: int, probes: int, force_index: bool) -> None:
    """Apply pgvector query settings."""
    if force_index:
        cur.execute("SET LOCAL enable_seqscan = off;")
        cur.execute("SET LOCAL enable_bitmapscan = off;")
    if index_mode == "hnsw":
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)};")
    else:
        cur.execute(f"SET LOCAL ivfflat.probes = {int(probes)};")


def ingest_file(
    uploaded,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Ingest and embed a file."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    data = uploaded.getvalue()
    filename = uploaded.name or "upload"
    file_type = detect_file_type(filename)
    doc_hash = hashlib.sha256(data).hexdigest()
    size_bytes = len(data)
    LOGGER.info(
        "ingest start filename=%s size=%d hash=%s",
        filename,
        size_bytes,
        doc_hash[:12],
    )
    if progress_cb:
        progress_cb(0.05, "hashing")

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename FROM documents WHERE doc_hash = %s", (doc_hash,))
            row = cur.fetchone()
            if row:
                LOGGER.info("ingest skip exists doc_id=%s", row[0])
                if progress_cb:
                    progress_cb(1.0, "already ingested")
                return {
                    "status": "exists",
                    "doc_id": row[0],
                    "doc_hash": doc_hash,
                    "filename": row[1],
                }

    text = extract_text(filename, data)
    if not text.strip():
        raise RuntimeError("no text extracted")
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("no chunks created")
    LOGGER.info("ingest chunks=%d", len(chunks))
    if progress_cb:
        progress_cb(0.15, "chunking")

    def embed_progress(done: int, total: int) -> None:
        if not progress_cb or total == 0:
            return
        ratio = 0.15 + (done / total) * 0.7
        progress_cb(ratio, f"embedding {done}/{total}")

    embeddings = embed_texts([chunk["text"] for chunk in chunks], embed_progress)
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding count mismatch")
    if progress_cb:
        progress_cb(0.9, "storing")

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (doc_hash, filename, file_type, size_bytes)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (doc_hash, filename, file_type, size_bytes),
            )
            doc_id = cur.fetchone()[0]
            rows = []
            for idx, chunk in enumerate(chunks):
                meta = {
                    "filename": filename,
                    "file_type": file_type,
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                }
                rows.append(
                    (
                        doc_id,
                        idx,
                        chunk["text"],
                        chunk["token_count"],
                        vector_literal(embeddings[idx]),
                        json.dumps(meta),
                    )
                )
            cur.executemany(
                """
                INSERT INTO chunks
                    (doc_id, chunk_index, content, content_tokens, embedding, meta)
                VALUES
                    (%s, %s, %s, %s, %s::vector, %s::jsonb);
                """,
                rows,
            )
    LOGGER.info("ingest done doc_id=%s chunks=%d", doc_id, len(chunks))
    if progress_cb:
        progress_cb(1.0, "done")
    return {
        "status": "inserted",
        "doc_id": doc_id,
        "doc_hash": doc_hash,
        "filename": filename,
        "chunk_count": len(chunks),
    }


def list_documents() -> list[dict]:
    """List stored documents."""
    if not DB_URL:
        return []
    with psycopg.connect(DB_URL) as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id,
                       d.filename,
                       d.file_type,
                       d.size_bytes,
                       d.created_at,
                       d.doc_hash,
                       COALESCE(v.chunk_count, 0) AS chunk_count
                FROM documents d
                LEFT JOIN v_chunk_counts v ON v.doc_id = d.id
                ORDER BY d.created_at DESC;
                """
            )
            return cur.fetchall()


def search_text(doc_id: int, query: str, limit: int) -> list[dict]:
    """Run BM25 text search."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    LOGGER.debug("text search doc_id=%s limit=%d", doc_id, limit)
    with psycopg.connect(DB_URL) as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       doc_id,
                       chunk_index,
                       content,
                       content <@> to_bm25query(%s, 'chunks_bm25_idx') AS text_score
                FROM chunks
                WHERE doc_id = %s
                ORDER BY content <@> to_bm25query(%s, 'chunks_bm25_idx')
                LIMIT %s;
                """,
                (query, doc_id, query, limit),
            )
            rows = cur.fetchall()
    results = []
    for idx, row in enumerate(rows, start=1):
        text_score = float(row["text_score"])
        results.append(
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "text_score": text_score,
                "bm25_score": -text_score,
                "rank_text": idx,
            }
        )
    return results


def search_vector(
    doc_id: int,
    query_vec: list[float],
    limit: int,
    index_mode: str,
    ef_search: int,
    probes: int,
    force_index: bool,
) -> list[dict]:
    """Run vector similarity search."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    LOGGER.debug(
        "vector search doc_id=%s limit=%d mode=%s",
        doc_id,
        limit,
        index_mode,
    )
    vec = vector_literal(query_vec)
    with psycopg.connect(DB_URL) as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            apply_vector_settings(cur, index_mode, ef_search, probes, force_index)
            cur.execute(
                """
                SELECT id,
                       doc_id,
                       chunk_index,
                       content,
                       embedding <=> %s::vector AS vector_distance
                FROM chunks
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vec, doc_id, vec, limit),
            )
            rows = cur.fetchall()
    results = []
    for idx, row in enumerate(rows, start=1):
        results.append(
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "vector_distance": float(row["vector_distance"]),
                "rank_vector": idx,
            }
        )
    return results


def merge_rrf(
    text_results: list[dict],
    vector_results: list[dict],
    rrf_k: int,
    weight_text: float,
    weight_vector: float,
) -> list[dict]:
    """Merge rankings with RRF."""
    merged: dict[int, dict] = {}
    for res in text_results:
        entry = merged.setdefault(
            res["id"],
            {
                "id": res["id"],
                "doc_id": res["doc_id"],
                "chunk_index": res["chunk_index"],
                "content": res["content"],
            },
        )
        entry["rank_text"] = res["rank_text"]
        entry["text_score"] = res["text_score"]
        entry["rrf_score"] = entry.get("rrf_score", 0.0) + weight_text / (
            rrf_k + res["rank_text"]
        )
    for res in vector_results:
        entry = merged.setdefault(
            res["id"],
            {
                "id": res["id"],
                "doc_id": res["doc_id"],
                "chunk_index": res["chunk_index"],
                "content": res["content"],
            },
        )
        entry["rank_vector"] = res["rank_vector"]
        entry["vector_distance"] = res["vector_distance"]
        entry["rrf_score"] = entry.get("rrf_score", 0.0) + weight_vector / (
            rrf_k + res["rank_vector"]
        )
    merged_list = list(merged.values())
    for item in merged_list:
        item["rrf_score"] = float(item.get("rrf_score", 0.0))
    ranked = sorted(merged_list, key=lambda item: item["rrf_score"], reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank_rrf"] = idx
    return ranked


def rerank_with_cohere(query: str, results: list[dict]) -> list[dict]:
    """Rerank results with Cohere."""
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY missing")
    if not results:
        return results
    client = cohere.Client(api_key)
    scored = [dict(row) for row in results]
    docs = [row["content"] for row in scored]
    rerank = client.rerank(
        model=COHERE_MODEL,
        query=query,
        documents=docs,
        top_n=len(docs),
    )
    for item in rerank.results:
        scored[item.index]["rerank_score"] = float(item.relevance_score)
    ranked = sorted(scored, key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank_rerank"] = idx
    return ranked


def run_search(
    doc_id: int,
    query: str,
    top_k: int,
    rrf_k: int,
    weight_text: float,
    weight_vector: float,
    index_mode: str,
    ef_search: int,
    probes: int,
    force_index: bool,
    use_rerank: bool,
) -> dict:
    """Run text, vector, and merge."""
    timings: dict[str, float] = {}
    LOGGER.info(
        "search start doc_id=%s top_k=%d index=%s rerank=%s",
        doc_id,
        top_k,
        index_mode,
        use_rerank,
    )
    start = time.perf_counter()
    text_results = search_text(doc_id, query, top_k)
    timings["text_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    query_vec = embed_texts([query])[0]
    timings["embed_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    vector_results = search_vector(
        doc_id,
        query_vec,
        top_k,
        index_mode,
        ef_search,
        probes,
        force_index,
    )
    timings["vector_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    merged = merge_rrf(text_results, vector_results, rrf_k, weight_text, weight_vector)
    timings["rrf_ms"] = (time.perf_counter() - start) * 1000

    reranked = merged
    if use_rerank:
        start = time.perf_counter()
        reranked = rerank_with_cohere(query, merged)
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000
    LOGGER.info(
        "search done doc_id=%s text=%d vector=%d merged=%d",
        doc_id,
        len(text_results),
        len(vector_results),
        len(merged),
    )

    return {
        "text_results": text_results,
        "vector_results": vector_results,
        "merged_results": merged,
        "reranked_results": reranked,
        "timings": timings,
        "query_vec": query_vec,
        "used_rerank": use_rerank,
    }


def explain_text(
    doc_id: int,
    query: str,
    limit: int,
    analyze: bool,
    force_index: bool,
) -> list[str]:
    """Explain BM25 query plan."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    options = "ANALYZE, BUFFERS, FORMAT TEXT" if analyze else "FORMAT TEXT"
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            if force_index:
                cur.execute("SET LOCAL enable_seqscan = off;")
            cur.execute(
                f"""
                EXPLAIN ({options})
                SELECT id
                FROM chunks
                WHERE doc_id = %s
                ORDER BY content <@> to_bm25query(%s, 'chunks_bm25_idx')
                LIMIT %s;
                """,
                (doc_id, query, limit),
            )
            return [row[0] for row in cur.fetchall()]


def explain_vector(
    doc_id: int,
    query_vec: list[float],
    limit: int,
    index_mode: str,
    ef_search: int,
    probes: int,
    force_index: bool,
    analyze: bool,
) -> list[str]:
    """Explain vector query plan."""
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    options = "ANALYZE, BUFFERS, FORMAT TEXT" if analyze else "FORMAT TEXT"
    vec = vector_literal(query_vec)
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            apply_vector_settings(cur, index_mode, ef_search, probes, force_index)
            cur.execute(
                f"""
                EXPLAIN ({options})
                SELECT id
                FROM chunks
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (doc_id, vec, limit),
            )
            return [row[0] for row in cur.fetchall()]


def render_rank_list(
    results: list[dict],
    rank_key: str,
    score_key: str,
    score_label: str,
    limit: int = 5,
) -> None:
    """Render a compact rank list."""
    if not results:
        st.write("no results")
        return
    for res in results[:limit]:
        score = res.get(score_key)
        score_text = f"{score_label} {score:.4f}" if score is not None else score_label
        st.write(
            f"{res.get(rank_key, '-')} · id {res['id']} · {score_text} · "
            f"{format_snippet(res['content'])}"
        )


def render_score_panes(
    text_results: list[dict],
    vector_results: list[dict],
    merged_results: list[dict],
    use_rerank: bool,
) -> None:
    """Render rank comparison panes."""
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Text rank**")
        render_rank_list(text_results, "rank_text", "bm25_score", "bm25")
    with cols[1]:
        st.markdown("**Vector rank**")
        render_rank_list(vector_results, "rank_vector", "vector_distance", "dist")
    with cols[2]:
        label = "Rerank rank" if use_rerank else "RRF rank"
        st.markdown(f"**{label}**")
        if use_rerank:
            render_rank_list(merged_results, "rank_rerank", "rerank_score", "rerank")
        else:
            render_rank_list(merged_results, "rank_rrf", "rrf_score", "rrf")


def compute_overlap_stats(
    text_results: list[dict],
    vector_results: list[dict],
    merged_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Compute overlap and rank stats."""
    text_ids = [res["id"] for res in text_results[:top_k]]
    vector_ids = [res["id"] for res in vector_results[:top_k]]
    merged_ids = [res["id"] for res in merged_results[:top_k]]
    merged_rank = {res["id"]: idx + 1 for idx, res in enumerate(merged_results)}

    def overlap_ratio(ids: list[int]) -> float:
        if not ids:
            return 0.0
        return len(set(ids) & set(merged_ids)) / len(ids)

    def avg_rank(ids: list[int]) -> float:
        ranks = [merged_rank[item] for item in ids if item in merged_rank]
        if not ranks:
            return 0.0
        return sum(ranks) / len(ranks)

    return [
        {
            "method": "text",
            "overlap@k": overlap_ratio(text_ids),
            "avg_merged_rank": avg_rank(text_ids),
        },
        {
            "method": "vector",
            "overlap@k": overlap_ratio(vector_ids),
            "avg_merged_rank": avg_rank(vector_ids),
        },
        {
            "method": "merged",
            "overlap@k": overlap_ratio(merged_ids),
            "avg_merged_rank": avg_rank(merged_ids),
        },
    ]


def render_results(title: str, results: list[dict], show_scores: bool) -> None:
    """Render result list with details."""
    st.markdown(f"**{title}**")
    if not results:
        st.write("no results")
        return
    for res in results:
        header = f"chunk {res['chunk_index']} (id {res['id']})"
        with st.expander(header, expanded=False):
            st.write(res["content"])
            if show_scores:
                st.json(
                    {
                        "rank_text": res.get("rank_text"),
                        "bm25_score": res.get("bm25_score"),
                        "text_score": res.get("text_score"),
                        "rank_vector": res.get("rank_vector"),
                        "vector_distance": res.get("vector_distance"),
                        "rank_rrf": res.get("rank_rrf"),
                        "rrf_score": res.get("rrf_score"),
                        "rank_rerank": res.get("rank_rerank"),
                        "rerank_score": res.get("rerank_score"),
                    }
                )


def main():
    """Render Streamlit UI."""
    st.set_page_config(page_title="PG Hybrid Search Demo", layout="centered")
    st.title("PG Hybrid Search Demo")
    st.caption("Hybrid search in Postgres. Single doc flow.")
    st.caption("Expecting pg_vector (vector) + pg_textsearch extensions.")

    st.subheader("Upload")
    uploaded = st.file_uploader(
        "Upload pdf, docx, or markdown",
        type=["pdf", "docx", "md", "markdown"],
    )
    if uploaded:
        file_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        st.caption(f"sha256: {file_hash}")
        ingest_state = st.session_state.setdefault("ingest_state", {})
        state = ingest_state.get(file_hash)
        if state and state.get("status") == "exists":
            st.info(f"already ingested: {state.get('filename', 'file')}")
        elif state and state.get("status") == "done":
            st.success(
                f"ingested {state.get('filename', 'file')} "
                f"({state.get('chunk_count', 0)} chunks)"
            )
        elif state and state.get("status") == "error":
            st.error(f"ingest failed: {state.get('error')}")
        else:
            progress_holder = st.empty()
            status_holder = st.empty()
            progress_bar = progress_holder.progress(0.0)

            def progress_cb(ratio: float, label: str) -> None:
                progress_bar.progress(max(0.0, min(1.0, ratio)))
                if label:
                    status_holder.caption(label)

            try:
                ingest_state[file_hash] = {"status": "running"}
                with st.spinner("Ingesting..."):
                    result = ingest_file(uploaded, progress_cb)
                if result["status"] == "exists":
                    ingest_state[file_hash] = {
                        "status": "exists",
                        "filename": result["filename"],
                    }
                    st.info(f"already ingested: {result['filename']}")
                else:
                    ingest_state[file_hash] = {
                        "status": "done",
                        "filename": result["filename"],
                        "chunk_count": result["chunk_count"],
                    }
                    st.success(
                        f"ingested {result['filename']} ({result['chunk_count']} chunks)"
                    )
            except Exception as exc:
                LOGGER.exception("ingest failed")
                ingest_state[file_hash] = {"status": "error", "error": str(exc)}
                st.error(f"ingest failed: {exc}")

    st.subheader("Query")
    docs = list_documents()
    if not docs:
        st.info("no documents yet")
    else:
        doc_map = {
            f"{d['filename']} (id {d['id']}, {d['chunk_count']} chunks)": d
            for d in docs
        }
        doc_label = st.selectbox("Document", options=list(doc_map.keys()))
        doc = doc_map[doc_label]
        st.caption(f"doc id: {doc['id']} | type: {doc['file_type']}")

        def trigger_search() -> None:
            st.session_state["do_search"] = True

        st.markdown("Query")
        query_col, button_col = st.columns([4, 1])
        with query_col:
            st.text_input(
                "Query",
                key="query_input",
                on_change=trigger_search,
                label_visibility="collapsed",
            )
        with button_col:
            if st.button("Search"):
                st.session_state["do_search"] = True

        use_rerank = st.checkbox("Use Cohere rerank", value=False)

        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top K", 5, 50, 20)
            weight_text = st.slider("Text weight", 0.0, 2.0, 1.0, 0.1)
        with col2:
            rrf_k = st.slider("RRF K", 1, 100, 60)
            weight_vector = st.slider("Vector weight", 0.0, 2.0, 1.0, 0.1)

        with st.expander("Index settings", expanded=False):
            index_mode = st.selectbox("Vector index", options=["hnsw", "ivfflat"])
            ef_search = st.slider("HNSW ef_search", 10, 400, 200)
            probes = st.slider("IVFFlat probes", 1, 200, 20)
            force_index = st.checkbox(
                "Force index scans",
                value=False,
                key="force_index_search",
            )
            st.caption("Only the selected index mode setting is applied.")

        if st.session_state.pop("do_search", False):
            query = st.session_state.get("query_input", "").strip()
            if not query:
                st.error("query required")
            else:
                try:
                    with st.spinner("Searching..."):
                        results = run_search(
                            doc["id"],
                            query,
                            top_k,
                            rrf_k,
                            weight_text,
                            weight_vector,
                            index_mode,
                            ef_search,
                            probes,
                            force_index,
                            use_rerank,
                        )
                    st.session_state["search_results"] = results
                    st.session_state["search_query"] = query
                    st.session_state["search_doc_id"] = doc["id"]
                    st.session_state["search_params"] = {
                        "top_k": top_k,
                        "index_mode": index_mode,
                        "ef_search": ef_search,
                        "probes": probes,
                        "force_index": force_index,
                    }
                except Exception as exc:
                    LOGGER.exception("search failed")
                    st.error(f"search failed: {exc}")

        results = st.session_state.get("search_results")
        if results and st.session_state.get("search_doc_id") == doc["id"]:
            params = st.session_state.get("search_params", {})
            used_rerank = results.get("used_rerank", False)
            search_query = st.session_state.get("search_query", "")
            timings = results["timings"]
            cols = st.columns(5)
            cols[0].metric("embed ms", f"{timings.get('embed_ms', 0):.1f}")
            cols[1].metric("text ms", f"{timings.get('text_ms', 0):.1f}")
            cols[2].metric("vector ms", f"{timings.get('vector_ms', 0):.1f}")
            cols[3].metric("rrf ms", f"{timings.get('rrf_ms', 0):.1f}")
            cols[4].metric("rerank ms", f"{timings.get('rerank_ms', 0):.1f}")

            merged = results["reranked_results"] if used_rerank else results["merged_results"]
            render_score_panes(
                results["text_results"],
                results["vector_results"],
                merged,
                used_rerank,
            )

            st.caption("Overlap@k is proxy recall vs merged results.")
            overlap = compute_overlap_stats(
                results["text_results"],
                results["vector_results"],
                merged,
                params.get("top_k", top_k),
            )
            st.table(overlap)

            tabs = st.tabs(["RRF merged", "Text fragments", "Embedding hits"])
            with tabs[0]:
                title = "Merged (reranked)" if used_rerank else "Merged (RRF)"
                render_results(title, merged, show_scores=True)
            with tabs[1]:
                render_results("Text BM25", results["text_results"], show_scores=True)
            with tabs[2]:
                render_results("Vector hits", results["vector_results"], show_scores=True)

            with st.expander("Explain plan", expanded=False):
                explain_analyze = st.checkbox(
                    "Analyze",
                    value=False,
                    key="explain_analyze",
                )
                explain_force = st.checkbox(
                    "Force index scans",
                    value=True,
                    key="force_index_explain",
                )
                if st.button("Run EXPLAIN"):
                    try:
                        text_plan = explain_text(
                            doc["id"],
                            search_query,
                            params.get("top_k", top_k),
                            explain_analyze,
                            explain_force,
                        )
                        vector_plan = explain_vector(
                            doc["id"],
                            results["query_vec"],
                            params.get("top_k", top_k),
                            params.get("index_mode", index_mode),
                            params.get("ef_search", ef_search),
                            params.get("probes", probes),
                            explain_force,
                            explain_analyze,
                        )
                        st.markdown("**Text plan**")
                        st.code("\n".join(text_plan))
                        st.markdown("**Vector plan**")
                        st.code("\n".join(vector_plan))
                    except Exception as exc:
                        LOGGER.exception("explain failed")
                        st.error(f"explain failed: {exc}")

    st.subheader("DB status")
    st.button("Recheck")
    msg, ok = check_db()
    if ok:
        st.success(msg)
    else:
        st.error(msg)


if __name__ == "__main__":
    main()
