CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    doc_hash TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    content_tokens INT,
    embedding VECTOR(1536) NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id, chunk_index);

CREATE INDEX IF NOT EXISTS chunks_bm25_idx
    ON chunks USING bm25 (content)
    WITH (text_config = 'english', k1 = 1.2, b = 0.75);

CREATE INDEX IF NOT EXISTS chunks_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_ivfflat_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

DROP VIEW IF EXISTS v_chunk_counts;
CREATE VIEW v_chunk_counts AS
SELECT d.id AS doc_id,
       d.filename,
       d.doc_hash,
       count(c.id) AS chunk_count,
       max(c.created_at) AS last_chunk_at
FROM documents d
LEFT JOIN chunks c ON c.doc_id = d.id
GROUP BY d.id, d.filename, d.doc_hash;

DROP VIEW IF EXISTS v_index_usage;
CREATE VIEW v_index_usage AS
SELECT schemaname,
       relname AS table_name,
       indexrelname AS index_name,
       idx_scan,
       idx_tup_read,
       idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname IN ('documents', 'chunks');
