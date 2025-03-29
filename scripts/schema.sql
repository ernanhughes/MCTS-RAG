-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tables
CREATE TABLE IF NOT EXISTS retrieval_memory (
    id SERIAL PRIMARY KEY,
    query TEXT UNIQUE,
    embedding VECTOR(1024),
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reasoning_traces (
    id SERIAL PRIMARY KEY,
    question TEXT,
    embedding VECTOR(1024),
    trace JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_trace_log (
    id SERIAL PRIMARY KEY,
    question TEXT,
    generated_query TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Create cosine similarity index for fast search
CREATE INDEX IF NOT EXISTS idx_retrieval_memory_embedding
ON retrieval_memory USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reasoning_traces_embedding
ON reasoning_traces USING ivfflat (embedding vector_cosine_ops);
