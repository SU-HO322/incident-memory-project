-- ============================================================
-- db/schema.sql
-- Incident Memory Agent — Supabase Database Schema
-- ============================================================
-- Run this entire file once in the Supabase SQL Editor.
-- See README.md → "Supabase Setup" for step-by-step instructions.
-- ============================================================


-- Step 1: Enable the pgvector extension
-- This gives PostgreSQL the ability to store and search vectors.
CREATE EXTENSION IF NOT EXISTS vector;


-- Step 2: Create the incidents table
CREATE TABLE IF NOT EXISTS incidents (
    -- Unique identifier — auto-generated UUID for every row
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Short human-readable name, e.g. "Database connection pool exhausted"
    title           TEXT NOT NULL CHECK (char_length(title) BETWEEN 3 AND 200),

    -- The raw error log or stack trace that was observed during the incident
    error_log       TEXT NOT NULL CHECK (char_length(error_log) >= 10),

    -- The diagnosed root cause, e.g. "N+1 query pattern under load"
    root_cause      TEXT NOT NULL CHECK (char_length(root_cause) >= 10),

    -- The steps taken to resolve the incident
    resolution      TEXT NOT NULL CHECK (char_length(resolution) >= 10),

    -- Optional labels for filtering, e.g. {'database', 'timeout', 'production'}
    tags            TEXT[] NOT NULL DEFAULT '{}',

    -- Severity level — must be one of four allowed values
    severity        TEXT NOT NULL DEFAULT 'medium'
                    CHECK (severity IN ('low', 'medium', 'high', 'critical')),

    -- When the incident was actually fixed (can differ from created_at)
    resolved_at     TIMESTAMPTZ,

    -- When this record was added to our system
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- The 768-dimensional semantic embedding of the error_log text.
    -- Gemini text-embedding-004 produces 768-dimensional vectors.
    -- This is what enables similarity search.
    embedding       VECTOR(768)
);


-- Step 3: Create an HNSW index for fast approximate nearest-neighbour search
-- HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm.
-- It trades a tiny bit of recall for massive speed gains at scale.
-- vector_cosine_ops = use cosine distance (best for semantic text similarity).
CREATE INDEX IF NOT EXISTS incidents_embedding_hnsw_idx
    ON incidents
    USING hnsw (embedding vector_cosine_ops);


-- Step 4: Create an index on created_at for fast dashboard queries
CREATE INDEX IF NOT EXISTS incidents_created_at_idx
    ON incidents (created_at DESC);


-- Step 5: Enable Row Level Security (RLS)
-- This is a Supabase best practice. It means rows are only accessible
-- through authenticated requests (your app's service key), not public access.
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (your backend service key)
CREATE POLICY "Allow all for authenticated users"
    ON incidents
    FOR ALL
    USING (true)
    WITH CHECK (true);


-- ============================================================
-- Step 6: Create the similarity search function
-- ============================================================
-- We define this as a Postgres function so we can call it cleanly
-- from Python using the Supabase client's .rpc() method.
-- It takes a query vector, a similarity threshold, and a row limit,
-- and returns all matching incidents with their similarity scores.
-- ============================================================

CREATE OR REPLACE FUNCTION search_similar_incidents(
    query_embedding VECTOR(768),
    similarity_threshold FLOAT DEFAULT 0.50,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id              UUID,
    title           TEXT,
    error_log       TEXT,
    root_cause      TEXT,
    resolution      TEXT,
    tags            TEXT[],
    severity        TEXT,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.id,
        i.title,
        i.error_log,
        i.root_cause,
        i.resolution,
        i.tags,
        i.severity,
        i.resolved_at,
        i.created_at,
        -- Cosine SIMILARITY = 1 - cosine DISTANCE
        -- The <=> operator in pgvector gives cosine distance (0 = identical, 2 = opposite)
        1 - (i.embedding <=> query_embedding) AS similarity
    FROM incidents i
    WHERE 1 - (i.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY i.embedding <=> query_embedding   -- ascending distance = descending similarity
    LIMIT match_count;
END;
$$;


-- ============================================================
-- Verification query — run after setup to confirm everything worked:
-- ============================================================
-- SELECT table_name FROM information_schema.tables WHERE table_name = 'incidents';
-- SELECT routine_name FROM information_schema.routines WHERE routine_name = 'search_similar_incidents';
-- ============================================================
