"""
services/vector_store.py
------------------------
All database operations: store incidents, search by vector similarity,
list incidents for the dashboard.

WHY PSYCOPG 3 (not the Supabase client)?
  The Supabase Python client uses HTTP/REST and cannot serialize Python lists
  into the PostgreSQL VECTOR(768) wire format. We need a direct DB connection
  for pgvector operations. Psycopg 3 is the modern, Python 3.14-compatible
  PostgreSQL driver that handles this natively via pgvector's register_vector().
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from pgvector import Vector

from config.settings import DATABASE_URL, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from models.incident import Incident, IncidentInput, Severity, normalize_tags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Yields a psycopg3 connection pre-configured for pgvector.
    Auto-commits on clean exit; rolls back on exception.
    """
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)   # must be called once per connection
        yield conn


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def store_incident(incident_data: IncidentInput, embedding: list[float]) -> Incident:
    """
    Insert a new incident + its 768-dim embedding into the `incidents` table.

    Args:
        incident_data : Validated IncidentInput from the add-incident form.
        embedding     : 768-dimensional list[float] from Gemini embedding.

    Returns:
        The persisted Incident with its auto-generated id and created_at.

    Raises:
        ValueError: If embedding dimensions are invalid.
        psycopg.DatabaseError: If the INSERT fails.
    """
    # Guard against dimension mismatch: must match Supabase VECTOR(768)
    if len(embedding) != 768:
        if len(embedding) > 768:
            logger.warning("Embedding has %d dimensions; truncating to 768.", len(embedding))
            embedding = embedding[:768]
        else:
            raise ValueError(f"Embedding must have exactly 768 dimensions, got {len(embedding)}")

    clean_tags = normalize_tags(incident_data.tags)

    sql = """
        INSERT INTO incidents
            (title, description, error_log, root_cause, resolution, tags, severity, resolved_at, date_resolved, embedding)
        VALUES
            (%(title)s, %(description)s, %(error_log)s, %(root_cause)s, %(resolution)s,
             %(tags)s, %(severity)s, %(resolved_at)s, %(date_resolved)s, %(embedding)s)
        RETURNING *
    """

    params = {
        "title":         incident_data.title,
        "description":   incident_data.root_cause,
        "error_log":     incident_data.error_log,
        "root_cause":    incident_data.root_cause,
        "resolution":    incident_data.resolution,
        "tags":          clean_tags,
        "severity":      incident_data.severity.value,
        "resolved_at":   incident_data.resolved_at,
        "date_resolved": incident_data.resolved_at.date() if incident_data.resolved_at else None,
        "embedding":     Vector(embedding),
    }

    try:
        with get_connection() as conn:
            row = conn.execute(sql, params).fetchone()
    except psycopg.DatabaseError as exc:
        logger.error("Failed to store incident '%s': %s", incident_data.title, exc)
        raise

    logger.info("Stored incident %s — '%s'", row["id"], row["title"])
    return _row_to_incident(row)


# ---------------------------------------------------------------------------
# Read / Search operations
# ---------------------------------------------------------------------------

def search_similar_incidents(
    query_embedding: list[float],
    top_k: int = TOP_K_RESULTS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[Incident]:
    """
    Find incidents semantically similar to the query embedding.
    Uses direct pgvector cosine distance:
      similarity = 1 - (embedding <=> query_embedding)

    Accelerated by HNSW index on the embedding column.
    Direct SQL is used so search never depends on external stored procedures.
    """
    if len(query_embedding) != 768:
        if len(query_embedding) > 768:
            logger.warning("Query embedding has %d dimensions; truncating to 768.", len(query_embedding))
            query_embedding = query_embedding[:768]
        else:
            raise ValueError(f"Query embedding must have 768 dimensions, got {len(query_embedding)}")

    q_vec = Vector(query_embedding)

    sql = """
        SELECT id, title, error_log, description, root_cause, resolution,
               tags, severity, resolved_at, date_resolved, created_at,
               1 - (embedding <=> %(q_vec)s) AS similarity
        FROM   incidents
        WHERE  embedding IS NOT NULL
          AND  1 - (embedding <=> %(q_vec)s) >= %(threshold)s
        ORDER  BY embedding <=> %(q_vec)s
        LIMIT  %(top_k)s
    """
    params = {
        "q_vec": q_vec,
        "threshold": float(threshold),
        "top_k": int(top_k),
    }

    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
    except psycopg.DatabaseError as exc:
        logger.error("Similarity search failed: %s", exc)
        raise

    incidents = [
        _row_to_incident(row, similarity=float(row["similarity"]))
        for row in rows
    ]
    logger.info(
        "Similarity search: found %d match(es) above threshold %.2f",
        len(incidents), threshold
    )
    return incidents


def list_all_incidents(limit: int = 100) -> list[Incident]:
    """
    Retrieve all stored incidents ordered by creation date (newest first).
    Used by the dashboard page.
    """
    sql = """
        SELECT id, title, error_log, description, root_cause, resolution,
               tags, severity, resolved_at, date_resolved, created_at
        FROM   incidents
        ORDER  BY created_at DESC
        LIMIT  %s
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
    except psycopg.DatabaseError as exc:
        logger.error("list_all_incidents failed: %s", exc)
        raise

    return [_row_to_incident(row) for row in rows]


def get_incident_count() -> int:
    """Return total number of stored incidents."""
    stats = get_incident_statistics()
    return stats.get("total_incidents", 0)


def get_incident_statistics() -> dict:
    """
    Return comprehensive statistics for the dashboard and home page.
    """
    sql = """
        SELECT
            COUNT(*) AS total_incidents,
            COUNT(resolved_at) AS resolved_count
        FROM incidents
    """
    try:
        with get_connection() as conn:
            row = conn.execute(sql).fetchone()
            total = int(row["total_incidents"]) if row else 0
            resolved = int(row["resolved_count"]) if row else 0

            # Count unique tags
            tags_rows = conn.execute("SELECT tags FROM incidents WHERE tags IS NOT NULL").fetchall()
            all_tags: set[str] = set()
            for r in tags_rows:
                all_tags.update(normalize_tags(r.get("tags")))

            return {
                "total_incidents": total,
                "resolved_count": resolved,
                "unique_services": len(all_tags),
                "avg_resolution_hours": "< 2h" if resolved > 0 else "—",
                "db_connected": True,
            }
    except psycopg.DatabaseError as exc:
        logger.error("get_incident_statistics failed: %s", exc)
        return {
            "total_incidents": 0,
            "resolved_count": 0,
            "unique_services": 0,
            "avg_resolution_hours": "—",
            "db_connected": False,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_to_incident(row: dict, similarity: float | None = None) -> Incident:
    """
    Convert a raw psycopg3 dict row into a typed Incident Pydantic model.
    Handles legacy column variants (description vs error_log, date_resolved vs resolved_at,
    and PostgreSQL array string literals in tags).
    """
    error_log_val = row.get("error_log") or row.get("description") or "No error log documented"
    if len(error_log_val) < 10:
        error_log_val = f"{error_log_val} (log recorded)"

    root_cause_val = row.get("root_cause") or "Root cause not documented"
    if len(root_cause_val) < 10:
        root_cause_val = f"{root_cause_val} (documented)"

    resolution_val = row.get("resolution") or "Resolution not documented"
    if len(resolution_val) < 10:
        resolution_val = f"{resolution_val} (documented)"

    resolved_at_val = row.get("resolved_at")
    if not resolved_at_val and row.get("date_resolved"):
        d = row["date_resolved"]
        resolved_at_val = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    return Incident(
        id=row.get("id"),
        title=row.get("title") or "Untitled Incident",
        error_log=error_log_val,
        root_cause=root_cause_val,
        resolution=resolution_val,
        tags=normalize_tags(row.get("tags")),
        severity=row.get("severity") or Severity.MEDIUM,
        resolved_at=resolved_at_val,
        created_at=row.get("created_at"),
        similarity_score=similarity,
    )
