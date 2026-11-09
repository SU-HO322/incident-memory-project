"""
services/incident_service.py
-----------------------------
The single orchestration layer between the UI and all backend services.

This is the ONLY module that Streamlit pages should ever import.
The UI never touches embedder, vector_store, or ai_advisor directly.

PUBLIC SURFACE:
  store_incident(IncidentInput)  → Incident
  search_incidents(str)          → SearchResult
  list_incidents(int)            → list[Incident]
  get_stats()                    → dict

DATA FLOW:

  store_incident(input):
    1. embedder.embed(input.error_log)      → 768-dim vector
    2. vector_store.store_incident(input, vector)  → Incident (persisted)

  search_incidents(query):
    1. embedder.embed(query)                → 768-dim vector
    2. vector_store.search_similar_incidents(vector)  → list[Incident]
    3. ai_advisor.generate_recommendation(query, matches)  → SearchResult
"""

from __future__ import annotations

import logging

from models.incident import Incident, IncidentInput, SearchResult
from services import ai_advisor, embedder, vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Write path — store a new resolved incident
# ---------------------------------------------------------------------------

def store_incident(incident_data: IncidentInput) -> Incident:
    """
    Persist a new resolved incident with its semantic embedding.

    Steps:
      1. Validate input (Pydantic handles this at IncidentInput creation time)
      2. Embed the error_log text → 768-dim vector
      3. Write incident + vector to Supabase via psycopg3

    Args:
        incident_data: Validated incident fields (title, error_log,
                       root_cause, resolution, tags, severity, resolved_at).

    Returns:
        The persisted Incident with its auto-generated id and created_at.

    Raises:
        ValueError:   If embedding input is empty.
        RuntimeError: If the Gemini API or database write fails.
    """
    logger.info("Storing new incident: '%s'", incident_data.title)

    # Step 1 — Generate embedding from the error log text.
    # We embed error_log (not title) because that's what users search with.
    try:
        vector = embedder.embed(incident_data.error_log)
    except (ValueError, RuntimeError) as exc:
        logger.error("Embedding failed for '%s': %s", incident_data.title, exc)
        raise

    # Step 2 — Persist to Supabase with the embedding vector.
    try:
        incident = vector_store.store_incident(incident_data, vector)
    except Exception as exc:
        logger.error("DB write failed for '%s': %s", incident_data.title, exc)
        raise

    logger.info("Incident stored successfully: id=%s", incident.id)
    return incident


# ---------------------------------------------------------------------------
# Read path — search and advise
# ---------------------------------------------------------------------------

def search_incidents(query: str) -> SearchResult:
    """
    Search historical incidents semantically and return an AI recommendation.

    Steps:
      1. Embed the query text → 768-dim vector
      2. Search the vector store for similar incidents (cosine similarity)
      3. Send query + matches to Gemini 2.5 Flash for a grounded recommendation

    Args:
        query: Raw error log or stack trace from the user. Can be long.

    Returns:
        SearchResult containing:
          - matches           : Top-K similar historical incidents
          - ai_recommendation : Grounded AI response (or graceful fallback)
          - confidence_level  : high / medium / low / no_match
          - top_similarity_score: float 0–1 (or None if no matches)
          - has_matches       : bool convenience flag

    Note:
        If no incidents are stored yet, returns a SearchResult with
        empty matches and a "no history" AI message — never crashes.
    """
    query = query.strip()
    if not query:
        logger.warning("search_incidents called with empty query.")
        from models.incident import ConfidenceLevel
        return SearchResult(
            query=query,
            matches=[],
            ai_recommendation="Please enter an error log or stack trace to search.",
            confidence_level=ConfidenceLevel.NO_MATCH,
            has_matches=False,
        )

    logger.info("Searching incidents for query (%d chars)...", len(query))

    # Step 1 — Embed the query. If this fails, surface the error clearly.
    try:
        query_vector = embedder.embed(query)
    except (ValueError, RuntimeError) as exc:
        logger.error("Query embedding failed: %s", exc)
        from models.incident import ConfidenceLevel
        return SearchResult(
            query=query,
            matches=[],
            ai_recommendation=(
                f"⚠️ Could not process your query: {exc}\n"
                "Check your GEMINI_API_KEY and network connection."
            ),
            confidence_level=ConfidenceLevel.NO_MATCH,
            has_matches=False,
        )

    # Step 2 — Vector similarity search. Returns [] gracefully if DB is empty.
    try:
        matches = vector_store.search_similar_incidents(query_vector)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        from models.incident import ConfidenceLevel
        return SearchResult(
            query=query,
            matches=[],
            ai_recommendation=(
                f"⚠️ Database search failed: {exc}\n"
                "Check your DATABASE_URL and Supabase connection."
            ),
            confidence_level=ConfidenceLevel.NO_MATCH,
            has_matches=False,
        )

    logger.info("Vector search returned %d match(es).", len(matches))

    # Step 3 — Generate AI recommendation. Even if matches is empty,
    # ai_advisor handles it gracefully with a "no match" response.
    result = ai_advisor.generate_recommendation(query, matches)

    return result


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def list_incidents(limit: int = 100) -> list[Incident]:
    """
    Return all stored incidents (newest first) for the history dashboard.

    Args:
        limit: Maximum number of incidents to return. Default 100.

    Returns:
        List of Incidents ordered by created_at DESC.
        Returns [] if the database is empty or unreachable.
    """
    try:
        return vector_store.list_all_incidents(limit=limit)
    except Exception as exc:
        logger.error("list_incidents failed: %s", exc)
        return []


def get_stats() -> dict:
    """
    Return summary statistics for the dashboard and home page stat cards.
    """
    try:
        stats = vector_store.get_incident_statistics()
        if isinstance(stats, dict) and "total_incidents" in stats and isinstance(stats["total_incidents"], int):
            return stats
        count = vector_store.get_incident_count()
        return {"total_incidents": count, "db_connected": True}
    except Exception as exc:
        logger.error("get_stats failed: %s", exc)
        return {
            "total_incidents": 0,
            "resolved_count": 0,
            "unique_services": 0,
            "avg_resolution_hours": "—",
            "db_connected": False,
        }
