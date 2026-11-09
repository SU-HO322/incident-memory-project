"""
tests/test_incident_service.py
-------------------------------
Unit tests for services/incident_service.py.

STRATEGY: Mock all three dependencies (embedder, vector_store, ai_advisor).
  - Tests verify the ORCHESTRATION logic:
      - Are the right functions called in the right order?
      - Does a failure in one layer produce a graceful result (not a crash)?
      - Are arguments passed correctly between layers?

Run with:  pytest tests/test_incident_service.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from models.incident import ConfidenceLevel, Incident, IncidentInput, SearchResult, Severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_input(**kwargs) -> IncidentInput:
    defaults = dict(
        title="Redis OOM crash",
        error_log="FATAL: Redis container killed by OOM killer. Used 2048MB / limit 2048MB",
        root_cause="Memory leak in session cache — unbounded growth over 48h",
        resolution="1. Restart Redis. 2. Set maxmemory-policy allkeys-lru. 3. Add eviction monitoring.",
        tags=["redis", "memory", "oom"],
        severity=Severity.CRITICAL,
    )
    defaults.update(kwargs)
    return IncidentInput(**defaults)


def _make_incident(similarity: float = 0.85) -> Incident:
    return Incident(
        id=uuid4(),
        title="Redis OOM crash",
        error_log="FATAL: Redis OOM",
        root_cause="Memory leak",
        resolution="Set maxmemory-policy",
        tags=["redis"],
        severity=Severity.HIGH,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        similarity_score=similarity,
    )


def _make_search_result(matches=None) -> SearchResult:
    matches = matches or [_make_incident()]
    return SearchResult(
        query="redis oom",
        matches=matches,
        ai_recommendation="Likely root cause: memory leak. Resolution: set eviction policy.",
        confidence_level=ConfidenceLevel.HIGH,
        top_similarity_score=matches[0].similarity_score if matches else None,
        has_matches=bool(matches),
    )


# ---------------------------------------------------------------------------
# store_incident tests
# ---------------------------------------------------------------------------

class TestStoreIncident:

    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_calls_embed_then_store(self, mock_embedder, mock_vector_store):
        """store_incident embeds error_log first, then writes to DB."""
        from services.incident_service import store_incident

        fake_vector = [0.1] * 768
        mock_embedder.embed.return_value = fake_vector
        stored = _make_incident()
        mock_vector_store.store_incident.return_value = stored

        input_data = _make_input()
        result = store_incident(input_data)

        # embed() was called with the error_log text
        mock_embedder.embed.assert_called_once_with(input_data.error_log)
        # vector_store.store_incident was called with input + the vector
        mock_vector_store.store_incident.assert_called_once_with(input_data, fake_vector)
        assert result == stored

    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_embedding_failure_propagates(self, mock_embedder, mock_vector_store):
        """If embedder raises, store_incident re-raises without touching the DB."""
        from services.incident_service import store_incident

        mock_embedder.embed.side_effect = RuntimeError("API quota exceeded")

        with pytest.raises(RuntimeError, match="API quota exceeded"):
            store_incident(_make_input())

        # DB must NOT be called if embedding failed
        mock_vector_store.store_incident.assert_not_called()

    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_db_failure_propagates(self, mock_embedder, mock_vector_store):
        """If vector_store raises, store_incident re-raises."""
        from services.incident_service import store_incident

        mock_embedder.embed.return_value = [0.1] * 768
        mock_vector_store.store_incident.side_effect = Exception("DB connection refused")

        with pytest.raises(Exception, match="DB connection refused"):
            store_incident(_make_input())

    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_embeds_error_log_not_title(self, mock_embedder, mock_vector_store):
        """The error_log (not title) is embedded — that's what users search with."""
        from services.incident_service import store_incident

        mock_embedder.embed.return_value = [0.1] * 768
        mock_vector_store.store_incident.return_value = _make_incident()

        input_data = _make_input(title="A short title", error_log="Long detailed error log...")
        store_incident(input_data)

        # Must embed error_log, not title
        embedded_text = mock_embedder.embed.call_args[0][0]
        assert embedded_text == "Long detailed error log..."
        assert embedded_text != "A short title"


# ---------------------------------------------------------------------------
# search_incidents tests
# ---------------------------------------------------------------------------

class TestSearchIncidents:

    @patch("services.incident_service.ai_advisor")
    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_full_pipeline_called_in_order(self, mock_embedder, mock_vector_store, mock_ai_advisor):
        """search_incidents calls embed → search → recommend in correct order."""
        from services.incident_service import search_incidents

        fake_vector = [0.5] * 768
        fake_matches = [_make_incident(similarity=0.88)]
        fake_result = _make_search_result(fake_matches)

        mock_embedder.embed.return_value = fake_vector
        mock_vector_store.search_similar_incidents.return_value = fake_matches
        mock_ai_advisor.generate_recommendation.return_value = fake_result

        result = search_incidents("Redis OOM killer triggered")

        mock_embedder.embed.assert_called_once_with("Redis OOM killer triggered")
        mock_vector_store.search_similar_incidents.assert_called_once_with(fake_vector)
        mock_ai_advisor.generate_recommendation.assert_called_once_with(
            "Redis OOM killer triggered", fake_matches
        )
        assert result == fake_result

    def test_empty_query_returns_safe_result_without_api_calls(self):
        """Empty query returns a graceful SearchResult — no API calls made."""
        from services.incident_service import search_incidents

        result = search_incidents("   ")  # whitespace-only

        assert result.has_matches is False
        assert result.confidence_level == ConfidenceLevel.NO_MATCH
        assert len(result.matches) == 0

    @patch("services.incident_service.ai_advisor")
    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_embedding_failure_returns_graceful_result(
        self, mock_embedder, mock_vector_store, mock_ai_advisor
    ):
        """If embedder fails, returns a SearchResult with error message — no crash."""
        from services.incident_service import search_incidents

        mock_embedder.embed.side_effect = RuntimeError("Network timeout")

        result = search_incidents("some error log")

        assert result.has_matches is False
        assert result.confidence_level == ConfidenceLevel.NO_MATCH
        assert "Network timeout" in result.ai_recommendation or "query" in result.ai_recommendation.lower()
        # DB and AI must NOT be called
        mock_vector_store.search_similar_incidents.assert_not_called()
        mock_ai_advisor.generate_recommendation.assert_not_called()

    @patch("services.incident_service.ai_advisor")
    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_db_failure_returns_graceful_result(
        self, mock_embedder, mock_vector_store, mock_ai_advisor
    ):
        """If vector search fails, returns a SearchResult with error message — no crash."""
        from services.incident_service import search_incidents

        mock_embedder.embed.return_value = [0.1] * 768
        mock_vector_store.search_similar_incidents.side_effect = Exception("pgvector error")

        result = search_incidents("error log")

        assert result.has_matches is False
        assert "Database search failed" in result.ai_recommendation or "pgvector" in result.ai_recommendation
        # AI advisor must NOT be called if DB failed
        mock_ai_advisor.generate_recommendation.assert_not_called()

    @patch("services.incident_service.ai_advisor")
    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_no_db_matches_still_calls_ai_advisor(
        self, mock_embedder, mock_vector_store, mock_ai_advisor
    ):
        """Even with zero DB matches, ai_advisor is still called (for general guidance)."""
        from services.incident_service import search_incidents

        mock_embedder.embed.return_value = [0.1] * 768
        mock_vector_store.search_similar_incidents.return_value = []  # no matches
        mock_ai_advisor.generate_recommendation.return_value = SearchResult(
            query="error",
            matches=[],
            ai_recommendation="NO HISTORICAL MATCH FOUND.",
            confidence_level=ConfidenceLevel.NO_MATCH,
            has_matches=False,
        )

        result = search_incidents("brand new unknown error")

        mock_ai_advisor.generate_recommendation.assert_called_once_with(
            "brand new unknown error", []
        )
        assert result.has_matches is False

    @patch("services.incident_service.ai_advisor")
    @patch("services.incident_service.vector_store")
    @patch("services.incident_service.embedder")
    def test_query_is_stripped_before_processing(
        self, mock_embedder, mock_vector_store, mock_ai_advisor
    ):
        """Leading/trailing whitespace is stripped from the query."""
        from services.incident_service import search_incidents

        mock_embedder.embed.return_value = [0.1] * 768
        mock_vector_store.search_similar_incidents.return_value = []
        mock_ai_advisor.generate_recommendation.return_value = _make_search_result([])

        search_incidents("  error log with spaces  ")

        embedded_text = mock_embedder.embed.call_args[0][0]
        assert embedded_text == "error log with spaces"


# ---------------------------------------------------------------------------
# list_incidents tests
# ---------------------------------------------------------------------------

class TestListIncidents:

    @patch("services.incident_service.vector_store")
    def test_returns_incidents_from_db(self, mock_vector_store):
        """list_incidents returns whatever vector_store.list_all_incidents returns."""
        from services.incident_service import list_incidents

        mock_vector_store.list_all_incidents.return_value = [_make_incident(), _make_incident()]
        result = list_incidents()
        assert len(result) == 2

    @patch("services.incident_service.vector_store")
    def test_db_failure_returns_empty_list(self, mock_vector_store):
        """DB failure returns [] instead of raising (safe for dashboard rendering)."""
        from services.incident_service import list_incidents

        mock_vector_store.list_all_incidents.side_effect = Exception("connection lost")
        result = list_incidents()
        assert result == []

    @patch("services.incident_service.vector_store")
    def test_passes_limit_to_vector_store(self, mock_vector_store):
        """list_incidents passes the limit arg through to vector_store."""
        from services.incident_service import list_incidents

        mock_vector_store.list_all_incidents.return_value = []
        list_incidents(limit=50)
        mock_vector_store.list_all_incidents.assert_called_once_with(limit=50)


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------

class TestGetStats:

    @patch("services.incident_service.vector_store")
    def test_returns_count_and_connected(self, mock_vector_store):
        """get_stats returns total_incidents and db_connected=True on success."""
        from services.incident_service import get_stats

        mock_vector_store.get_incident_count.return_value = 42
        stats = get_stats()
        assert stats["total_incidents"] == 42
        assert stats["db_connected"] is True

    @patch("services.incident_service.vector_store")
    def test_db_failure_returns_safe_stats(self, mock_vector_store):
        """get_stats returns zeroed-out safe dict when DB is unreachable."""
        from services.incident_service import get_stats

        mock_vector_store.get_incident_count.side_effect = Exception("DB offline")
        stats = get_stats()
        assert stats["total_incidents"] == 0
        assert stats["db_connected"] is False
