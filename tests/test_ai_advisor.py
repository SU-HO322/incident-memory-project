"""
tests/test_ai_advisor.py
------------------------
Unit tests for services/ai_advisor.py.

STRATEGY: Mock the Gemini generate_content API entirely.
  - Tests focus on our prompt-building logic, confidence calculation,
    and graceful error handling — not on what Gemini outputs.

Run with:  pytest tests/test_ai_advisor.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from models.incident import ConfidenceLevel, Incident, Severity


# ---------------------------------------------------------------------------
# Fixtures — build realistic Incident objects for tests
# ---------------------------------------------------------------------------

def _make_incident(
    title: str = "DB timeout",
    similarity: float = 0.85,
    severity: Severity = Severity.HIGH,
) -> Incident:
    return Incident(
        id=uuid4(),
        title=title,
        error_log="ERROR: connection pool exhausted after 30s",
        root_cause="N+1 queries under high traffic caused connection pool exhaustion",
        resolution="1. Increase pool size to 50. 2. Add query result caching. 3. Add index on user_id.",
        tags=["database", "timeout", "production"],
        severity=severity,
        resolved_at=datetime(2024, 3, 15, tzinfo=timezone.utc),
        created_at=datetime(2024, 3, 15, tzinfo=timezone.utc),
        similarity_score=similarity,
    )


def _make_generate_response(text: str) -> MagicMock:
    """Build a mock matching GenerateContentResponse.text."""
    response = MagicMock()
    response.text = text
    return response


# ---------------------------------------------------------------------------
# Confidence calculation tests (no mocking needed — pure logic)
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    """Tests for the _compute_confidence() private helper."""

    def test_no_matches_returns_no_match(self):
        from services.ai_advisor import _compute_confidence
        assert _compute_confidence([]) == ConfidenceLevel.NO_MATCH

    def test_high_similarity_returns_high(self):
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=0.90)
        assert _compute_confidence([incident]) == ConfidenceLevel.HIGH

    def test_medium_similarity_returns_medium(self):
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=0.65)
        assert _compute_confidence([incident]) == ConfidenceLevel.MEDIUM

    def test_low_similarity_returns_low(self):
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=0.52)
        assert _compute_confidence([incident]) == ConfidenceLevel.LOW

    def test_exact_high_threshold_returns_high(self):
        from config.settings import HIGH_CONFIDENCE_THRESHOLD
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=HIGH_CONFIDENCE_THRESHOLD)
        assert _compute_confidence([incident]) == ConfidenceLevel.HIGH

    def test_exact_mid_threshold_returns_medium(self):
        from config.settings import MED_CONFIDENCE_THRESHOLD
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=MED_CONFIDENCE_THRESHOLD)
        assert _compute_confidence([incident]) == ConfidenceLevel.MEDIUM

    def test_none_similarity_score_returns_no_match(self):
        from services.ai_advisor import _compute_confidence
        incident = _make_incident(similarity=0.85)
        incident.similarity_score = None
        assert _compute_confidence([incident]) == ConfidenceLevel.NO_MATCH


# ---------------------------------------------------------------------------
# Prompt builder tests (no API calls needed)
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Tests for the _build_prompt() private helper."""

    def test_includes_query_text(self):
        from services.ai_advisor import _build_prompt
        prompt = _build_prompt("OOMKilled: container memory limit exceeded", [])
        assert "OOMKilled" in prompt

    def test_includes_incident_title_when_matches_present(self):
        from services.ai_advisor import _build_prompt
        incident = _make_incident(title="Redis OOM crash")
        prompt = _build_prompt("memory error", [incident])
        assert "Redis OOM crash" in prompt

    def test_includes_similarity_percentage(self):
        from services.ai_advisor import _build_prompt
        incident = _make_incident(similarity=0.876)
        prompt = _build_prompt("error", [incident])
        assert "87.6%" in prompt

    def test_includes_root_cause_and_resolution(self):
        from services.ai_advisor import _build_prompt
        incident = _make_incident()
        prompt = _build_prompt("error", [incident])
        assert incident.root_cause in prompt
        assert incident.resolution in prompt

    def test_no_match_message_when_no_incidents(self):
        from services.ai_advisor import _build_prompt
        prompt = _build_prompt("some error", [])
        assert "None found" in prompt or "No similar" in prompt.lower() or "none" in prompt.lower()

    def test_no_match_instructions_when_no_incidents(self):
        from services.ai_advisor import _build_prompt
        prompt = _build_prompt("error", [])
        assert "NO HISTORICAL MATCH FOUND" in prompt

    def test_multiple_incidents_numbered(self):
        from services.ai_advisor import _build_prompt
        incidents = [
            _make_incident(title="Incident Alpha", similarity=0.90),
            _make_incident(title="Incident Beta", similarity=0.72),
        ]
        prompt = _build_prompt("error", incidents)
        assert "Incident 1" in prompt
        assert "Incident 2" in prompt


# ---------------------------------------------------------------------------
# generate_recommendation integration tests
# ---------------------------------------------------------------------------

class TestGenerateRecommendation:
    """Tests for the main generate_recommendation() function."""

    @patch("services.ai_advisor._get_client")
    def test_returns_search_result_with_matches(self, mock_get_client):
        """Returns a SearchResult with the correct matches and AI text."""
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_generate_response(
            "1. LIKELY ROOT CAUSE\n   Connection pool exhausted.\n"
        )
        mock_get_client.return_value = mock_client

        incident = _make_incident(similarity=0.88)
        result = generate_recommendation("DB connection failed", [incident])

        assert result.has_matches is True
        assert len(result.matches) == 1
        assert "Connection pool exhausted" in result.ai_recommendation
        assert result.confidence_level == ConfidenceLevel.HIGH

    @patch("services.ai_advisor._get_client")
    def test_returns_no_match_when_empty_matches(self, mock_get_client):
        """Returns NO_MATCH confidence when no incidents are provided."""
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_generate_response(
            "NO HISTORICAL MATCH FOUND. General triage advice..."
        )
        mock_get_client.return_value = mock_client

        result = generate_recommendation("unknown error", [])

        assert result.has_matches is False
        assert result.confidence_level == ConfidenceLevel.NO_MATCH
        assert result.top_similarity_score is None

    @patch("services.ai_advisor._get_client")
    def test_top_similarity_score_is_set(self, mock_get_client):
        """top_similarity_score reflects the best match's score."""
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_generate_response("ok")
        mock_get_client.return_value = mock_client

        incident = _make_incident(similarity=0.73)
        result = generate_recommendation("error", [incident])

        assert result.top_similarity_score == pytest.approx(0.73)

    @patch("services.ai_advisor._get_client")
    def test_api_failure_returns_graceful_fallback(self, mock_get_client):
        """API failure returns a SearchResult with fallback text, does not raise."""
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable")
        mock_get_client.return_value = mock_client

        incident = _make_incident(similarity=0.80)
        result = generate_recommendation("some error", [incident])

        # Should NOT raise — returns a result with error info in recommendation text
        assert "unavailable" in result.ai_recommendation.lower()
        # Matches are still returned (search worked, only AI generation failed)
        assert len(result.matches) == 1

    def test_empty_query_returns_safe_result(self):
        """Empty query returns a SearchResult without calling the API."""
        from services.ai_advisor import generate_recommendation

        result = generate_recommendation("", [])

        assert result.has_matches is False
        assert result.confidence_level == ConfidenceLevel.NO_MATCH
        assert "empty" in result.ai_recommendation.lower() or "cannot" in result.ai_recommendation.lower()

    @patch("services.ai_advisor._get_client")
    def test_uses_correct_generation_model(self, mock_get_client):
        """generate_content is called with the configured model name."""
        from config.settings import GEMINI_GENERATION_MODEL
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_generate_response("response")
        mock_get_client.return_value = mock_client

        generate_recommendation("error log text", [_make_incident()])

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == GEMINI_GENERATION_MODEL

    @patch("services.ai_advisor._get_client")
    def test_uses_low_temperature_for_factual_output(self, mock_get_client):
        """GenerateContentConfig uses low temperature for factual/grounded output."""
        from services.ai_advisor import generate_recommendation

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_generate_response("ok")
        mock_get_client.return_value = mock_client

        generate_recommendation("error", [_make_incident()])

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        assert config.temperature <= 0.3  # must be low for factual output
