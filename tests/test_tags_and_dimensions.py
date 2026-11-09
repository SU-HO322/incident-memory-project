"""
tests/test_tags_and_dimensions.py
---------------------------------
Unit tests verifying bug fixes:
  1. Tag normalization for PostgreSQL array string representations and lists.
  2. Severity normalization (case insensitivity).
  3. Embedding dimension enforcement (768 dimensions).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from models.incident import Incident, IncidentInput, Severity, normalize_tags


class TestTagNormalization:
    """Test that all tag formats are cleanly normalized to list[str]."""

    def test_postgres_array_string(self):
        result = normalize_tags("{vpn,network,remote-access}")
        assert result == ["vpn", "network", "remote-access"]

    def test_comma_separated_string(self):
        result = normalize_tags("vpn, network, remote-access")
        assert result == ["vpn", "network", "remote-access"]

    def test_json_list_string(self):
        result = normalize_tags('["vpn", "network"]')
        assert result == ["vpn", "network"]

    def test_python_list(self):
        result = normalize_tags(["vpn", "network"])
        assert result == ["vpn", "network"]

    def test_empty_string_and_none(self):
        assert normalize_tags("") == []
        assert normalize_tags("{}") == []
        assert normalize_tags("[]") == []
        assert normalize_tags(None) == []

    def test_incident_model_accepts_postgres_tag_string(self):
        """Verify Incident model parses '{vpn,network}' without validation error."""
        inc = Incident(
            title="VPN Gateway Down",
            error_log="Connection refused on port 1194",
            root_cause="OpenVPN daemon crashed",
            resolution="Restarted OpenVPN systemd unit",
            tags="{vpn,network,remote-access}",
            severity="MEDIUM",
        )
        assert inc.tags == ["vpn", "network", "remote-access"]
        assert inc.severity == Severity.MEDIUM

    def test_incident_input_model_accepts_postgres_tag_string(self):
        """Verify IncidentInput model parses '{db,timeout}' without validation error."""
        inc_in = IncidentInput(
            title="Database Connection Pool Exhausted",
            error_log="QueuePool limit of size 20 overflow 10 reached",
            root_cause="Unclosed database sessions",
            resolution="Added pool_pre_ping and increased pool_size",
            tags="{db,timeout}",
            severity="critical",
        )
        assert inc_in.tags == ["db", "timeout"]
        assert inc_in.severity == Severity.CRITICAL


class TestDimensionEnforcement:
    """Test that embedding generation strictly returns 768 dimensions."""

    @patch("services.embedder._get_client")
    def test_truncates_oversized_embedding_to_768(self, mock_get_client):
        """If API returns 3072 dimensions, embed() truncates to 768 to match db vector(768)."""
        from services.embedder import embed

        mock_resp = MagicMock()
        mock_embedding = MagicMock()
        # Return 3072 dimensions
        mock_embedding.values = [0.1] * 3072
        mock_resp.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        vector = embed("some error log text")
        assert len(vector) == 768
