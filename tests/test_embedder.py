"""
tests/test_embedder.py
----------------------
Unit tests for services/embedder.py.

STRATEGY: Mock the Gemini API client entirely.
  - No real API calls are made — tests run offline and instantly.
  - We test OUR logic (validation, truncation, error handling),
    not Google's API.

Run with:  pytest tests/test_embedder.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build realistic mock responses matching the real SDK structure
# ---------------------------------------------------------------------------

def _make_embed_response(values: list[float]) -> MagicMock:
    """
    Build a mock that mirrors the real EmbedContentResponse structure:
      response.embeddings[0].values → list[float]
    """
    embedding = MagicMock()
    embedding.values = values

    response = MagicMock()
    response.embeddings = [embedding]
    return response


def _make_768_vector() -> list[float]:
    """Return a fake 768-dimensional vector (all 0.1 values)."""
    return [0.1] * 768


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests for the main embed() function."""

    @patch("services.embedder._get_client")
    def test_returns_768_dim_vector(self, mock_get_client):
        """embed() returns a list of 768 floats on success."""
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        result = embed("Database connection timeout error")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(v, float) for v in result)

    @patch("services.embedder._get_client")
    def test_calls_correct_model(self, mock_get_client):
        """embed() calls embed_content with the configured model name."""
        from config.settings import GEMINI_EMBEDDING_MODEL
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        embed("some error text")

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == GEMINI_EMBEDDING_MODEL

    @patch("services.embedder._get_client")
    def test_passes_text_as_contents(self, mock_get_client):
        """embed() passes the stripped text as the 'contents' argument."""
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        embed("  my error text  ")  # has leading/trailing whitespace

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == "my error text"  # stripped

    def test_raises_on_empty_string(self):
        """embed() raises ValueError for empty input."""
        from services.embedder import embed

        with pytest.raises(ValueError, match="empty text"):
            embed("")

    def test_raises_on_whitespace_only(self):
        """embed() raises ValueError for whitespace-only input."""
        from services.embedder import embed

        with pytest.raises(ValueError, match="empty text"):
            embed("   \n\t  ")

    @patch("services.embedder._get_client")
    def test_truncates_long_input(self, mock_get_client):
        """embed() truncates inputs longer than 8000 characters."""
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        long_text = "x" * 10_000
        embed(long_text)

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert len(call_kwargs["contents"]) == 8000

    @patch("services.embedder._get_client")
    def test_does_not_truncate_short_input(self, mock_get_client):
        """embed() does NOT truncate inputs under 8000 characters."""
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        text = "short error message"
        embed(text)

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == text

    @patch("services.embedder._get_client")
    def test_raises_runtime_error_on_api_failure(self, mock_get_client):
        """embed() wraps API exceptions in RuntimeError with a helpful message."""
        from services.embedder import embed

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("API quota exceeded")
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            embed("some text")

    @patch("services.embedder._get_client")
    def test_raises_on_empty_api_response(self, mock_get_client):
        """embed() raises RuntimeError if the API returns no embeddings."""
        from services.embedder import embed

        mock_response = MagicMock()
        mock_response.embeddings = []  # empty list — no embedding returned

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="empty embedding"):
            embed("some text")


class TestEmbedBatch:
    """Tests for the embed_batch() helper."""

    @patch("services.embedder._get_client")
    def test_returns_list_of_vectors(self, mock_get_client):
        """embed_batch() returns one vector per input text."""
        from services.embedder import embed_batch

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _make_embed_response(_make_768_vector())
        mock_get_client.return_value = mock_client

        texts = ["error one", "error two", "error three"]
        results = embed_batch(texts)

        assert len(results) == 3
        assert all(len(v) == 768 for v in results)

    def test_empty_list_returns_empty(self):
        """embed_batch([]) returns [] without calling the API."""
        from services.embedder import embed_batch

        result = embed_batch([])
        assert result == []

    @patch("services.embedder._get_client")
    def test_preserves_order(self, mock_get_client):
        """embed_batch() returns vectors in the same order as input texts."""
        from services.embedder import embed_batch

        # Return incrementing vectors so we can check order
        call_count = [0]
        def side_effect(**kwargs):
            val = float(call_count[0])
            call_count[0] += 1
            return _make_embed_response([val] * 768)

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = side_effect
        mock_get_client.return_value = mock_client

        texts = ["first", "second", "third"]
        results = embed_batch(texts)

        assert results[0][0] == 0.0
        assert results[1][0] == 1.0
        assert results[2][0] == 2.0
