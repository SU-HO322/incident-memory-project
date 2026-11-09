"""
services/embedder.py
--------------------
Converts raw text (error logs, stack traces) into 768-dimensional
semantic vectors using Gemini's text-embedding-004 model.

WHY A SEPARATE MODULE?
  Keeping embedding isolated means:
  - Easy to swap embedding models in one place
  - Fully testable without touching the database
  - Clear single responsibility

SDK: google-genai >= 3.0.0
  import google.genai as genai
  client.models.embed_content(model=..., contents=...)
  → response.embeddings[0].values  (list[float], 768 dims)
"""

from __future__ import annotations

import logging
from functools import lru_cache

import google.genai as genai
from google.genai import types as genai_types

from config.settings import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client — created once per process (thread-safe, connection-pooled internally)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """
    Return a cached Gemini client.
    lru_cache(maxsize=1) means only one Client is ever created per process —
    no repeated authentication overhead on every embed() call.
    """
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """
    Convert a text string into a 768-dimensional embedding vector.

    This is the ONLY public function in this module.
    Everything else (search, storage, AI advice) depends on this output.

    Args:
        text: Any text — an error log, stack trace, or search query.
              Must be non-empty after stripping whitespace.

    Returns:
        list[float] of exactly 768 values representing the semantic
        meaning of the text in high-dimensional space.

    Raises:
        ValueError:   If text is empty or whitespace-only.
        RuntimeError: If the Gemini API call fails (network, auth, quota).
    """
    text = text.strip()
    if not text:
        raise ValueError("embed() received empty text — cannot generate embedding.")

    # Truncate very long inputs to stay within model token limits (~2048 tokens).
    # We keep the first 8000 characters which covers ~2000 tokens safely.
    if len(text) > 8000:
        logger.warning(
            "Input text truncated from %d to 8000 characters for embedding.", len(text)
        )
        text = text[:8000]

    try:
        client = _get_client()
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(output_dimensionality=768),
        )

        # Validate response structure before accessing values
        if not response.embeddings or response.embeddings[0].values is None:
            raise RuntimeError(
                f"Gemini returned an empty embedding for model '{GEMINI_EMBEDDING_MODEL}'. "
                "Check that the model name is correct and your API key has access."
            )

        vector: list[float] = response.embeddings[0].values

        # Ensure vector strictly conforms to 768 dimensions for database vector(768)
        if len(vector) != 768:
            if len(vector) > 768:
                logger.warning(
                    "Embedding model returned %d dimensions (expected 768). Truncating to 768.",
                    len(vector),
                )
                vector = vector[:768]
            else:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected 768 dimensions, received {len(vector)}."
                )

        logger.debug(
            "Embedded %d chars → %d-dim vector (first 3 values: %s...)",
            len(text), len(vector), vector[:3],
        )
        return vector

    except ValueError:
        raise   # re-raise our own validation errors unchanged
    except Exception as exc:
        # Catch any google-genai API error (auth, quota, network, etc.)
        logger.error("Gemini embed_content failed: %s", exc)
        raise RuntimeError(
            f"Failed to generate embedding: {exc}\n"
            "Check your GEMINI_API_KEY and network connection."
        ) from exc


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts sequentially.

    The Gemini free tier doesn't support true batch embedding,
    so this calls embed() in a loop. Useful for seeding the database
    with multiple incidents at once.

    Args:
        texts: List of non-empty strings.

    Returns:
        List of embedding vectors in the same order as the input.

    Raises:
        ValueError:   If any text is empty.
        RuntimeError: If any embedding call fails.
    """
    if not texts:
        return []

    logger.info("Embedding batch of %d texts...", len(texts))
    return [embed(t) for t in texts]


# Backwards compatibility alias
embed_text = embed

