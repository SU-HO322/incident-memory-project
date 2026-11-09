"""
services/ai_advisor.py
----------------------
RAG-based incident analysis.

Takes a user's error query + a list of semantically similar historical
incidents and asks Gemini 2.5 Flash to synthesise a grounded recommendation.

THE CORE RULE: The AI must ONLY recommend what real incidents support.
  - High confidence  (>=0.80 similarity) → recommend the stored resolution directly
  - Medium confidence (>=0.60 similarity) → suggest investigation paths from similar cases
  - Low confidence   (>=0.50 similarity)  → AI reasons broadly, explicitly flags uncertainty
  - No matches                            → Returns "no similar incidents found" — never guesses

SDK: google-genai >= 3.0.0
  client.models.generate_content(model=..., contents=..., config=GenerateContentConfig(...))
  → response.text  (str)
"""

from __future__ import annotations

import logging
from functools import lru_cache

import google.genai as genai
from google.genai import types as genai_types

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_GENERATION_MODEL,
    HIGH_CONFIDENCE_THRESHOLD,
    MED_CONFIDENCE_THRESHOLD,
    SIMILARITY_THRESHOLD,
)
from models.incident import ConfidenceLevel, Incident, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines the AI's persona and strict grounding rules
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer and incident response advisor.

Your job is to analyse a production incident and recommend a resolution based ONLY on the
historical incident records provided to you. These records are real, documented cases from
the team's incident history.

STRICT RULES:
1. Base every recommendation on the provided historical incidents. Do not invent fixes.
2. Always cite which historical incident(s) support your recommendation (use their titles).
3. If the historical incidents are not relevant enough, say so clearly — do not guess.
4. Be concise and actionable. Engineers read this under pressure.
5. Structure your response exactly as specified in the user prompt.
"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Cached Gemini client — created once per process."""
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(query: str, matches: list[Incident]) -> str:
    """
    Construct the RAG prompt by combining the user's query with the
    retrieved historical incidents as grounding context.

    The structure is:
      [Retrieved incidents block]
      [Current incident query]
      [Instructions for response format]
    """
    sections: list[str] = []

    # --- Historical context block ---
    if matches:
        sections.append("=" * 60)
        sections.append("HISTORICAL INCIDENTS (retrieved by semantic similarity)")
        sections.append("=" * 60)

        for i, incident in enumerate(matches, start=1):
            score = incident.similarity_score or 0.0
            pct = f"{score * 100:.1f}%"
            sections.append(f"\n--- Incident {i} of {len(matches)} | Similarity: {pct} ---")
            sections.append(f"Title      : {incident.title}")
            sections.append(f"Severity   : {incident.severity.value.upper()}")
            if incident.tags:
                sections.append(f"Tags       : {', '.join(incident.tags)}")
            sections.append(f"Error log  :\n{incident.error_log[:500]}")
            sections.append(f"Root cause : {incident.root_cause}")
            sections.append(f"Resolution : {incident.resolution}")
            sections.append(f"--- End Incident {i} ---")
    else:
        sections.append("HISTORICAL INCIDENTS: None found above the similarity threshold.")

    # --- Current incident ---
    sections.append("\n" + "=" * 60)
    sections.append("CURRENT INCIDENT (needs analysis)")
    sections.append("=" * 60)
    sections.append(query)

    # --- Instructions ---
    sections.append("\n" + "=" * 60)
    sections.append("REQUIRED RESPONSE FORMAT")
    sections.append("=" * 60)

    if matches:
        sections.append(
            "Based on the historical incidents above, provide:\n\n"
            "1. LIKELY ROOT CAUSE\n"
            "   What is probably causing this based on the historical patterns?\n\n"
            "2. RECOMMENDED RESOLUTION\n"
            "   Step-by-step actions to resolve this, grounded in the historical fixes.\n\n"
            "3. SOURCE REFERENCES\n"
            "   Which historical incident(s) support this recommendation? (cite by title)\n\n"
            "4. CAVEATS\n"
            "   What is different about this incident vs the historical ones? Any unknowns?\n\n"
            "Be concise. Use bullet points. Do not repeat the incident data back."
        )
    else:
        sections.append(
            "No similar historical incidents were found.\n\n"
            "Provide a brief general triage guide for this type of error, clearly noting "
            "that this recommendation is NOT based on historical team data and should be "
            "treated as general advice only.\n\n"
            "Start your response with: 'NO HISTORICAL MATCH FOUND.'"
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendation(query: str, matches: list[Incident]) -> SearchResult:
    """
    Run the RAG pipeline: build a grounded prompt → call Gemini → return SearchResult.

    Args:
        query:   The raw error log / stack trace the user submitted.
        matches: Similar incidents retrieved from the vector store
                 (may be empty if no incidents are stored or none match).

    Returns:
        SearchResult with:
          - matches         : the passed-in similar incidents
          - ai_recommendation: Gemini's grounded response text
          - confidence_level : high / medium / low / no_match
          - top_similarity_score: score of the best match (or None)
    """
    if not query.strip():
        logger.warning("generate_recommendation called with empty query.")
        return _empty_result(query)

    # Determine confidence level from the top match's similarity score
    confidence = _compute_confidence(matches)

    # Build the grounded prompt
    prompt = _build_prompt(query, matches)

    # Call Gemini generation models with graceful fallback
    client = _get_client()
    candidate_models = [GEMINI_GENERATION_MODEL]
    for fallback in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    recommendation_text = ""
    last_error: Exception | None = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,          # low temperature = more factual, less creative
                    max_output_tokens=1024,
                ),
            )
            if response and response.text:
                recommendation_text = response.text
                last_error = None
                break
        except Exception as exc:
            logger.warning("Gemini generate_content failed for model '%s': %s", model_name, exc)
            last_error = exc

    if not recommendation_text and last_error:
        logger.error("All Gemini generation attempts failed: %s", last_error)
        recommendation_text = (
            f"⚠️ AI recommendation unavailable: {last_error}\n\n"
            "The similarity search results above are still valid — "
            "review the matched incidents manually."
        )

    logger.info(
        "Generated recommendation | confidence=%s | matches=%d",
        confidence.value, len(matches),
    )

    return SearchResult(
        query=query,
        matches=matches,
        ai_recommendation=recommendation_text,
        confidence_level=confidence,
        top_similarity_score=matches[0].similarity_score if matches else None,
        has_matches=bool(matches),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_confidence(matches: list[Incident]) -> ConfidenceLevel:
    """
    Derive a confidence tier from the top match's similarity score.

    Thresholds are set in config/settings.py so they can be tuned
    without touching this logic.
    """
    if not matches or matches[0].similarity_score is None:
        return ConfidenceLevel.NO_MATCH

    top_score = matches[0].similarity_score

    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    if top_score >= MED_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    if top_score >= SIMILARITY_THRESHOLD:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.NO_MATCH


def _empty_result(query: str) -> SearchResult:
    """Return a safe empty SearchResult when query is blank."""
    return SearchResult(
        query=query,
        matches=[],
        ai_recommendation="Cannot analyse an empty query.",
        confidence_level=ConfidenceLevel.NO_MATCH,
        top_similarity_score=None,
        has_matches=False,
    )
