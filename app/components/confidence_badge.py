"""
app/components/confidence_badge.py
-----------------------------------
Renders a colour-coded confidence badge in the Streamlit UI.

Confidence tiers (driven by cosine similarity score):
  HIGH     >= 0.80  → green  success box
  MEDIUM   >= 0.60  → yellow warning box
  LOW      >= 0.50  → orange warning box
  NO_MATCH  < 0.50  → grey   info box
"""

from __future__ import annotations

import streamlit as st

from models.incident import ConfidenceLevel


# Tier configuration — icon, label, Streamlit alert type, description
_TIER_CONFIG: dict[ConfidenceLevel, tuple[str, str, str, str]] = {
    ConfidenceLevel.HIGH: (
        "🟢", "High Confidence",
        "success",
        "Strong historical match found. Recommendation is well-grounded.",
    ),
    ConfidenceLevel.MEDIUM: (
        "🟡", "Medium Confidence",
        "warning",
        "Partial match found. Recommendation is likely relevant but verify carefully.",
    ),
    ConfidenceLevel.LOW: (
        "🟠", "Low Confidence",
        "warning",
        "Weak match found. Treat recommendation as a starting point, not a proven fix.",
    ),
    ConfidenceLevel.NO_MATCH: (
        "⚫", "No Historical Match",
        "info",
        "No similar incidents found. AI response is general guidance only — not from team history.",
    ),
}


def render_confidence_badge(
    level: ConfidenceLevel,
    score: float | None = None,
    show_description: bool = True,
) -> None:
    """
    Render a colour-coded confidence badge.

    Args:
        level:            The ConfidenceLevel enum value.
        score:            Raw cosine similarity score (0.0–1.0), shown as a percentage.
        show_description: If True, renders a one-line explanation under the badge.
    """
    icon, label, alert_type, description = _TIER_CONFIG.get(
        level, _TIER_CONFIG[ConfidenceLevel.NO_MATCH]
    )

    score_str = f"  |  Top match: **{score * 100:.1f}%** similarity" if score is not None else ""
    badge_text = f"{icon} **{label}**{score_str}"

    if alert_type == "success":
        st.success(badge_text)
    elif alert_type == "warning":
        st.warning(badge_text)
    else:
        st.info(badge_text)

    if show_description:
        st.caption(description)


def confidence_color(level: ConfidenceLevel) -> str:
    """Return a hex color string for use in custom HTML/CSS."""
    return {
        ConfidenceLevel.HIGH:     "#28a745",
        ConfidenceLevel.MEDIUM:   "#ffc107",
        ConfidenceLevel.LOW:      "#fd7e14",
        ConfidenceLevel.NO_MATCH: "#6c757d",
    }.get(level, "#6c757d")
