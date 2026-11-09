"""
app/components/incident_card.py
--------------------------------
Renders a single historical incident as a collapsible card.

Used by:
  - 01_search.py   (shows similarity score, expanded by default for top result)
  - 03_dashboard.py (no similarity score, all collapsed)
"""

from __future__ import annotations

import streamlit as st

from models.incident import Incident, Severity


# Severity → emoji mapping for quick visual scanning
_SEVERITY_ICON = {
    Severity.LOW:      "🔵",
    Severity.MEDIUM:   "🟡",
    Severity.HIGH:     "🟠",
    Severity.CRITICAL: "🔴",
}


def render_incident_card(
    incident: Incident,
    *,
    show_similarity: bool = True,
    expanded: bool = False,
    index: int | None = None,
) -> None:
    """
    Render one incident as a Streamlit expander card.

    Args:
        incident:        The Incident model to display.
        show_similarity: Whether to show the cosine similarity score.
        expanded:        Whether the card starts open (True for top result).
        index:           Optional 1-based position number shown in the header.
    """
    severity_icon = _SEVERITY_ICON.get(incident.severity, "⚪")

    # --- Header line ---
    prefix = f"#{index}  " if index is not None else ""
    score_str = ""
    if show_similarity and incident.similarity_score is not None:
        score_str = f"  —  {incident.similarity_score * 100:.1f}% match"

    header = f"{prefix}{severity_icon} **{incident.title}**{score_str}"

    with st.expander(header, expanded=expanded):
        # Top metrics row
        col_sev, col_score, col_date = st.columns(3)
        with col_sev:
            st.markdown(f"**Severity**")
            st.markdown(f"{severity_icon} {incident.severity.value.upper()}")
        with col_score:
            if show_similarity and incident.similarity_score is not None:
                st.metric("Similarity", f"{incident.similarity_score * 100:.1f}%")
        with col_date:
            if incident.resolved_at:
                st.markdown("**Resolved**")
                st.markdown(incident.resolved_at.strftime("%Y-%m-%d"))
            elif incident.created_at:
                st.markdown("**Logged**")
                st.markdown(incident.created_at.strftime("%Y-%m-%d"))

        st.divider()

        # Root cause + Resolution side-by-side
        col_rc, col_res = st.columns(2)
        with col_rc:
            st.markdown("**🔍 Root Cause**")
            st.markdown(incident.root_cause)
        with col_res:
            st.markdown("**✅ Resolution**")
            st.markdown(incident.resolution)

        # Tags
        if incident.tags:
            st.markdown("**Tags:** " + "  ".join(f"`{t}`" for t in incident.tags))

        # Raw error log (collapsed by default to save space)
        with st.expander("📋 View raw error log"):
            st.code(incident.error_log, language="text")


def render_incident_row(incident: Incident) -> dict:
    """
    Return a flat dict representation of an incident for use in st.dataframe().
    Called by the dashboard to build the history table.
    """
    return {
        "Title":      incident.title,
        "Severity":   f"{_SEVERITY_ICON.get(incident.severity,'⚪')} {incident.severity.value.upper()}",
        "Tags":       ", ".join(incident.tags) if incident.tags else "—",
        "Resolved":   incident.resolved_at.strftime("%Y-%m-%d") if incident.resolved_at else "—",
        "Logged":     incident.created_at.strftime("%Y-%m-%d %H:%M") if incident.created_at else "—",
        "ID":         str(incident.id)[:8] + "…",
    }
