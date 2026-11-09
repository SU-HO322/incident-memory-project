"""
app/pages/03_dashboard.py
--------------------------
Page 3: Dashboard — browse all stored incidents, view statistics.

Shows:
  - Summary stat cards (total incidents, by severity)
  - Sortable, searchable incident table
  - Expandable detail cards for any incident
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

try:
    from app.components.incident_card import render_incident_card, render_incident_row
    from models.incident import Severity
    from services import incident_service
    _IMPORTS_OK = True
except EnvironmentError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)
except Exception as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Incident Memory Agent",
    page_icon="📊",
    layout="wide",
)

if not _IMPORTS_OK:
    st.error("⚙️ **Setup required** — could not load configuration.")
    st.code(_IMPORT_ERROR)
    st.info("👉 Copy `.env.example` to `.env` and fill in your API keys, then restart the app.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 Incident History Dashboard")
st.markdown("All resolved incidents stored in the team's incident memory.")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading incidents from database…"):
    stats       = incident_service.get_stats()
    incidents   = incident_service.list_incidents(limit=200)

# ── Stat cards ────────────────────────────────────────────────────────────────
st.subheader("📈 Summary")

col1, col2, col3, col4, col5 = st.columns(5)

total = stats.get("total_incidents", 0)
db_ok = stats.get("db_connected", False)

# Count by severity (from loaded incidents for accuracy)
severity_counts = {s: 0 for s in Severity}
for inc in incidents:
    severity_counts[inc.severity] = severity_counts.get(inc.severity, 0) + 1

with col1:
    st.metric("Total Incidents", total, help="All resolved incidents in the database")
with col2:
    st.metric("🔴 Critical", severity_counts.get(Severity.CRITICAL, 0))
with col3:
    st.metric("🟠 High", severity_counts.get(Severity.HIGH, 0))
with col4:
    st.metric("🟡 Medium", severity_counts.get(Severity.MEDIUM, 0))
with col5:
    st.metric("🔵 Low", severity_counts.get(Severity.LOW, 0))

# DB connection status
if not db_ok:
    st.warning(
        "⚠️ Could not connect to the database. "
        "Check your `DATABASE_URL` in `.env` and ensure Supabase is reachable."
    )

st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if not incidents:
    st.info(
        "📭 **No incidents stored yet.**\n\n"
        "Use the **Add Incident** page to log your first resolved incident. "
        "Once you have a few incidents, searches will return meaningful results."
    )
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
st.subheader("🔎 Filter & Browse")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    severity_filter = st.multiselect(
        "Severity",
        options=[s.value for s in Severity],
        default=[],
        placeholder="All severities",
        format_func=lambda v: {
            "low": "🔵 Low", "medium": "🟡 Medium",
            "high": "🟠 High", "critical": "🔴 Critical",
        }.get(v, v),
    )

with filter_col2:
    all_tags = sorted({tag for inc in incidents for tag in inc.tags})
    tag_filter = st.multiselect(
        "Tags",
        options=all_tags,
        default=[],
        placeholder="All tags",
    )

with filter_col3:
    search_text = st.text_input(
        "Search titles",
        placeholder="e.g. database, timeout…",
        help="Filters incident titles by keyword (case-insensitive).",
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = incidents

if severity_filter:
    filtered = [i for i in filtered if i.severity.value in severity_filter]

if tag_filter:
    filtered = [i for i in filtered if any(t in i.tags for t in tag_filter)]

if search_text.strip():
    kw = search_text.strip().lower()
    filtered = [i for i in filtered if kw in i.title.lower()]

st.caption(f"Showing {len(filtered)} of {len(incidents)} incidents")
st.divider()

# ── Table view ────────────────────────────────────────────────────────────────
if filtered:
    st.subheader("📋 Incidents Table")
    rows = [render_incident_row(inc) for inc in filtered]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.drop(columns=["ID"]),   # hide raw UUID column
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Detail cards ─────────────────────────────────────────────────────────
    st.subheader("🗂️ Incident Details")
    st.caption("Click any incident below to expand its full details.")

    for incident in filtered:
        render_incident_card(
            incident,
            show_similarity=False,
            expanded=False,
        )
else:
    st.warning("No incidents match the current filters.")
