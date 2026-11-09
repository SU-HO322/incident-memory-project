"""
app/pages/01_search.py
-----------------------
Page 1: Search — paste an error log and get AI-powered recommendations.

This is the core demo page. The flow is:
  1. User pastes error log / stack trace into the text area
  2. Clicks "Search Incidents"
  3. App embeds the query → searches pgvector → calls Gemini → renders results

Path setup: adds project root to sys.path so services/* can be imported
regardless of where `streamlit run` is executed from.
"""

from __future__ import annotations

import os
import sys

# --- Path setup (must be before any project imports) ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

# ── Project imports (after path setup) ──────────────────────────────────────
try:
    from app.components.confidence_badge import render_confidence_badge
    from app.components.incident_card import render_incident_card
    from models.incident import ConfidenceLevel
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
    page_title="Search — Incident Memory Agent",
    page_icon="🔍",
    layout="wide",
)

# ── Setup guard ───────────────────────────────────────────────────────────────
if not _IMPORTS_OK:
    st.error("⚙️ **Setup required** — could not load configuration.")
    st.code(_IMPORT_ERROR)
    st.info("👉 Copy `.env.example` to `.env` and fill in your API keys, then restart the app.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔍 Search Incidents")
st.markdown(
    "Paste an error log or stack trace below. "
    "The system will find similar past incidents and generate a grounded recommendation."
)
st.divider()

# ── Sample error (for demos) ──────────────────────────────────────────────────
_SAMPLE_ERROR = """\
ERROR 2024-03-15 14:32:11 [app-server-01] Database connection failed
  psycopg2.OperationalError: could not connect to server: Connection refused
  Is the server running on host "db-prod-01" (10.0.1.50) and accepting
  TCP/IP connections on port 5432?

Traceback (most recent call last):
  File "/app/db/pool.py", line 87, in get_connection
    conn = pool.getconn(timeout=30)
  File "/usr/local/lib/python3.11/site-packages/psycopg2/pool.py", line 168, in getconn
    raise PoolError("connection pool exhausted")
psycopg2.pool.PoolError: connection pool exhausted

Active connections: 50/50  |  Waiting queue: 143  |  Avg query time: 8400ms
""".strip()

col_header, col_sample = st.columns([4, 1])
with col_sample:
    if st.button("📋 Load sample error", use_container_width=True):
        st.session_state["search_query"] = _SAMPLE_ERROR

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_area(
    label="Error log / Stack trace",
    value=st.session_state.get("search_query", ""),
    height=200,
    placeholder="Paste your error log, stack trace, or describe the incident here…",
    help="The more detail you provide, the better the semantic match will be.",
    key="search_query",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    search_clicked = st.button(
        "🔍 Search Incidents",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["search_query"] = ""
        st.rerun()

# ── Search execution ──────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("🔎 Embedding query and searching incident history…"):
        result = incident_service.search_incidents(query)

    st.session_state["last_result"] = result

# ── Results rendering ─────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result is not None:
    st.divider()
    st.subheader("📊 Results")

    # --- Confidence badge ---
    render_confidence_badge(
        result.confidence_level,
        score=result.top_similarity_score,
        show_description=True,
    )

    st.divider()

    # --- Two-column layout: AI recommendation | Similar incidents ---
    col_ai, col_matches = st.columns([3, 2], gap="large")

    with col_ai:
        st.markdown("### 🤖 AI Recommendation")
        if result.ai_recommendation:
            st.markdown(result.ai_recommendation)
        else:
            st.info("No recommendation generated.")

    with col_matches:
        st.markdown(f"### 📂 Similar Incidents ({len(result.matches)} found)")

        if result.has_matches:
            for i, incident in enumerate(result.matches, start=1):
                render_incident_card(
                    incident,
                    show_similarity=True,
                    expanded=(i == 1),   # auto-expand top result only
                    index=i,
                )
        else:
            st.info(
                "No similar incidents found in the database.\n\n"
                "**Tip:** Use the **Add Incident** page to log your first resolved incident."
            )

    # --- Source query summary ---
    with st.expander("📋 View your original query"):
        st.code(result.query, language="text")
