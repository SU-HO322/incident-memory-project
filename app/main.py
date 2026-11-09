"""
app/main.py — Streamlit entry point for Incident Memory Agent.

Run with:
    streamlit run app/main.py
(from the project root directory)

Streamlit discovers multi-page entries automatically from app/pages/.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path so all packages resolve
# correctly regardless of where the user launches `streamlit run` from.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # …/app
_ROOT_DIR = os.path.dirname(_THIS_DIR)                    # …/incident-memory-agent
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import streamlit as st

# ---------------------------------------------------------------------------
# Config validation — catch missing .env early and show a helpful setup guide
# ---------------------------------------------------------------------------
_CONFIG_OK: bool = True
_CONFIG_ERROR: str = ""
_CONFIG: dict = {}

try:
    from config.settings import (
        APP_TITLE,
        APP_SUBTITLE,
        GEMINI_API_KEY,
        SUPABASE_URL,
        SUPABASE_KEY,
        DATABASE_URL,
    )
    _CONFIG = {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
    }
except EnvironmentError as exc:
    _CONFIG_OK = False
    _CONFIG_ERROR = str(exc)
except Exception as exc:  # pragma: no cover
    _CONFIG_OK = False
    _CONFIG_ERROR = f"Unexpected config error: {exc}"

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Incident Memory Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/incident-memory-agent",
        "Report a bug": "https://github.com/your-org/incident-memory-agent/issues",
        "About": "# Incident Memory Agent\nAI that remembers how your team fixed it last time.",
    },
)

# ---------------------------------------------------------------------------
# Setup guide — shown if .env / credentials are missing
# ---------------------------------------------------------------------------
if not _CONFIG_OK:
    st.error("⚠️ **Configuration missing** — the app cannot start without valid credentials.")
    st.markdown("---")
    st.subheader("🔧 Quick Setup")

    st.markdown(
        """
        **Step 1 — Copy the example env file and fill in your secrets:**
        ```bash
        cp .env.example .env
        # then edit .env with your actual keys
        ```

        **Step 2 — Required variables:**
        | Variable | Description |
        |---|---|
        | `GEMINI_API_KEY` | Google AI Studio API key |
        | `SUPABASE_URL` | Your Supabase project URL |
        | `SUPABASE_KEY` | Supabase service-role or anon key |
        | `DATABASE_URL` | PostgreSQL connection string from Supabase |

        **Step 3 — Run the database schema** in your Supabase SQL Editor:
        ```
        db/schema.sql
        ```

        **Step 4 — Restart the app:**
        ```bash
        streamlit run app/main.py
        ```
        """
    )

    with st.expander("🔍 Technical error details"):
        st.code(_CONFIG_ERROR, language="text")

    st.stop()

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
title = _CONFIG.get("title", "Incident Memory Agent")
subtitle = _CONFIG.get("subtitle", "AI that remembers how your team fixed it last time.")

st.title(f"🧠 {title}")
st.markdown(f"*{subtitle}*")
st.markdown("---")

# --- Hero columns -----------------------------------------------------------
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("What can I do?")
    st.markdown(
        """
        - 🔍 **Search** past incidents using natural language — no exact keyword match needed
        - ➕ **Record** new incidents with structured resolution notes
        - 📊 **Explore** incident trends, severity distributions, and resolution patterns
        - 🤖 **Get AI recommendations** grounded in your team's real incident history
        """
    )

    st.subheader("How it works")
    st.markdown(
        """
        1. Your incident descriptions are converted to **768-dimensional embeddings** via Google's
           `text-embedding-004` model.
        2. Embeddings are stored in **Supabase pgvector** for fast similarity search.
        3. When you search, the top-N most semantically similar past incidents are retrieved.
        4. **Gemini 2.5 Flash** analyses the similar incidents and generates a grounded recommendation
           — including confidence level and step-by-step resolution guidance.
        """
    )

with col_right:
    st.subheader("Quick navigation")

    btn_cols = st.columns(1)

    with btn_cols[0]:
        st.page_link("pages/01_search.py", label="🔍 Search Incidents", icon="🔍")
        st.page_link("pages/02_add_incident.py", label="➕ Add Incident", icon="➕")
        st.page_link("pages/03_dashboard.py", label="📊 Dashboard", icon="📊")

    st.markdown("---")

    # --- Live stats (best-effort — degrade gracefully if DB not ready) ---
    st.subheader("📈 Live stats")
    try:
        from services.incident_service import get_stats

        @st.cache_data(ttl=60, show_spinner=False)
        def _get_stats() -> dict:
            return get_stats()

        stats = _get_stats()

        metric_cols = st.columns(2)
        metric_cols[0].metric("Total incidents", stats.get("total_incidents", "—"))
        metric_cols[1].metric("Resolved", stats.get("resolved_count", "—"))

        metric_cols2 = st.columns(2)
        metric_cols2[0].metric("Avg resolution (hrs)", stats.get("avg_resolution_hours", "—"))
        metric_cols2[1].metric("Services tracked", stats.get("unique_services", "—"))

    except Exception as exc:  # DB not set up yet — show placeholder
        st.info(
            "📭 No incident data yet. "
            "[Add your first incident](pages/02_add_incident.py) to get started.",
            icon="ℹ️",
        )
        with st.expander("Debug info"):
            st.code(str(exc), language="text")

st.markdown("---")

# --- Getting started checklist ----------------------------------------------
with st.expander("✅ Getting started checklist", expanded=False):
    st.markdown(
        """
        - [x] Install dependencies: `pip install -r requirements.txt`
        - [x] Copy `.env.example` → `.env` and fill in your API keys
        - [x] Run `db/schema.sql` in Supabase SQL Editor to create the `incidents` table + pgvector index
        - [ ] Add your first incident via the **Add Incident** page
        - [ ] Search for a similar past incident using natural language
        """
    )

st.caption(
    "Built with ❤️ using Streamlit · Google Gemini · Supabase pgvector · psycopg3"
)
