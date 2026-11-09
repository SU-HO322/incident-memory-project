"""
app/pages/02_add_incident.py
-----------------------------
Page 2: Add Incident — log a new resolved incident into the memory system.

After a team resolves a production incident, they fill out this form.
The system embeds the error_log and stores everything in Supabase.
Future searches can then surface this incident for similar problems.

IMPORTANT: We only accept RESOLVED incidents with a documented root cause
and resolution. The system is a memory of what worked — not a ticket tracker.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

try:
    from models.incident import IncidentInput, Severity
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
    page_title="Add Incident — Incident Memory Agent",
    page_icon="➕",
    layout="centered",
)

if not _IMPORTS_OK:
    st.error("⚙️ **Setup required** — could not load configuration.")
    st.code(_IMPORT_ERROR)
    st.info("👉 Copy `.env.example` to `.env` and fill in your API keys, then restart the app.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("➕ Add Resolved Incident")
st.markdown(
    "Log a production incident that your team has **already resolved**. "
    "This builds the team's incident memory — future searches will surface this case."
)
st.info(
    "💡 **Tip:** The more detail you provide in the Error Log and Resolution fields, "
    "the better the semantic search will work for future incidents."
)
st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("add_incident_form", clear_on_submit=True):

    st.subheader("📋 Incident Details")

    # Title
    title = st.text_input(
        "Incident Title *",
        placeholder="e.g. Database connection pool exhausted under peak load",
        help="A short, descriptive name. Used as the heading in search results.",
        max_chars=200,
    )

    # Severity
    severity_options = {s.value: s for s in Severity}
    severity_labels = {
        "low":      "🔵 Low — minor degradation, workaround available",
        "medium":   "🟡 Medium — significant impact, partial outage",
        "high":     "🟠 High — major impact, most users affected",
        "critical": "🔴 Critical — full outage, all users affected",
    }
    severity_choice = st.selectbox(
        "Severity *",
        options=list(severity_labels.keys()),
        index=1,  # default: medium
        format_func=lambda k: severity_labels[k],
    )

    # Error log
    error_log = st.text_area(
        "Error Log / Stack Trace *",
        height=180,
        placeholder=(
            "Paste the actual error messages, stack traces, or log lines observed "
            "during the incident. This is what gets embedded for semantic search."
        ),
        help="This field is used for semantic similarity matching. More detail = better matches.",
    )

    # Root cause
    root_cause = st.text_area(
        "Root Cause *",
        height=100,
        placeholder="e.g. N+1 query pattern in the user feed endpoint caused connection pool exhaustion under load",
        help="What actually caused the incident? Be specific.",
    )

    # Resolution
    resolution = st.text_area(
        "Resolution *",
        height=130,
        placeholder=(
            "e.g.\n"
            "1. Restarted application pods to clear the pool.\n"
            "2. Added eager loading to the feed query (`.includes(:posts)`).\n"
            "3. Increased connection pool limit from 10 to 25.\n"
            "4. Added Datadog alert for pool utilisation > 80%."
        ),
        help="Step-by-step actions that resolved the incident. Numbered lists work great.",
    )

    # Tags
    tags_raw = st.text_input(
        "Tags (optional)",
        placeholder="database, timeout, production, api",
        help="Comma-separated labels for filtering on the dashboard.",
    )

    # Resolved date
    resolved_date = st.date_input(
        "Date Resolved (optional)",
        value=None,
        help="When was this incident resolved? Leave blank if unknown.",
    )

    st.divider()
    submitted = st.form_submit_button(
        "💾 Save Incident",
        type="primary",
        use_container_width=True,
    )

# ── Form submission ───────────────────────────────────────────────────────────
if submitted:
    # Client-side validation
    errors: list[str] = []
    if not title.strip():
        errors.append("**Title** is required.")
    if not error_log.strip():
        errors.append("**Error Log** is required.")
    elif len(error_log.strip()) < 10:
        errors.append("**Error Log** must be at least 10 characters.")
    if not root_cause.strip():
        errors.append("**Root Cause** is required.")
    elif len(root_cause.strip()) < 10:
        errors.append("**Root Cause** must be at least 10 characters.")
    if not resolution.strip():
        errors.append("**Resolution** is required.")
    elif len(resolution.strip()) < 10:
        errors.append("**Resolution** must be at least 10 characters.")

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
    else:
        # Parse tags
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        # Parse resolved_at
        resolved_at: datetime | None = None
        if resolved_date:
            resolved_at = datetime(
                resolved_date.year,
                resolved_date.month,
                resolved_date.day,
                tzinfo=timezone.utc,
            )

        # Build the input model
        incident_data = IncidentInput(
            title=title.strip(),
            error_log=error_log.strip(),
            root_cause=root_cause.strip(),
            resolution=resolution.strip(),
            tags=tags,
            severity=Severity(severity_choice),
            resolved_at=resolved_at,
        )

        # Store it
        with st.spinner("🔄 Generating embedding and saving to database…"):
            try:
                saved = incident_service.store_incident(incident_data)
                st.success(
                    f"✅ **Incident saved successfully!**\n\n"
                    f"**'{saved.title}'** has been added to the incident memory.\n\n"
                    f"ID: `{saved.id}`"
                )
                st.balloons()
            except ValueError as e:
                st.error(f"❌ Validation error: {e}")
            except RuntimeError as e:
                st.error(
                    f"❌ Failed to save incident.\n\n"
                    f"**Error:** {e}\n\n"
                    "Check your GEMINI_API_KEY and DATABASE_URL in `.env`."
                )
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
