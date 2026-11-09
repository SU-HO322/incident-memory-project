"""
scripts/smoke_test.py
---------------------
End-to-end diagnostics and smoke testing tool for Incident Memory Agent.

Verifies:
  1. Configuration & .env loading
  2. Database connectivity (psycopg3 + PostgreSQL)
  3. pgvector extension and table schema
  4. Google GenAI API (text-embedding-004)
  5. Google GenAI API (gemini-2.5-flash)
  6. End-to-end search pipeline

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_smoke_test() -> None:
    print("=" * 70)
    print("🔍 Incident Memory Agent — End-to-End System Smoke Test")
    print("=" * 70)

    all_passed = True

    # -----------------------------------------------------------------------
    # Step 1: Configuration check
    # -----------------------------------------------------------------------
    print("\n[1/6] Checking configuration and environment variables...")
    try:
        from config.settings import (
            DATABASE_URL,
            GEMINI_API_KEY,
            GEMINI_EMBEDDING_MODEL,
            GEMINI_GENERATION_MODEL,
            SUPABASE_KEY,
            SUPABASE_URL,
        )

        EMBEDDING_DIMENSIONS = 768
        masked_gemini = GEMINI_API_KEY[:6] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 10 else "***"
        masked_db = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "***"

        print("  ✅ Environment variables loaded:")
        print(f"     • GEMINI_API_KEY: {masked_gemini}")
        print(f"     • SUPABASE_URL  : {SUPABASE_URL}")
        print(f"     • DATABASE_URL  : postgresql://...@{masked_db}")
        print(f"     • Models        : {GEMINI_EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS}d) | {GEMINI_GENERATION_MODEL}")
    except EnvironmentError as exc:
        print(f"  ❌ Missing configuration: {exc}")
        print("  💡 Create a .env file based on .env.example with your real keys.")
        all_passed = False
        return

    # -----------------------------------------------------------------------
    # Step 2: Database connectivity
    # -----------------------------------------------------------------------
    print("\n[2/6] Testing PostgreSQL connection via psycopg3...")
    try:
        from services.vector_store import get_connection

        with get_connection() as conn:
            row = conn.execute("SELECT version();").fetchone()
            pg_ver = row["version"] if row else "Unknown"
        print(f"  ✅ Connected successfully: {pg_ver[:55]}...")
    except Exception as exc:
        print(f"  ❌ Database connection failed: {exc}")
        print("  💡 Verify DATABASE_URL in .env and check your Supabase network access.")
        all_passed = False

    # -----------------------------------------------------------------------
    # Step 3: Schema & pgvector check
    # -----------------------------------------------------------------------
    print("\n[3/6] Verifying pgvector extension & incidents table...")
    try:
        with get_connection() as conn:
            # Check vector extension
            ext_row = conn.execute(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
            ).fetchone()
            if ext_row:
                print(f"  ✅ pgvector extension active: v{ext_row['extversion']}")
            else:
                print("  ❌ pgvector extension NOT installed. Run 'CREATE EXTENSION vector;' in Supabase.")
                all_passed = False

            # Check incidents table
            tbl_row = conn.execute(
                "SELECT count(*) as cnt FROM information_schema.tables WHERE table_name = 'incidents';"
            ).fetchone()
            if tbl_row and tbl_row["cnt"] > 0:
                count_row = conn.execute("SELECT count(*) as total FROM incidents;").fetchone()
                total = count_row["total"] if count_row else 0
                print(f"  ✅ 'incidents' table exists (Current records: {total})")
            else:
                print("  ❌ 'incidents' table missing. Run db/schema.sql in Supabase SQL Editor.")
                all_passed = False
    except Exception as exc:
        print(f"  ❌ Schema verification query failed: {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # Step 4: Gemini Embeddings API check
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Testing Gemini Embeddings ({GEMINI_EMBEDDING_MODEL})...")
    test_vector = None
    try:
        from services.embedder import embed

        test_phrase = "Database connection timed out during peak load"
        test_vector = embed(test_phrase)
        if len(test_vector) == EMBEDDING_DIMENSIONS:
            print(f"  ✅ Embedding generated successfully ({len(test_vector)} dimensions)")
        else:
            print(f"  ❌ Dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(test_vector)}")
            all_passed = False
    except Exception as exc:
        print(f"  ❌ Embedding generation failed: {exc}")
        print("  💡 Verify your GEMINI_API_KEY and Google AI Studio quota.")
        all_passed = False

    # -----------------------------------------------------------------------
    # Step 5: Gemini Generation API check
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Testing Gemini Generation ({GEMINI_GENERATION_MODEL})...")
    try:
        from services.ai_advisor import generate_recommendation

        result = generate_recommendation("QueuePool timeout error", matches=[])
        if result and result.ai_recommendation:
            preview = result.ai_recommendation.strip().split("\n")[0][:75]
            print(f"  ✅ Generation response received: \"{preview}...\"")
        else:
            print("  ❌ Generation returned empty recommendation.")
            all_passed = False
    except Exception as exc:
        print(f"  ❌ Generation failed: {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # Step 6: End-to-end search pipeline check
    # -----------------------------------------------------------------------
    print("\n[6/6] Testing complete IncidentService search pipeline...")
    try:
        from services.incident_service import search_incidents

        search_res = search_incidents("timeout error in database connection pool")
        print(f"  ✅ Pipeline executed cleanly:")
        print(f"     • Matched incidents : {len(search_res.matches)}")
        print(f"     • Confidence level  : {search_res.confidence_level.value}")
        print(f"     • Recommendation len: {len(search_res.ai_recommendation)} characters")
    except Exception as exc:
        print(f"  ❌ Search pipeline failed: {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL SYSTEMS GO! All 6 integration checks passed.")
        print("You are ready to launch the app: streamlit run app/main.py")
    else:
        print("⚠️ SOME CHECKS FAILED. Review the errors above to complete setup.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
