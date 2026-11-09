"""
scripts/seed_data.py
---------------------
Seeds the Supabase database with realistic production incidents for demo / testing.

Usage:
    # Live seed (requires valid .env with GEMINI_API_KEY and DATABASE_URL):
    python scripts/seed_data.py

    # Dry-run (prints incidents without connecting to Gemini or Supabase):
    python scripts/seed_data.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

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

from models.incident import IncidentInput, Severity

# ---------------------------------------------------------------------------
# 5 Realistic Production Incidents for Hackathon Demo
# ---------------------------------------------------------------------------

SAMPLE_INCIDENTS: list[IncidentInput] = [
    IncidentInput(
        title="PostgreSQL Connection Pool Exhaustion under Peak Traffic",
        error_log=(
            "sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 10 reached, "
            "connection timed out, timeout 30.00 (Background on this error at: https://sqlalche.me/e/20/3o7r)\n"
            "  File '/app/routers/orders.py', line 142, in create_order\n"
            "    async with db_session() as session:\n"
            "  File 'sqlalchemy/pool/base.py', line 378, in connect"
        ),
        root_cause=(
            "Unclosed database sessions in FastAPI async background tasks holding connections open during "
            "external payment gateway HTTP webhook calls, exhausting the SQLAlchemy connection pool of 30."
        ),
        resolution=(
            "1. Increased pool_size to 40 and max_overflow to 20 in SQLAlchemy engine config.\n"
            "2. Enabled pool_pre_ping=True and set pool_recycle=1800.\n"
            "3. Refactored background tasks to use explicit async context managers releasing DB sessions before making external HTTP requests."
        ),
        tags=["database", "postgresql", "fastapi", "connection-pool", "sqlalchemy"],
        severity=Severity.CRITICAL,
        resolved_at=datetime(2026, 10, 15, 14, 30, tzinfo=timezone.utc),
    ),
    IncidentInput(
        title="Redis Cache Stampede causing 504 Gateway Timeouts",
        error_log=(
            "redis.exceptions.ConnectionError: Error 111 connecting to 10.0.4.12:6379. Connection refused.\n"
            "Downstream 504 Gateway Timeout from nginx/1.24.0 on /api/v1/products/trending\n"
            "HTTP/1.1 504 Gateway Time-out - Client response time > 15000ms"
        ),
        root_cause=(
            "Hot product catalog cache keys had synchronized fixed TTL of 3600s with no jitter. "
            "Simultaneous expiration caused 50k requests to stampede directly into the primary database, "
            "saturating Redis connection limit (maxclients=10000)."
        ),
        resolution=(
            "1. Added randomized jitter to cache expiration: TTL = 3600 + random.randint(0, 300).\n"
            "2. Implemented probabilistic early expiration (XFetch algorithm) for top 100 hot keys.\n"
            "3. Raised Redis maxclients to 30000 and added Redis connection pooling in backend client."
        ),
        tags=["redis", "cache", "timeout", "nginx", "performance"],
        severity=Severity.HIGH,
        resolved_at=datetime(2026, 10, 22, 9, 15, tzinfo=timezone.utc),
    ),
    IncidentInput(
        title="JWT Expired Signature Verification Failure during Token Key Rotation",
        error_log=(
            "jwt.exceptions.ExpiredSignatureError: Signature has expired.\n"
            "Key ID 'auth-key-2026-q3' not found in JWKS cache during public key rotation.\n"
            "HTTP 401 Unauthorized returned on 1,420 user API requests in 3 minutes."
        ),
        root_cause=(
            "Identity provider rotated RSA signing keys without a 24-hour overlap grace period. "
            "Backend API gateway cached the JWKS keyset statically at boot and failed to refresh "
            "when encountering tokens signed by the new key."
        ),
        resolution=(
            "1. Added background JWKS refresh thread checking for unknown kid (Key ID) headers with 10s cooldown.\n"
            "2. Configured auth service to sign with new key but keep previous key valid in JWKS for 48-hour deprecation window.\n"
            "3. Added fallback cache with stale-while-revalidate policy."
        ),
        tags=["auth", "jwt", "security", "api-gateway", "oauth2"],
        severity=Severity.HIGH,
        resolved_at=datetime(2026, 10, 28, 17, 45, tzinfo=timezone.utc),
    ),
    IncidentInput(
        title="Worker Out Of Memory (OOMKilled) in Image Processing Pipeline",
        error_log=(
            "State: Waiting -> CrashLoopBackOff\n"
            "Last State: Terminated with Exit Code 137 (OOMKilled)\n"
            "System kernel: Memory cgroup out of memory: Killed process 4118 (python worker.py) "
            "total-vm:4194304kB, anon-rss:2097152kB, file-rss:0kB"
        ),
        root_cause=(
            "Pillow library decompressed high-resolution user-uploaded PNG files (decompression bombs) "
            "directly into memory without checking max pixel limits, exceeding Kubernetes container memory limit of 2Gi."
        ),
        resolution=(
            "1. Enforced Image.MAX_IMAGE_PIXELS = 50_000_000 in Pillow before opening images.\n"
            "2. Added upload gateway header validation rejecting image files exceeding 15MB before hitting the processing queue.\n"
            "3. Increased worker container memory limit to 3Gi with horizontal pod autoscaler (HPA) based on memory utilization."
        ),
        tags=["kubernetes", "docker", "oom", "python", "pillow"],
        severity=Severity.MEDIUM,
        resolved_at=datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc),
    ),
    IncidentInput(
        title="Stripe Webhook Delivery Failure due to Database Transaction Deadlock",
        error_log=(
            "stripe.error.APIConnectionError: Request to Stripe API failed.\n"
            "psycopg.errors.DeadlockDetected: deadlock detected\n"
            "DETAIL: Process 14201 waits for ShareLock on transaction 89201; blocked by process 14205.\n"
            "Process 14205 waits for ExclusiveLock on tuple (42, 18) of relation 'orders'; blocked by process 14201.\n"
            "HTTP 500: Webhook delivery failed after 3 automatic retries."
        ),
        root_cause=(
            "Stripe webhook retries were processed concurrently without idempotency locks. "
            "Simultaneous threads updated the order status and ledger entries in reverse sequence, "
            "creating circular row lock dependencies in PostgreSQL."
        ),
        resolution=(
            "1. Implemented Redis distributed lock (Redlock) keyed by stripe_event_id with 15s TTL.\n"
            "2. Added an idempotency_events table to skip duplicate webhook processing immediately.\n"
            "3. Enforced consistent row-locking order (always lock 'orders' row before inserting into 'ledger_entries')."
        ),
        tags=["payments", "stripe", "deadlock", "postgresql", "webhooks"],
        severity=Severity.HIGH,
        resolved_at=datetime(2026, 11, 5, 16, 20, tzinfo=timezone.utc),
    ),
]


def seed(dry_run: bool = False) -> None:
    """Insert the sample incidents into the database."""
    print("=" * 70)
    print("🧠 Incident Memory Agent — Database Seed Script")
    print("=" * 70)

    if dry_run:
        print("\n🔍 DRY-RUN MODE: Displaying 5 incidents to be seeded (no DB or API calls):\n")
        for i, inc in enumerate(SAMPLE_INCIDENTS, 1):
            print(f"[{i}/5] [{inc.severity.value.upper()}] {inc.title}")
            print(f"      Tags: {', '.join(inc.tags)}")
            print(f"      Root cause: {inc.root_cause[:80]}...")
            print(f"      Resolution: {inc.resolution[:80]}...")
            print()
        print("Dry-run complete. Run without --dry-run to insert into Supabase.")
        return

    # Check that environment configuration is available
    try:
        from config.settings import DATABASE_URL, GEMINI_API_KEY
    except EnvironmentError as exc:
        print("\n❌ Environment configuration error:")
        print(f"   {exc}")
        print("\n💡 Please ensure .env exists with GEMINI_API_KEY and DATABASE_URL.")
        sys.exit(1)

    from services.incident_service import list_incidents, store_incident

    # Check existing incidents to avoid duplicate insertions
    print("\nConnecting to database and checking existing records...")
    try:
        existing = list_incidents(limit=200)
        existing_titles = {inc.title for inc in existing}
        print(f"Found {len(existing)} existing incident(s) in database.\n")
    except Exception as exc:
        print(f"❌ Failed to query database: {exc}")
        print("Please check DATABASE_URL and ensure db/schema.sql has been executed in Supabase.")
        sys.exit(1)

    # Insert sample incidents
    inserted_count = 0
    skipped_count = 0

    for i, inc in enumerate(SAMPLE_INCIDENTS, 1):
        if inc.title in existing_titles:
            print(f"[{i}/5] ⏩ Skipped (already exists): '{inc.title}'")
            skipped_count += 1
            continue

        print(f"[{i}/5] 🔄 Embedding & Storing: '{inc.title}'...")
        try:
            stored = store_incident(inc)
            print(f"      ✅ Stored with ID: {stored.id}")
            inserted_count += 1
        except Exception as exc:
            print(f"      ❌ Failed to store: {exc}")

    print("\n" + "=" * 70)
    print(f"🎉 Seeding complete! Inserted: {inserted_count}, Skipped: {skipped_count}, Total in DB: {len(existing) + inserted_count}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed database with sample production incidents.")
    parser.add_argument("--dry-run", action="store_true", help="Print incidents without inserting.")
    args = parser.parse_args()
    seed(dry_run=args.dry_run)
