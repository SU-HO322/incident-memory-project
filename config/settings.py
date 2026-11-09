"""
config/settings.py
------------------
Central configuration loader.
Reads all environment variables from .env and exposes them as typed constants.
Any module that needs a secret imports from here — never from os.environ directly.
"""

import os
from dotenv import load_dotenv

# Load .env file into environment variables (safe no-op if already loaded)
load_dotenv()


def _require(key: str) -> str:
    """
    Fetch a required environment variable.
    Raises a clear error if it's missing so the developer knows exactly what to fix.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"\n\n❌  Missing required environment variable: {key}\n"
            f"    → Copy .env.example to .env and fill in your real values.\n"
        )
    return value


# --- Gemini ---
# SDK: `import google.genai as genai` (google-genai>=3.0.0, replaces deprecated google-generativeai)
# Docs: https://googleapis.github.io/python-genai/
GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_GENERATION_MODEL: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_DIMENSIONS: int = 768

# --- Supabase ---
SUPABASE_URL: str = _require("SUPABASE_URL")
SUPABASE_KEY: str = _require("SUPABASE_KEY")
DATABASE_URL: str = _require("DATABASE_URL")

# --- Search Tuning ---
TOP_K_RESULTS: int = 5                   # How many similar incidents to retrieve
SIMILARITY_THRESHOLD: float = 0.50       # Minimum cosine similarity to show a result
HIGH_CONFIDENCE_THRESHOLD: float = 0.80  # Above this → "High confidence" badge
MED_CONFIDENCE_THRESHOLD: float = 0.60   # Above this → "Medium confidence" badge

# --- App ---
APP_TITLE: str = "🧠 Incident Memory Agent"
APP_SUBTITLE: str = "AI that remembers how your team fixed it last time."
