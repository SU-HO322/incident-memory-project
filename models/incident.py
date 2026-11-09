"""
models/incident.py
------------------
Pydantic data models — the "shape" of data flowing through the system.

Why Pydantic?
  - Validates data types automatically (catches bad DB rows, bad API responses early)
  - Gives IDE autocompletion everywhere
  - Makes the codebase self-documenting
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Allowed severity levels for an incident."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence tier shown in the UI."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_MATCH = "no_match"


# ---------------------------------------------------------------------------
# Tag normalization helper — shared across models and DB adapters
# ---------------------------------------------------------------------------

def normalize_tags(v: Any) -> list[str]:
    """
    Safely convert any tag representation into a clean list of strings:
      - PostgreSQL array literal: '{vpn,network,remote-access}' -> ['vpn', 'network', 'remote-access']
      - Comma-separated string:   'vpn, network, remote-access' -> ['vpn', 'network', 'remote-access']
      - JSON-like list string:    '["vpn", "network"]'          -> ['vpn', 'network']
      - Python list/tuple/set:    ['vpn', 'network']            -> ['vpn', 'network']
      - Empty or None:            '{}', '[]', '', None          -> []
    """
    if v is None:
        return []
    if isinstance(v, str):
        v_str = v.strip()
        if not v_str or v_str in ("{}", "[]"):
            return []
        if (v_str.startswith("{") and v_str.endswith("}")) or (v_str.startswith("[") and v_str.endswith("]")):
            v_str = v_str[1:-1]
        return [t.strip().strip("'\"") for t in v_str.split(",") if t.strip().strip("'\"")]
    if isinstance(v, (list, tuple, set)):
        items: list[str] = []
        for item in v:
            if item is None:
                continue
            s = str(item).strip().strip("'\"")
            if s:
                items.append(s)
        return items
    s_single = str(v).strip()
    return [s_single] if s_single else []


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class Incident(BaseModel):
    """
    Represents a single resolved production incident stored in the database.
    Maps 1-to-1 with a row in the `incidents` table.
    """
    id: Optional[UUID] = None
    title: str = Field(..., min_length=3, max_length=200, description="Short incident name")
    error_log: str = Field(..., min_length=10, description="Raw error log or stack trace")
    root_cause: str = Field(..., min_length=10, description="What caused the incident")
    resolution: str = Field(..., min_length=10, description="How the team fixed it")
    tags: list[str] = Field(default_factory=list, description="e.g. ['database', 'timeout']")
    severity: Severity = Severity.MEDIUM
    resolved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    # embedding is NOT included here — it lives only in the DB, never sent to UI
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity vs the query (populated during search, not stored)",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: Any) -> list[str]:
        return normalize_tags(v)

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(cls, v: Any) -> Severity:
        if isinstance(v, Severity):
            return v
        if isinstance(v, str):
            v_clean = v.strip().lower()
            try:
                return Severity(v_clean)
            except ValueError:
                return Severity.MEDIUM
        return Severity.MEDIUM

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Derive a human-readable confidence tier from the raw similarity score."""
        from config.settings import HIGH_CONFIDENCE_THRESHOLD, MED_CONFIDENCE_THRESHOLD
        if self.similarity_score is None:
            return ConfidenceLevel.NO_MATCH
        if self.similarity_score >= HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if self.similarity_score >= MED_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Input / Output transfer objects
# ---------------------------------------------------------------------------

class IncidentInput(BaseModel):
    """
    What the user submits when logging a new incident.
    Intentionally simpler than Incident — no id, timestamps, or score.
    """
    title: str = Field(..., min_length=3, max_length=200)
    error_log: str = Field(..., min_length=10)
    root_cause: str = Field(..., min_length=10)
    resolution: str = Field(..., min_length=10)
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    resolved_at: Optional[datetime] = None

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: Any) -> list[str]:
        return normalize_tags(v)

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(cls, v: Any) -> Severity:
        if isinstance(v, Severity):
            return v
        if isinstance(v, str):
            v_clean = v.strip().lower()
            try:
                return Severity(v_clean)
            except ValueError:
                return Severity.MEDIUM
        return Severity.MEDIUM


class SearchResult(BaseModel):
    """
    The full response returned to the UI after a similarity search.
    Contains both the raw matched incidents and the AI-generated recommendation.
    """
    query: str = Field(..., description="The original error text the user searched for")
    matches: list[Incident] = Field(default_factory=list, description="Top-K similar incidents")
    ai_recommendation: str = Field(
        default="",
        description="Gemini-generated recommendation grounded in the matched incidents",
    )
    confidence_level: ConfidenceLevel = ConfidenceLevel.NO_MATCH
    top_similarity_score: Optional[float] = None
    has_matches: bool = False

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        """Auto-populate derived fields after construction."""
        if self.matches:
            self.has_matches = True
            self.top_similarity_score = self.matches[0].similarity_score
            self.confidence_level = self.matches[0].confidence_level
