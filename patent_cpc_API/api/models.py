"""
api/models.py — Pydantic request/response schemas for the CPC classification API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator


class ClassifyRequest(BaseModel):
    """Incoming classification request."""

    query: str
    claims: Optional[str] = None
    description: Optional[str] = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class PillarItem(BaseModel):
    """A single pillar in the Tri-Pillar output."""

    role: str          # primary_goal / ai_methodology / domain_context
    symbol: str
    title: str
    family: str


class ClassifyResponse(BaseModel):
    """Full pipeline result returned to the caller."""

    # Core output
    primary_cpc: str
    primary_title: str
    confidence: str                     # HIGH / MEDIUM / LOW
    score: float

    # Tri-Pillar breakdown
    pillars: List[PillarItem] = []

    # Professional justification (from Phase 5B LLM)
    justification: str = ""

    # Supporting codes with roles
    supporting_codes: List[Dict[str, Any]] = []

    # Per-phase debug data (included when ?debug=true)
    phase_details: Optional[Dict[str, Any]] = None

    # Pipeline metadata
    elapsed_ms: int = 0
    warnings: List[str] = []
    errors: List[str] = []


class HealthResponse(BaseModel):
    status: str
    kg_loaded: bool
    model: str
