"""
api/router.py — FastAPI router for the CPC classification API.

All routes live here. The engine instance is injected via FastAPI dependency.
"""

import concurrent.futures
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .models import ClassifyRequest, ClassifyResponse, HealthResponse, PillarItem

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency: engine singleton ─────────────────────────────────────────────

def _get_engine():
    """Retrieve the engine set at startup via set_engine()."""
    from . import _engine
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(engine=Depends(_get_engine)):
    from . import _kg
    return HealthResponse(
        status="ok",
        kg_loaded=_kg is not None,
        model="patent_cpc_api",
    )


@router.post("/classify", response_model=ClassifyResponse, tags=["classification"])
def classify(
    req: ClassifyRequest,
    debug: bool = Query(False, description="Include per-phase debug data"),
    engine=Depends(_get_engine),
):
    """
    Run the full 13-phase CPC classification pipeline.

    - **query**: Patent abstract or title (required)
    - **claims**: Claims text (recommended — used for Phase 1B audit)
    - **description**: Detailed description (optional — boosts Phase 1A extraction)
    - **debug**: Include raw per-phase data in the response
    """
    _TIMEOUT_S = 270  # hard cap — client timeout is 300 s
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                engine.run,
                query=req.query,
                claims=req.claims,
                description=req.description,
            )
            raw = future.result(timeout=_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out (>270 s)")
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    phase4b = raw.get("phase4b", {})
    phase5b = raw.get("phase5b", {})

    primary_cpc = phase4b.get("primary_cpc", "")
    primary_title = phase4b.get("primary_title", "")
    confidence = phase4b.get("confidence", "LOW")
    score = float(phase4b.get("score", 0.0))
    justification = phase5b.get("executive_summary", phase5b.get("justification", ""))

    # pillars from CPCHypothesisResolver is a dict {pillar1_goal: {...}, ...}
    raw_pillars = phase4b.get("pillars", {})
    pillar_items = raw_pillars.values() if isinstance(raw_pillars, dict) else raw_pillars
    pillars = []
    for p in pillar_items:
        if isinstance(p, dict) and p.get("symbol"):
            pillars.append(
                PillarItem(
                    role=p.get("role", p.get("label", "")),
                    symbol=p.get("symbol", ""),
                    title=p.get("title", ""),
                    family=p.get("symbol", "")[:4],
                )
            )

    # supporting_codes may be strings (from resolver) or dicts (from Phase 5B)
    raw_sc = phase5b.get("all_codes", phase4b.get("supporting_codes", []))
    supporting_codes = []
    for sc in raw_sc:
        if isinstance(sc, dict):
            supporting_codes.append(sc)
        elif isinstance(sc, str) and sc:
            supporting_codes.append({"symbol": sc, "title": "", "role": "SUPPORT"})

    return ClassifyResponse(
        primary_cpc=primary_cpc,
        primary_title=primary_title,
        confidence=confidence,
        score=score,
        pillars=pillars,
        justification=justification,
        supporting_codes=supporting_codes,
        phase_details=raw if debug else None,
        elapsed_ms=raw.get("elapsed_ms", 0),
        warnings=raw.get("warnings", []),
        errors=raw.get("errors", []),
    )
