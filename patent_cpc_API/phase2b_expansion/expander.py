"""
phase2b_expansion/expander.py — Phase 2B: Subgroup Expansion.

For each top family from Phase 2A, expands into candidate subgroups via:
  1. KG hierarchy traversal (BFS from the family root)
  2. XML title pre-loading (guarantees unique, real CPC descriptions)

Non-allocatable nodes (no "/" in symbol, cross-ref codes) are filtered here.

Old equivalent: Phase2BExpander in patent_cpc_fastapi/pipeline/phase2b_expander.py

expand() signature: (families, family_scores=None, phase1=None, max_subgroups=500)
expand() returns:
  {
    "expanded_cpcs": [...],          # bare symbol list
    "expanded_details": [...],       # [{symbol, title, score, family}, ...]
    "family_expansions": [...],
    "source": str,
    "fallback_used": bool,
    "family_counts": {family: int},
    "expansion_balance": {family: int},
    "pruned_count": int,
  }
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..shared.cpc_filters import filter_allocatable
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase2BExpander(BasePhase):
    """
    KG + XML subgroup expansion with non-allocatable pre-filtering.

    Config keys used:
      max_subgroups  (int)  hard cap on total subgroups returned
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.pipeline.phase2b_expander import (
                Phase2BExpander as _Expander,
            )
        except ImportError as exc:
            state.record_error("2b", f"Import failed: {exc}")
            return {"expanded_candidates": []}

        phase1 = state.phase1a
        phase2a = state.phase2a
        family_scores: Dict[str, float] = phase2a.get("family_scores", {})
        top_families: List[str] = phase2a.get("top_families", list(family_scores.keys()))

        if not top_families:
            state.record_warning("2b", "No top families from Phase 2A")
            return {"expanded_candidates": []}

        max_sg = self.cfg.get("max_subgroups", 500)

        try:
            expander = _Expander(
                knowledge_graph=self.kg,
                xml_parser=self.xml_parser,
            )
            result = expander.expand(
                top_families,
                family_scores=family_scores,
                phase1=phase1,
                max_subgroups=max_sg,
            )
        except Exception as exc:
            state.record_error("2b", str(exc))
            return {"expanded_candidates": []}

        # expanded_details has the full {symbol, title, score, family} dicts
        candidates: List[Dict[str, Any]] = result.get("expanded_details", [])

        # Filter non-allocatable immediately after expansion
        before = len(candidates)
        candidates = filter_allocatable(candidates)
        removed = before - len(candidates)
        if removed:
            logger.info("Phase 2B: removed %d non-allocatable nodes", removed)

        logger.info(
            "Phase 2B: %d subgroup candidates across %d families (source=%s)",
            len(candidates),
            len(top_families),
            result.get("source", "?"),
        )
        return {
            "expanded_candidates": candidates,
            "expansion_details": result,
        }
