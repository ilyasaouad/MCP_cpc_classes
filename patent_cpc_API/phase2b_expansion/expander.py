"""
phase2b_expansion/expander.py â€” Phase 2B: Subgroup Expansion.

For each top family from Phase 2A, expands into candidate subgroups via:
  1. KG hierarchy traversal (BFS from the family root)
  2. XML title pre-loading (guarantees unique, real CPC descriptions)

Non-allocatable nodes (no "/" in symbol, cross-ref codes) are filtered here.

Old equivalent: Phase2BExpander in patent_cpc_fastapi/pipeline/phase2b_expander.py
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
      max_subgroups       (int)   hard cap on subgroups per family
      min_expansion_count (int)   minimum subgroups before fallback
      min_subclass_score  (float) drop subgroups below this base score
      max_traversal_depth (int)   BFS depth in KG hierarchy
      proportional_cap    (bool)  cap slots proportionally to Phase 2A weight
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
        min_exp = self.cfg.get("min_expansion_count", 5)
        min_score = self.cfg.get("min_subclass_score", 0.30)
        max_depth = self.cfg.get("max_traversal_depth", 3)
        prop_cap = self.cfg.get("proportional_cap", True)

        try:
            expander = _Expander(
                kg=self.kg,
                xml_parser=self.xml_parser,
                max_subgroups=max_sg,
                min_expansion_count=min_exp,
                min_subclass_score=min_score,
                max_traversal_depth=max_depth,
                proportional_cap=prop_cap,
            )
            candidates: List[Dict[str, Any]] = expander.expand(
                top_families, phase1, family_scores
            )
        except Exception as exc:
            state.record_error("2b", str(exc))
            return {"expanded_candidates": []}

        # Filter non-allocatable immediately after expansion
        before = len(candidates)
        candidates = filter_allocatable(candidates)
        removed = before - len(candidates)
        if removed:
            logger.info("Phase 2B: removed %d non-allocatable nodes", removed)

        logger.info("Phase 2B: %d subgroup candidates across %d families", len(candidates), len(top_families))
        return {"expanded_candidates": candidates}
