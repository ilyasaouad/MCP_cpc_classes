"""
phase4b_resolution/resolver.py â€” Phase 4B: Hypothesis Resolution.

Scores each Phase 4A hypothesis on three dimensions and picks the winner:
  - phase4_weight        (0.5) â€” cluster coherence and support from Phase 4A
  - functional_alignment (0.3) â€” how well the CPC title matches core_function
  - technical_coverage   (0.2) â€” how many Phase 1A terms appear in CPC titles

Final score = 0.5Ã—phase4 + 0.3Ã—functional_alignment + 0.2Ã—technical_coverage

Also resolves the Tri-Pillar output:
  - Pillar 1 (primary_goal):    best subgroup within the winning family
  - Pillar 2 (ai_methodology):  best G06N subgroup from the pool
  - Pillar 3 (domain_context):  best remaining non-visual subgroup

Old equivalent: CPCHypothesisResolver in patent_cpc_fastapi/cpc_hypothesis_resolver.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase4BResolver(BasePhase):
    """
    Multi-criteria hypothesis scoring and pillar resolution.

    Config keys used:
      phase4_weight               (float) weight for cluster evidence
      functional_alignment_weight (float) weight for CPC title match
      technical_coverage_weight   (float) weight for term coverage
      high_confidence_threshold   (float) score above this â†’ HIGH confidence
      secondary_gap_threshold     (float) accept secondary if gap < this
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_hypothesis_resolver import (
                CPCHypothesisResolver,
            )
        except ImportError as exc:
            state.record_error("4b", f"Import failed: {exc}")
            return {}

        hypotheses: List[Dict[str, Any]] = state.phase4a.get("hypotheses", [])
        phase1 = state.phase1a
        phase2a = state.phase2a
        all_raw_candidates: List[Dict[str, Any]] = state.phase2d.get(
            "all_raw_candidates", []
        )

        if not hypotheses:
            state.record_warning("4b", "No hypotheses from Phase 4A")
            return {}

        # Build symbolâ†’title lookup from full raw pool
        symbol_to_title: Dict[str, str] = {
            c.get("symbol", ""): c.get("title", "")
            for c in all_raw_candidates
            if c.get("symbol") and c.get("title")
        }

        p4_w = self.cfg.get("phase4_weight", 0.5)
        fa_w = self.cfg.get("functional_alignment_weight", 0.3)
        tc_w = self.cfg.get("technical_coverage_weight", 0.2)
        hi_thresh = self.cfg.get("high_confidence_threshold", 0.75)
        gap_thresh = self.cfg.get("secondary_gap_threshold", 0.25)

        result: Dict[str, Any] = {}
        try:
            resolver = CPCHypothesisResolver(
                phase4_weight=p4_w,
                functional_alignment_weight=fa_w,
                technical_coverage_weight=tc_w,
                high_confidence_threshold=hi_thresh,
                secondary_gap_threshold=gap_thresh,
            )
            result = resolver.resolve(
                hypotheses,
                phase1,
                phase2a,
                all_raw_candidates,
                symbol_to_title=symbol_to_title,
            )
        except Exception as exc:
            state.record_error("4b", str(exc))
            return {}

        primary = result.get("primary_cpc", "")
        score = result.get("score", 0.0)
        conf = result.get("confidence", "LOW")

        logger.info(
            "Phase 4B: primary=%s score=%.3f confidence=%s",
            primary, score, conf,
        )
        return result
