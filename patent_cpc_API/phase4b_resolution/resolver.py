"""
phase4b_resolution/resolver.py — Phase 4B: Hypothesis Resolution.

Scores each Phase 4A hypothesis on three dimensions and picks the winner:
  - functional_alignment: how well the CPC title matches core_function
  - technical_coverage:   how many Phase 1A terms appear in CPC titles
  - specificity_match:    hypothesis coherence from Phase 4A

Tri-Pillar back-scan finds the best subgroup champion for each role.

Old equivalent: CPCHypothesisResolver in patent_cpc_fastapi/cpc_hypothesis_resolver.py

resolve() signature: (phase4_result, phase1_data, all_raw_candidates=None)
  phase4_result  = state.phase4a["consolidation_details"]  (has "phase4_hypotheses" key)
  Returns:
    {
      "primary":   {family, final_score, confidence, supporting_codes, ...},
      "secondary": {same} or absent,
      "decision_logic": {...},
      "pillars":   {pillar1_goal, pillar2_method, pillar3_context},
    }
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase4BResolver(BasePhase):
    """
    Multi-criteria hypothesis scoring and Tri-Pillar resolution.

    No constructor config keys — resolver uses its own internal weights.
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_hypothesis_resolver import (
                CPCHypothesisResolver,
            )
        except ImportError as exc:
            state.record_error("4b", f"Import failed: {exc}")
            return {}

        # resolve() needs the full Phase 4A consolidation dict (has "phase4_hypotheses")
        phase4_result: Dict[str, Any] = state.phase4a.get("consolidation_details", {})
        phase1 = state.phase1a
        all_raw_candidates: List[Dict[str, Any]] = state.phase2d.get(
            "all_raw_candidates", []
        )

        if not phase4_result.get("phase4_hypotheses"):
            state.record_warning("4b", "No hypotheses from Phase 4A")
            return {}

        result: Dict[str, Any] = {}
        try:
            resolver = CPCHypothesisResolver(llm_client=self.llm)
            result = resolver.resolve(phase4_result, phase1, all_raw_candidates)
        except Exception as exc:
            state.record_error("4b", str(exc))
            return {}

        primary = result.get("primary", {})
        family = primary.get("family", "")
        supporting = primary.get("supporting_codes", [])

        # Derive a concrete primary symbol: first supporting code, else family itself
        primary_cpc = supporting[0] if supporting else family
        confidence = primary.get("confidence", "LOW")
        score = float(primary.get("final_score", 0.0))

        # Try to look up the title for the primary symbol
        primary_title = ""
        for c in all_raw_candidates:
            if c.get("symbol", "") == primary_cpc:
                primary_title = c.get("title", "")
                break

        logger.info(
            "Phase 4B: primary=%s score=%.3f confidence=%s",
            primary_cpc, score, confidence,
        )

        # Merge resolver result with flat aliases the API router reads
        return {
            **result,
            "primary_cpc": primary_cpc,
            "primary_title": primary_title,
            "confidence": confidence,
            "score": score,
            "supporting_codes": supporting,
        }
