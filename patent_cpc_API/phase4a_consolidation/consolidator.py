"""
phase4a_consolidation/consolidator.py â€” Phase 4A: Hypothesis Consolidation.

Clusters the Phase 3B candidates into up to `max_hypotheses` coherent groups.
Each hypothesis represents a plausible CPC classification family with:
  - A primary family (4-char prefix)
  - Supporting candidates within that family
  - A coherence score (how tightly the candidates agree)
  - A support_weight (fraction of total candidate mass in this cluster)

When support_weight > 0.8 and coherence < 0.6, this is "sub-branch spread"
(the cluster spans multiple sub-branches of one family) â€” NOT hallucination.

Old equivalent: CPCHypothesisConsolidator in patent_cpc_fastapi/cpc_hypothesis_consolidation.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase4AConsolidator(BasePhase):
    """
    Family-based hypothesis clustering.

    Config keys used:
      max_hypotheses      (int)   maximum clusters to return
      min_support_weight  (float) cluster must hold >= this fraction
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_hypothesis_consolidation import (
                CPCHypothesisConsolidator,
            )
        except ImportError as exc:
            state.record_error("4a", f"Import failed: {exc}")
            return {}

        candidates: List[Dict[str, Any]] = state.phase3b.get("candidates", [])
        phase1 = state.phase1a

        max_hyp = self.cfg.get("max_hypotheses", 3)

        # Build term_importance from Phase 1A terms — required by consolidate()
        term_importance: Dict[str, float] = {}
        for t in phase1.get("terms", phase1.get("essential_terms", [])):
            if isinstance(t, dict):
                term = t.get("term", "").lower()
                imp = t.get("importance", 5)
                if term:
                    term_importance[term] = max(term_importance.get(term, 0), float(imp))
            else:
                term = str(t).lower()
                if term:
                    term_importance[term] = 5.0

        result: Dict[str, Any] = {}
        try:
            consolidator = CPCHypothesisConsolidator(max_hypotheses=max_hyp)
            result = consolidator.consolidate(candidates, term_importance)
        except Exception as exc:
            state.record_error("4a", str(exc))
            return {}

        # consolidate() returns keys prefixed with "phase4_"
        hypotheses: List[Dict[str, Any]] = result.get("phase4_hypotheses", [])
        logger.info("Phase 4A: %d hypotheses consolidated", len(hypotheses))
        return {
            "hypotheses": hypotheses,
            "primary_hypothesis": hypotheses[0] if hypotheses else {},
            "consolidation_details": result,
        }
