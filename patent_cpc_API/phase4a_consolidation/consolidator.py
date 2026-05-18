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
        phase3a_result = state.phase3a.get("constraint_details", {})

        max_hyp = self.cfg.get("max_hypotheses", 3)
        min_sup = self.cfg.get("min_support_weight", 0.10)

        result: Dict[str, Any] = {}
        try:
            consolidator = CPCHypothesisConsolidator()
            result = consolidator.consolidate(
                candidates,
                phase1,
                phase3a_result,
                max_hypotheses=max_hyp,
                min_support_weight=min_sup,
            )
        except Exception as exc:
            state.record_error("4a", str(exc))
            return {}

        hypotheses: List[Dict[str, Any]] = result.get("hypotheses", [])
        logger.info("Phase 4A: %d hypotheses consolidated", len(hypotheses))
        return {
            "hypotheses": hypotheses,
            "primary_hypothesis": hypotheses[0] if hypotheses else {},
            "consolidation_details": result,
        }
