"""
phase2a_routing/router.py â€” Phase 2A: CPC Family Routing.

Scores CPC families (4-char prefix) using a blend of:
  - Semantic embedding similarity (all-mpnet-base-v2)
  - Knowledge-graph structural proximity
  - Hard keyword anchors from Phase 1A/1B

Returns family_scores dict {family: weight} whose values are used
to normalise Phase 2C scores (so volume of subgroups doesn't dominate).

Old equivalent: Phase2AV2Router in patent_cpc_fastapi/pipeline/phase_2a_v2/
"""

import logging
from typing import Any, Dict

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase2ARouter(BasePhase):
    """
    Family-level routing using embedding + KG + anchor blend.

    Config keys used:
      top_k_families    (int)   number of families to pass to Phase 2B
      embedding_weight  (float) weight of semantic similarity
      kg_weight         (float) weight of KG proximity score
      anchor_weight     (float) weight of hard anchor signal
      embedding_model   (str)   sentence-transformer model name
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.pipeline.phase_2a_v2 import (
                Phase2AV2Router,
            )
        except ImportError as exc:
            state.record_error("2a", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        phase1b = state.phase1b
        tcr_result = state.phase1c.get("tcr_details", {})

        top_k = self.cfg.get("top_k_families", 5)

        try:
            router = Phase2AV2Router(knowledge_graph=self.kg)
            result = router.route(phase1, phase15_result=tcr_result, top_k=top_k)
        except Exception as exc:
            state.record_error("2a", str(exc))
            return {}

        # route() returns {"families": [{"family": "G10L", "score": 0.91, ...}], ...}
        families = result.get("families", [])
        family_scores = {f["family"]: f["score"] for f in families}
        top_families = result.get("family_names", [f["family"] for f in families])

        logger.info("Phase 2A: top families=%s", top_families[:top_k])

        return {
            "family_scores": family_scores,
            "top_families": top_families,
            "layer_result": {},
            "domain_confidence": families[0]["score"] if families else 0.0,
            "routing_details": result,
        }
