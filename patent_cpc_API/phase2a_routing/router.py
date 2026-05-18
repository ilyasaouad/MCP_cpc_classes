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
        emb_w = self.cfg.get("embedding_weight", 0.45)
        kg_w = self.cfg.get("kg_weight", 0.35)
        anc_w = self.cfg.get("anchor_weight", 0.20)
        model = self.cfg.get("embedding_model", "all-mpnet-base-v2")

        try:
            class _FakeCls:
                kg = self.kg
                xml_parser = self.xml_parser
                llm = self.llm

            router = Phase2AV2Router(
                kg=self.kg,
                xml_parser=self.xml_parser,
                embedding_weight=emb_w,
                kg_weight=kg_w,
                anchor_weight=anc_w,
                top_k=top_k,
            )
            result = router.route(phase1, phase1b, tcr_result)
        except Exception as exc:
            state.record_error("2a", str(exc))
            return {}

        family_scores = result.get("family_scores", {})
        logger.info(
            "Phase 2A: top families=%s",
            sorted(family_scores, key=family_scores.get, reverse=True)[:top_k],
        )

        return {
            "family_scores": family_scores,
            "top_families": result.get("top_families", []),
            "layer_result": result.get("layer_result", {}),
            "domain_confidence": result.get("domain_confidence", 0.0),
            "routing_details": result,
        }
