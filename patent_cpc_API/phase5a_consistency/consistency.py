"""
phase5a_consistency/consistency.py — Phase 5A: Consistency Check.

LLM-driven sanity check: given the Phase 4B primary CPC, the top candidates,
and the Phase 1A semantic terms, asks the model whether the classification
is self-consistent and whether any reranking is warranted.

This is the last LLM call before the final labeling phase.

Old equivalent: run_phase7() in patent_cpc_fastapi/pipeline/phase7_runner.py

run_phase7 returns a 6-tuple:
  (consistency_result, validated_candidates, filtered_out, best_code, premier_data, feedback_info)
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase5AConsistency(BasePhase):
    """
    LLM consistency validation of the Phase 4B primary recommendation.

    No config keys — logic is deterministic given Phase 4B output.
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.pipeline.phase7_runner import run_phase7
            from cpc_classification.search_cpc import CPCClassifier
        except ImportError as exc:
            state.record_error("5a", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        phase4b = state.phase4b
        phase3b_candidates: List[Dict[str, Any]] = state.phase3b.get("candidates", [])
        phase2d_candidates: List[Dict[str, Any]] = state.phase2d.get("candidates", [])
        tcr_result = state.phase1c.get("tcr_details", {})

        if not phase4b:
            state.record_warning("5a", "Phase 4B result empty — skipping consistency check")
            return {"is_consistent": True, "adjustments": []}

        # Phase 7 signature: run_phase7(classifier, phase1, phase5_result, ranked, candidates, phase4_result, tcr_result)
        # phase5_result maps to phase4b in new naming; phase4_result maps to phase4a
        try:
            class _FakeCls:
                llm = self.llm
                # _resolve_premier is @staticmethod on CPCClassifier — borrow it directly
                _resolve_premier = staticmethod(CPCClassifier._resolve_premier)

            raw = run_phase7(
                _FakeCls(),
                phase1,
                phase4b,                   # phase5_result (old name)
                phase3b_candidates,        # ranked
                phase2d_candidates,        # candidates
                state.phase4a,             # phase4_result
                tcr_result,
            )
        except Exception as exc:
            state.record_error("5a", str(exc))
            return {"is_consistent": True, "adjustments": []}

        # run_phase7 returns a 6-tuple:
        # (consistency_result, validated_candidates, filtered_out, best_code, premier_data, feedback_info)
        (
            consistency_result,
            validated_candidates,
            _filtered_out,
            best_code,
            premier_data,
            feedback_info,
        ) = raw

        is_consistent = consistency_result.get("coherent", True)
        logger.info("Phase 5A: is_consistent=%s", is_consistent)

        return {
            "is_consistent": is_consistent,
            "consistency_check": consistency_result,
            "recommended_primary": consistency_result.get("recommended_primary", ""),
            "adjustments": consistency_result.get("recommended_secondary", []),
            "validated_candidates": validated_candidates,
            "best_code": best_code,
            "premier_data": premier_data,
            "feedback_info": feedback_info,
        }
