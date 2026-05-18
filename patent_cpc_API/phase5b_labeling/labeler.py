"""
phase5b_labeling/labeler.py â€” Phase 5B: Professional Labeling & Justification.

The final output phase. Uses the LLM to:
  1. Assign each CPC code a role: CORE / SUPPORT / CONTEXT / LEGAL_COVERAGE
  2. Write a 3â€“4 sentence "Technical Reasoning Graph" justification
  3. Produce a confidence label (HIGH / MEDIUM / LOW) with human-readable explanation

Input: Phase 4B primary CPC, Tri-Pillar breakdown, Phase 5A consistency result.
Output: The dict consumed directly by the API response and the Streamlit UI.

Old equivalent: CPCRoleLabeling + run_phase8()
               in patent_cpc_fastapi/cpc_role_labeling.py + pipeline/phase8_runner.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase5BLabeler(BasePhase):
    """
    LLM-driven role labeling and professional justification.

    Config keys used:
      llm_model      (str) Claude model for justification (e.g. claude-opus-4-7)
      llm_max_tokens (int) token budget for the justification call
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_role_labeling import (
                CPCRoleLabeling,
            )
            from cpc_classification.pipeline.phase8_runner import (
                run_phase8,
            )
        except ImportError as exc:
            state.record_error("5b", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        phase4b = state.phase4b
        phase5a = state.phase5a
        phase3b_candidates: List[Dict[str, Any]] = state.phase3b.get("candidates", [])
        phase3a_result = state.phase3a.get("constraint_details", {})
        tcr_result = state.phase1c.get("tcr_details", {})

        if not phase4b:
            state.record_warning("5b", "Phase 4B result empty â€” no final label")
            return {}

        # Phase 8 signature:
        # run_phase8(classifier, phase1, phase5_result, ranked, enhanced_candidates,
        #            consistency_result, tcr_result, phase36_result, premier_data)
        result: Dict[str, Any] = {}
        try:
            class _FakeCls:
                llm = self.llm

            consistency_result = phase5a.get("consistency_check", {})
            consistency_result["recommended_primary"] = phase5a.get("recommended_primary", "")
            consistency_result["recommended_secondary"] = phase5a.get("adjustments", [])

            result = run_phase8(
                _FakeCls(),
                phase1,
                phase4b,                       # phase5_result (old name)
                phase3b_candidates,            # ranked
                phase5a.get("validated_candidates", phase3b_candidates),
                consistency_result,
                tcr_result,
                phase36_result=state.phase3b.get("validation_details", {}),
                premier_data=phase4b,
            )
        except Exception as exc:
            state.record_error("5b", str(exc))
            return {}

        primary_label = result.get("primary_label", "")
        confidence = result.get("confidence", "LOW")
        logger.info("Phase 5B: primary=%s confidence=%s", primary_label, confidence)
        return result
