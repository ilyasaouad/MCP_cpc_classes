"""
phase5b_labeling/labeler.py — Phase 5B: Professional Labeling & Justification.

The final output phase. Uses the LLM to:
  1. Assign each CPC code a role: CORE / SUPPORT / CONTEXT / LEGAL_COVERAGE
  2. Write a 3–4 sentence "Technical Reasoning Graph" justification
  3. Produce a confidence label (HIGH / MEDIUM / LOW) with human-readable explanation

Input: Phase 4B primary CPC, Tri-Pillar breakdown, Phase 5A consistency result.
Output: The dict consumed directly by the API response and the Streamlit UI.

Old equivalent: CPCRoleLabeling + run_phase8()
               in patent_cpc_fastapi/cpc_role_labeling.py + pipeline/phase8_runner.py

run_phase8 returns a 2-tuple: (role_labeling_result, formatted_report)
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
      llm_max_tokens (int) token budget for the justification call
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.pipeline.phase8_runner import run_phase8
            from cpc_classification.search_cpc import CPCClassifier
        except ImportError as exc:
            state.record_error("5b", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        phase4b = state.phase4b
        phase5a = state.phase5a
        phase3b_candidates: List[Dict[str, Any]] = state.phase3b.get("candidates", [])
        tcr_result = state.phase1c.get("tcr_details", {})

        if not phase4b:
            state.record_warning("5b", "Phase 4B result empty — no final label")
            return {}

        # Phase 8 signature:
        # run_phase8(classifier, phase1, phase5_result, ranked, enhanced_candidates,
        #            consistency_result, tcr_result, phase36_result, premier_data)
        try:
            class _FakeCls:
                llm = self.llm
                # Borrow instance method + static helper from CPCClassifier
                _build_formatted_report = CPCClassifier._build_formatted_report
                _shorten = staticmethod(CPCClassifier._shorten)

            consistency_result = dict(phase5a.get("consistency_check", {}))
            consistency_result["recommended_primary"] = phase5a.get("recommended_primary", "")
            consistency_result["recommended_secondary"] = phase5a.get("adjustments", [])

            raw = run_phase8(
                _FakeCls(),
                phase1,
                phase4b,                                               # phase5_result
                phase3b_candidates,                                    # ranked
                phase5a.get("validated_candidates", phase3b_candidates),
                consistency_result,
                tcr_result,
                phase36_result=state.phase3b.get("validation_details", {}),
                premier_data=phase5a.get("premier_data", phase4b),
            )
        except Exception as exc:
            state.record_error("5b", str(exc))
            return {}

        # run_phase8 returns (role_labeling_result, formatted_report)
        role_labeling_result, formatted_report = raw

        core = role_labeling_result.get("layer1_core", [])
        primary_label = core[0].get("symbol", "") if core else ""
        confidence = "HIGH" if core else "LOW"

        logger.info("Phase 5B: primary=%s confidence=%s", primary_label, confidence)

        return {
            "primary_label": primary_label,
            "confidence": confidence,
            "formatted_report": formatted_report,
            "role_labeling": role_labeling_result,
            "layer1_core": role_labeling_result.get("layer1_core", []),
            "layer2_support": role_labeling_result.get("layer2_support", []),
            "layer2_context": role_labeling_result.get("layer2_context", []),
            "layer3_coverage": role_labeling_result.get("layer3_coverage", []),
            "all_codes": role_labeling_result.get("all_codes", []),
            "role_summary": role_labeling_result.get("role_summary", ""),
            "detailed_report": role_labeling_result.get("detailed_report", ""),
            "reasoning_graph": role_labeling_result.get("reasoning_graph", ""),
            "executive_summary": role_labeling_result.get("phase85_executive_summary", ""),
            "code_reasoning": role_labeling_result.get("phase85_code_reasoning", []),
        }
