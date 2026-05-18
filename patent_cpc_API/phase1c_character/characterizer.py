"""
phase1c_character/characterizer.py â€” Phase 1C: Technical Character Analysis.

Computes the Technology Computation Ratio (TCR): a scalar that captures
how computational vs. physical the invention is. Drives Phase 2A routing
toward software-heavy (G06x) or hardware-heavy (Hxx/Fxx) CPC families.

Also classifies the patent's role (CORE_TECH / SYSTEM / APPLICATION / SUPPORT).

Old equivalent: run_phase15_tcr() in patent_cpc_fastapi/pipeline/phase15_tcr_runner.py
"""

import logging
from typing import Any, Dict

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)

_VALID_ROLES = {"CORE_TECH", "SYSTEM", "APPLICATION", "SUPPORT"}


class Phase1CCharacterizer(BasePhase):
    """
    Deterministic TCR computation + LLM role classification.

    Config keys used:
      tcr_software_threshold (float) TCR > this â†’ software bias
      tcr_hardware_threshold (float) TCR < this â†’ hardware bias
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_role_classifier import (
                CPCRoleClassifier,
            )
            from cpc_classification.technical_weight_analyzer import (
                TechnicalWeightAnalyzer,
            )
        except ImportError as exc:
            state.record_error("1c", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        if not phase1:
            state.record_warning("1c", "Phase 1A result empty â€” skipping TCR")
            return {"tcr_score": 1.0, "tcr_label": "neutral", "role": "SYSTEM"}

        phase15_result: Dict[str, Any] = {}
        try:
            role_clf = CPCRoleClassifier(self.llm)
            phase15_result = role_clf.classify_role(phase1)

            role = phase15_result.get("role", "SYSTEM")
            conf = phase15_result.get("confidence", 0.5)
            if role not in _VALID_ROLES:
                raise ValueError(f"Invalid role: {role}")
            if not (0.0 < conf <= 0.92):
                raise ValueError(f"Confidence out of range: {conf}")
        except Exception as exc:
            state.record_warning("1c", f"Role classification fallback: {exc}")
            phase15_result = {"role": "SYSTEM", "confidence": 0.5}

        tcr_result: Dict[str, Any] = {}
        try:
            analyzer = TechnicalWeightAnalyzer()
            tcr_result = analyzer.analyze(
                phase1,
                phase1_5_result=phase15_result,
                domain_signals=phase1.get("domain_signals", []),
            )
        except Exception as exc:
            state.record_warning("1c", f"TCR analysis fallback: {exc}")
            tcr_result = {"tcr": 1.0, "tcr_bias": 0.0}

        sw_thresh = self.cfg.get("tcr_software_threshold", 0.3)
        hw_thresh = self.cfg.get("tcr_hardware_threshold", -0.3)
        tcr_bias = tcr_result.get("tcr_bias", 0.0)

        if tcr_bias > sw_thresh:
            tcr_label = "software"
        elif tcr_bias < hw_thresh:
            tcr_label = "hardware"
        else:
            tcr_label = "neutral"

        logger.info(
            "Phase 1C: role=%s conf=%.2f  TCR=%.3f bias=%.3f label=%s",
            phase15_result.get("role"),
            phase15_result.get("confidence", 0),
            tcr_result.get("tcr", 1.0),
            tcr_bias,
            tcr_label,
        )

        return {
            "role": phase15_result.get("role", "SYSTEM"),
            "role_confidence": phase15_result.get("confidence", 0.5),
            "tcr_score": tcr_result.get("tcr", 1.0),
            "tcr_bias": tcr_bias,
            "tcr_label": tcr_label,
            "software_bias": tcr_label == "software",
            "hardware_bias": tcr_label == "hardware",
            "tcr_details": tcr_result,
        }
