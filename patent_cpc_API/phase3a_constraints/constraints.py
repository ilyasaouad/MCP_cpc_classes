"""
phase3a_constraints/constraints.py â€” Phase 3A: Decision-Tree Constraints.

Applies deterministic, rule-based adjustments to Phase 2D candidates:
  - Domain dominance boosts (e.g. G10L gets DOMAIN_ANCHOR boost)
  - Invalid-class penalties per domain (e.g. G06T penalised in speech domain)
  - Functional boosting (e.g. "quantization" boosts G06N3/063)
  - Hierarchy priority (subgroups before classes)
  - Non-allocatable pre-filter (second guard after Phase 2B)

Always runs â€” do NOT skip even for single-family patents.
For future optimisation: if top-5 share same 4-char family AND
domain_confidence >= 0.85, the constraint pass is a no-op.

Old equivalent: CPCDecisionTreeConstraint in patent_cpc_fastapi/cpc_decision_tree.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..shared.cpc_filters import filter_allocatable
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase3AConstraints(BasePhase):
    """
    Rule-based constraint and boosting layer.

    Config keys used:
      domain_confidence_threshold (float) minimum to apply domain dominance rules
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_decision_tree import (
                CPCDecisionTreeConstraint,
            )
        except ImportError as exc:
            state.record_error("3a", f"Import failed: {exc}")
            return {}

        candidates: List[Dict[str, Any]] = state.phase2d.get("candidates", [])
        phase1 = state.phase1a
        layer_result = state.phase2a.get("layer_result", {})
        tcr_result = state.phase1c.get("tcr_details", {})

        # Sort top-20 for constraint evaluation
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

        # Pre-filter non-allocatable (belt-and-suspenders)
        before = len(ranked)
        ranked = filter_allocatable(ranked)
        removed = before - len(ranked)
        if removed:
            logger.info("Phase 3A: removed %d non-allocatable class nodes", removed)

        result: Dict[str, Any] = {}
        try:
            dt = CPCDecisionTreeConstraint()
            result = dt.apply_constraints(ranked, phase1, layer_result, tcr_result)
            ranked = result.get("phase35_candidates", ranked)
        except Exception as exc:
            state.record_error("3a", str(exc))

        logger.info(
            "Phase 3A: %d adjustments | domain=%s (conf=%.2f)",
            result.get("phase35_adjustments", 0),
            result.get("phase35_domain", "unknown"),
            result.get("phase35_domain_confidence", 0),
        )
        return {
            "candidates": ranked,
            "adjustments": result.get("phase35_adjustments", 0),
            "domain": result.get("phase35_domain", "unknown"),
            "domain_confidence": result.get("phase35_domain_confidence", 0.0),
            "layer_mode": result.get("phase35_layer_mode", False),
            "constraint_details": result,
        }
