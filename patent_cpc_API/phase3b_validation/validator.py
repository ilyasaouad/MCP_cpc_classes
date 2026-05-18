"""
phase3b_validation/validator.py â€” Phase 3B: Cross-Domain Validation.

Validates that the top candidates after Phase 3A are internally consistent:
  - DOMAIN_ANCHOR_CONFIRMED: primary family verified against claim signals
  - ANTI_COLLAPSE_BOOST: secondary candidates boosted when domain context present
  - ANTI_COLLAPSE_PENALTY: secondary candidates penalised without domain context

"Anchor" = a CPC family that the Phase 1B claim-forensic audit confirmed
is explicitly evidenced in the claims text. Anchors receive score boosts;
families not evidenced in claims receive penalties.

Old equivalent: CrossDomainValidator in patent_cpc_fastapi/cpc_cross_domain_validator.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase3BValidator(BasePhase):
    """
    Cross-domain consistency validation with anchor boosts/penalties.

    Config keys used:
      domain_anchor_boost    (float) multiplier when domain signals confirmed
      domain_anchor_penalty  (float) multiplier when signals below threshold
      anti_collapse_boost    (float) multiplier when domain context present
      anti_collapse_penalty  (float) multiplier when domain context absent
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.cpc_cross_domain_validator import (
                CrossDomainValidator,
            )
        except ImportError as exc:
            state.record_error("3b", f"Import failed: {exc}")
            return {}

        candidates: List[Dict[str, Any]] = state.phase3a.get("candidates", [])
        phase1 = state.phase1a
        phase3a_result = state.phase3a.get("constraint_details", {})

        result: Dict[str, Any] = {}
        try:
            validator = CrossDomainValidator(
                domain_anchor_boost=self.cfg.get("domain_anchor_boost", 1.2),
                domain_anchor_penalty=self.cfg.get("domain_anchor_penalty", 0.5),
                anti_collapse_boost=self.cfg.get("anti_collapse_boost", 2.0),
                anti_collapse_penalty=self.cfg.get("anti_collapse_penalty", 0.05),
            )
            result = validator.validate(candidates, phase1, phase3a_result)
            candidates = result.get("phase36_candidates", candidates)
        except Exception as exc:
            state.record_error("3b", str(exc))

        logger.info(
            "Phase 3B: verified=%s  adjustments=%d",
            result.get("phase36_domain_verified", False),
            result.get("phase36_adjustments", 0),
        )
        return {
            "candidates": candidates,
            "domain_verified": result.get("phase36_domain_verified", False),
            "adjustments": result.get("phase36_adjustments", 0),
            "validation_rules": result.get("phase36_rules_applied", []),
            "validation_details": result,
        }
