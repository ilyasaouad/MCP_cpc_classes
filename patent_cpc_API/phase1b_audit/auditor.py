"""
phase1b_audit/auditor.py â€” Phase 1B: Claim-to-Domain Forensic Audit.

Validates that every domain family proposed by Phase 1A is actually
evidenced in the claims text. Produces anchor_families (signal-confirmed)
and kill_log (families that cannot re-enter the pipeline).

Old equivalent: run_phase1_2() + apply_phase1_2_result()
               in patent_cpc_fastapi/pipeline/phase1_2_runner.py
"""

import logging
from typing import Any, Dict

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase1BAuditor(BasePhase):
    """
    Deterministic claim-signal audit.

    Config keys used:
      min_claim_signal_count (int) signals needed to keep a CPC family
      kill_log_hard_block    (bool) rejected families cannot re-enter
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.pipeline.phase1_2_runner import (
                run_phase1_2,
                apply_phase1_2_result,
            )
        except ImportError as exc:
            state.record_error("1b", f"Import failed: {exc}")
            return {}

        phase1 = state.phase1a
        if not phase1:
            state.record_warning("1b", "Phase 1A result empty â€” skipping audit")
            return {"anchor_families": [], "kill_log": [], "domain_weights": {}}

        claims = state.claims or ""

        try:
            class _FakeCls:
                llm = self.llm

            result = run_phase1_2(_FakeCls(), phase1, claims)
            apply_phase1_2_result(phase1, result)
        except Exception as exc:
            state.record_error("1b", str(exc))
            return {"anchor_families": [], "kill_log": [], "domain_weights": {}}

        anchor_families = result.get("anchor_families", [])
        kill_log = result.get("kill_log", [])

        logger.info(
            "Phase 1B: anchors=%s  kill_log=%s",
            anchor_families,
            kill_log,
        )
        return {
            "anchor_families": anchor_families,
            "kill_log": kill_log,
            "domain_weights": result.get("domain_weights", {}),
            "final_primary_anchor": result.get("final_primary_anchor", ""),
            "audit_details": result,
        }
