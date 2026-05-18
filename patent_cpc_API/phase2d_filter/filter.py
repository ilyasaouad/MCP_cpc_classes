"""
phase2d_filter/filter.py â€” Phase 2D: Candidate Filtering.

Keeps the top-N candidates from Phase 2C for downstream processing.
Also preserves the full raw pool (all_raw_candidates) so Phase 4B
can look up titles and resolve pillar champions.

Optionally applies Phase2DSubclassAnchor to promote subgroups that
are within the confirmed Phase 1B anchor families.

Old equivalent: Phase2DSubclassAnchor + top-N slice in phase2_runner.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..shared.cpc_filters import filter_allocatable
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase2DFilter(BasePhase):
    """
    Top-N filter + optional anchor promotion.

    Config keys used:
      top_n (int) keep this many candidates after Phase 2C scoring
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = state.phase2c.get("scored_candidates", [])
        top_n = self.cfg.get("top_n", 50)

        all_raw_candidates = list(candidates)

        # Optional anchor boost via Phase2DSubclassAnchor
        anchor_families: List[str] = state.phase1b.get("anchor_families", [])
        if anchor_families:
            try:
                from cpc_classification.cpc_phase2d_anchor import (
                    Phase2DSubclassAnchor,
                )
                anchor = Phase2DSubclassAnchor()
                candidates = anchor.apply(candidates, anchor_families)
            except Exception as exc:
                state.record_warning("2d", f"Anchor promotion failed: {exc}")

        # Final allocatable guard (belt-and-suspenders)
        candidates = filter_allocatable(candidates)
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_candidates = candidates[:top_n]

        logger.info(
            "Phase 2D: kept %d/%d candidates (top_n=%d)",
            len(top_candidates), len(all_raw_candidates), top_n,
        )
        return {
            "candidates": top_candidates,
            "all_raw_candidates": all_raw_candidates,
        }
