"""
phase1a_extraction/extractor.py â€” Phase 1A: Signal Extraction.

Calls the LLM to extract technical terms, core function, technical object,
inventive step, and domain signals from the patent query + claims text.

Old equivalent: run_phase1() in patent_cpc_fastapi/pipeline/phase1_runner.py
"""

import logging
from typing import Any, Dict

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase1AExtractor(BasePhase):
    """
    LLM-driven signal extraction.

    Delegates to the existing CPCExtractor / extracting_cpc.py logic.
    Config keys used:
      min_terms               (int) minimum extracted terms expected
      max_terms               (int) cap on terms passed downstream
      min_importance_threshold (int) drop terms below this score
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.extracting_cpc import CPCExtractor
            from cpc_classification.pipeline.phase1_runner import run_phase1
            from cpc_classification.prompts.shared import label_claims
        except ImportError as exc:
            state.record_error("1a", f"Import failed: {exc}")
            return {}

        query = state.query
        claims = state.claims or ""
        description = state.description or ""

        # run_phase1 uses description as the main text; query goes into description
        full_text = "\n\n".join(filter(None, [query, description]))

        # Claims must be labelled with [INDEPENDENT]/[DEPENDENT] markers
        labeled_claims = label_claims(claims) if claims else ""

        min_thresh = self.cfg.get("min_importance_threshold", 3)

        try:
            # run_phase1 calls classifier.extractor.extract() — must provide it
            extractor_instance = CPCExtractor(self.llm)

            class _FakeCls:
                pass

            _FakeCls.llm = self.llm
            _FakeCls.extractor = extractor_instance

            result = run_phase1(_FakeCls(), full_text, labeled_claims)
        except Exception as exc:
            state.record_error("1a", str(exc))
            return {}

        # Drop terms below importance threshold
        raw_terms = result.get("terms", result.get("essential_terms", []))
        filtered = [
            t for t in raw_terms
            if not isinstance(t, dict) or t.get("importance", 5) >= min_thresh
        ]

        max_terms = self.cfg.get("max_terms", 15)
        filtered = filtered[:max_terms]

        result["terms"] = filtered
        logger.info(
            "Phase 1A: extracted %d terms (threshold=%d)", len(filtered), min_thresh
        )
        return result
