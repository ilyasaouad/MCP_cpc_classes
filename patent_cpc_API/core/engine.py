"""
engine.py — Pipeline orchestrator.

Instantiates each phase module, wires them together via PipelineState,
and runs the full A→B→C chain within each group.

All thresholds come from pipeline_config.yaml — no magic numbers here.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import yaml

from .state_manager import PipelineState

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_HERE, "..", "config", "pipeline_config.yaml")
)


def _load_config(path: str = _DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class CPCPipelineEngine:
    """
    Orchestrates the five phase-groups of the CPC classification pipeline.

    Instantiate once at startup (engine is stateless between requests).
    Call `.run(query, claims, description)` per request.
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        kg=None,
        xml_parser=None,
        llm_client=None,
    ):
        self.cfg = _load_config(config_path)
        self.kg = kg
        self.xml_parser = xml_parser
        self.llm = llm_client

        # Lazy-import phase modules to keep startup fast
        self._phases = self._init_phases()
        logger.info("CPCPipelineEngine ready — %d phases loaded", len(self._phases))

    # ── Phase initialisation ──────────────────────────────────────────────────

    def _init_phases(self):
        """Import and instantiate all phase runners."""
        from ..phase1a_extraction.extractor import Phase1AExtractor
        from ..phase1b_audit.auditor import Phase1BAuditor
        from ..phase1c_character.characterizer import Phase1CCharacterizer
        from ..phase2a_routing.router import Phase2ARouter
        from ..phase2b_expansion.expander import Phase2BExpander
        from ..phase2c_scoring.scorer import Phase2CScorer
        from ..phase2d_filter.filter import Phase2DFilter
        from ..phase3a_constraints.constraints import Phase3AConstraints
        from ..phase3b_validation.validator import Phase3BValidator
        from ..phase4a_consolidation.consolidator import Phase4AConsolidator
        from ..phase4b_resolution.resolver import Phase4BResolver
        from ..phase5a_consistency.consistency import Phase5AConsistency
        from ..phase5b_labeling.labeler import Phase5BLabeler

        shared = dict(kg=self.kg, xml_parser=self.xml_parser, llm=self.llm)

        return {
            "1a": Phase1AExtractor(self.cfg.get("phase1a_extraction", {}), **shared),
            "1b": Phase1BAuditor(self.cfg.get("phase1b_audit", {}), **shared),
            "1c": Phase1CCharacterizer(self.cfg.get("phase1c_character", {}), **shared),
            "2a": Phase2ARouter(self.cfg.get("phase2a_routing", {}), **shared),
            "2b": Phase2BExpander(self.cfg.get("phase2b_expansion", {}), **shared),
            "2c": Phase2CScorer(self.cfg.get("phase2c_scoring", {}), **shared),
            "2d": Phase2DFilter(self.cfg.get("phase2d_filter", {}), **shared),
            "3a": Phase3AConstraints(self.cfg.get("phase3a_constraints", {}), **shared),
            "3b": Phase3BValidator(self.cfg.get("phase3b_validation", {}), **shared),
            "4a": Phase4AConsolidator(self.cfg.get("phase4a_consolidation", {}), **shared),
            "4b": Phase4BResolver(self.cfg.get("phase4b_resolution", {}), **shared),
            "5a": Phase5AConsistency(self.cfg.get("phase5a_consistency", {}), **shared),
            "5b": Phase5BLabeler(self.cfg.get("phase5b_labeling", {}), **shared),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        claims: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline and return the final state as a dict.

        Each phase writes its result into `state` and passes control to the next.
        Errors in any phase are recorded but do not abort the pipeline.
        """
        state = PipelineState(query=query, claims=claims, description=description)
        t0 = time.perf_counter()

        _SEQUENCE = [
            ("1a", "phase1a"),
            ("1b", "phase1b"),
            ("1c", "phase1c"),
            ("2a", "phase2a"),
            ("2b", "phase2b"),
            ("2c", "phase2c"),
            ("2d", "phase2d"),
            ("3a", "phase3a"),
            ("3b", "phase3b"),
            ("4a", "phase4a"),
            ("4b", "phase4b"),
            ("5a", "phase5a"),
            ("5b", "phase5b"),
        ]

        for phase_key, attr in _SEQUENCE:
            runner = self._phases.get(phase_key)
            if runner is None:
                continue
            t_phase = time.perf_counter()
            try:
                result = runner.run(state)
                setattr(state, attr, result or {})
            except Exception as exc:
                logger.exception("Phase %s failed: %s", phase_key, exc)
                state.record_error(phase_key, str(exc))
            logger.debug(
                "Phase %s: %.0f ms", phase_key,
                (time.perf_counter() - t_phase) * 1000,
            )

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("Pipeline complete in %.0f ms | errors=%d", total_ms, len(state.errors))

        result = state.to_dict()
        result["elapsed_ms"] = round(total_ms)
        return result
