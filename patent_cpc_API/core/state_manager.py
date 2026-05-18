"""
state_manager.py — Pipeline state container.

A single PipelineState instance flows through all phases.
Each phase reads what it needs and writes its results back.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineState:
    # ── Input ────────────────────────────────────────────────────────────────
    query: str = ""
    claims: Optional[str] = None
    description: Optional[str] = None

    # ── Phase 1A: Signal Extraction ──────────────────────────────────────────
    phase1a: Dict[str, Any] = field(default_factory=dict)
    # keys: terms, essential_terms, core_function, technical_object,
    #       inventive_step, domain_signals, quality_score

    # ── Phase 1B: Audit ───────────────────────────────────────────────────────
    phase1b: Dict[str, Any] = field(default_factory=dict)
    # keys: anchor_families, kill_log, domain_weights

    # ── Phase 1C: Character ───────────────────────────────────────────────────
    phase1c: Dict[str, Any] = field(default_factory=dict)
    # keys: tcr_score, tcr_label, software_bias, hardware_bias

    # ── Phase 2A: Routing ─────────────────────────────────────────────────────
    phase2a: Dict[str, Any] = field(default_factory=dict)
    # keys: family_scores {G10L: 0.45, G06N: 0.35, ...},
    #       layer_result {mode, families}

    # ── Phase 2B: Expansion ───────────────────────────────────────────────────
    phase2b: Dict[str, Any] = field(default_factory=dict)
    # keys: expanded_candidates [{symbol, title, score, family}, ...]

    # ── Phase 2C: Scoring ─────────────────────────────────────────────────────
    phase2c: Dict[str, Any] = field(default_factory=dict)
    # keys: scored_candidates (after RRF + family normalization)

    # ── Phase 2D: Filter ──────────────────────────────────────────────────────
    phase2d: Dict[str, Any] = field(default_factory=dict)
    # keys: candidates (top N), all_raw_candidates

    # ── Phase 3A: Constraints ─────────────────────────────────────────────────
    phase3a: Dict[str, Any] = field(default_factory=dict)
    # keys: candidates, adjustments, domain, domain_confidence, layer_mode

    # ── Phase 3B: Validation ─────────────────────────────────────────────────
    phase3b: Dict[str, Any] = field(default_factory=dict)
    # keys: candidates, domain_verified, adjustments, validation_rules

    # ── Phase 4A: Consolidation ──────────────────────────────────────────────
    phase4a: Dict[str, Any] = field(default_factory=dict)
    # keys: hypotheses [{family, candidates, coherence, support_weight}]

    # ── Phase 4B: Resolution ─────────────────────────────────────────────────
    phase4b: Dict[str, Any] = field(default_factory=dict)
    # keys: primary_cpc, confidence, pillars, functional_alignment,
    #       technical_coverage, score

    # ── Phase 5A: Consistency ────────────────────────────────────────────────
    phase5a: Dict[str, Any] = field(default_factory=dict)
    # keys: consistency_check, is_consistent, adjustments

    # ── Phase 5B: Labeling ────────────────────────────────────────────────────
    phase5b: Dict[str, Any] = field(default_factory=dict)
    # keys: primary_label, justification, supporting_codes, confidence_label

    # ── Pipeline metadata ────────────────────────────────────────────────────
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def record_error(self, phase: str, msg: str) -> None:
        self.errors.append(f"[{phase}] {msg}")

    def record_warning(self, phase: str, msg: str) -> None:
        self.warnings.append(f"[{phase}] {msg}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "phase1a": self.phase1a,
            "phase1b": self.phase1b,
            "phase1c": self.phase1c,
            "phase2a": self.phase2a,
            "phase2b": self.phase2b,
            "phase2c": self.phase2c,
            "phase2d": self.phase2d,
            "phase3a": self.phase3a,
            "phase3b": self.phase3b,
            "phase4a": self.phase4a,
            "phase4b": self.phase4b,
            "phase5a": self.phase5a,
            "phase5b": self.phase5b,
            "errors": self.errors,
            "warnings": self.warnings,
        }
