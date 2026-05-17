"""
technical_weight_analyzer.py - Stable Technical Character Analyzer (v2.2)

PURPOSE:
Replaces fragile keyword-heavy TCR with a balanced, deterministic scoring system.

DESIGN PRINCIPLES:
- No LLM dependency
- No CPC-based leakage
- Balanced computational vs physical signal space
- Uses structured term matching only
- Acts as a ROUTING SIGNAL, not a classifier

CHANGELOG:
v2.1 — Fixed signature mismatch with phase15_tcr_runner.py
        Added missing output keys (tcr_bias, force_mode, override_applied, confidence)
        Improved _match() to use word-boundary matching
v2.2 — Fixed TCR explosion when physical_score = 0
        Expanded computational keywords (lstm, weights, speech, phoneme, etc.)
        Lowered confidence thresholds for low evidence counts
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)


# =========================
# BALANCED TERM SEEDS
# =========================

COMPUTATIONAL_KEYWORDS: Set[str] = {
    # Core computation
    "algorithm",
    "computation",
    "processing",
    "pipeline",
    "workflow",
    "software",
    "module",
    "service",
    "interface",
    "program",
    # ML / AI architectures
    "machine learning",
    "deep learning",
    "neural network",
    "model",
    "training",
    "inference",
    "optimization",
    "embedding",
    "feature extraction",
    "classification",
    "clustering",
    "self-supervised",
    "pre-training",
    "pretraining",
    "self supervised",
    "encoder",
    "decoder",
    "transformer",
    "attention",
    "attention mechanism",
    # Specific ML architectures (v2.2 — added)
    "lstm",
    "rnn",
    "cnn",
    "gru",
    "bidirectional",
    "weights",
    "parameters",
    "gradients",
    "backpropagation",
    "representation",
    "representations",
    "embedding",
    "hidden layer",
    "output layer",
    "input layer",
    "activation",
    "loss function",
    "softmax",
    "token",
    "tokenization",
    "vocabulary",
    # Data systems
    "database",
    "query",
    "retrieval",
    "indexing",
    "vector",
    "graph",
    "api",
    "cloud",
    "distributed",
    "parallel",
    # Signal / language / speech (v2.2 — expanded)
    "speech",
    "speech recognition",
    "speech processing",
    "audio",
    "audio processing",
    "phoneme",
    "phonemic",
    "phonetic",
    "acoustic",
    "acoustic model",
    "spectral",
    "mfcc",
    "nlp",
    "tokenization",
    "encoding",
    "decoding",
    "semantic",
    "parser",
    "utterance",
    "voice",
    "speaker",
}


PHYSICAL_KEYWORDS: Set[str] = {
    # Mechanical
    "motor",
    "engine",
    "pump",
    "valve",
    "gear",
    "mechanism",
    "turbine",
    "actuator",
    "machine",
    "structure",
    "shaft",
    "bearing",
    "spring",
    "cam",
    "crankshaft",
    # Electrical / electronics
    "circuit",
    "sensor",
    "transistor",
    "chip",
    "semiconductor",
    "pcb",
    "battery",
    "voltage",
    "current",
    "electrical",
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    # Chemical / material
    "chemical",
    "compound",
    "reaction",
    "material",
    "polymer",
    "solution",
    "mixture",
    "catalyst",
    "substrate",
    # Biomedical
    "patient",
    "tissue",
    "cell",
    "implant",
    "drug",
    "medical",
    "diagnosis",
    "therapy",
    "surgical",
    # Physical systems
    "fluid",
    "pressure",
    "temperature",
    "force",
    "mechanical system",
    "environment",
    "vehicle",
    "torque",
    "friction",
    "vibration",
    "thermal",
}

# ═══════════════════════════════════════════════════════════════
# Pre-compiled regex patterns for word-boundary matching
# ═══════════════════════════════════════════════════════════════
_COMP_PATTERNS = [
    re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in COMPUTATIONAL_KEYWORDS
]
_PHYS_PATTERNS = [
    re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in PHYSICAL_KEYWORDS
]


# =========================
# ANALYZER
# =========================


class TechnicalWeightAnalyzer:
    """
    Stable deterministic analyzer for computing technical bias.
    """

    def __init__(
        self,
        computational_keywords: Optional[Set[str]] = None,
        physical_keywords: Optional[Set[str]] = None,
    ):
        self.computational_keywords = computational_keywords or COMPUTATIONAL_KEYWORDS
        self.physical_keywords = physical_keywords or PHYSICAL_KEYWORDS

        # Build patterns if custom keywords provided
        if computational_keywords is not None:
            self._comp_patterns = [
                re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE)
                for k in computational_keywords
            ]
        else:
            self._comp_patterns = _COMP_PATTERNS

        if physical_keywords is not None:
            self._phys_patterns = [
                re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE)
                for k in physical_keywords
            ]
        else:
            self._phys_patterns = _PHYS_PATTERNS

    # -------------------------
    # MAIN ENTRY
    # -------------------------

    def analyze(
        self,
        phase1_data: Dict[str, Any],
        phase1_5_result: Optional[Dict[str, Any]] = None,
        domain_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze technical weight of the invention.

        Args:
            phase1_data: Phase 1 output with terms, domain_signals, etc.
            phase1_5_result: Phase 1.5 role classification result (optional, for context).
            domain_signals: Domain signals from Phase 1 (optional, for additional context).

        Returns:
            Dict with TCR score, force mode, and detailed breakdown.
        """
        role = (phase1_5_result or {}).get("role", "")
        role_confidence = (phase1_5_result or {}).get("confidence", 0.0)

        terms = self._extract_terms(phase1_data)

        # Also extract from raw claims text and core_function for wider coverage
        claims_terms = self._extract_claims_terms(phase1_data)
        core_fn = phase1_data.get("core_function", "")
        if core_fn:
            parts = [
                p.strip().lower() for p in re.split(r"[,;.\n]", core_fn) if p.strip()
            ]
            for p in parts:
                word_count = len(p.split())
                if word_count <= 6 and len(p) > 2:
                    terms.append({"term": p, "importance": 8})

        # Deduplicate terms with length guard
        seen_terms = set()
        deduped = []
        for t in terms:
            term_text = t["term"].lower().strip()
            word_count = len(term_text.split())
            if term_text and word_count <= 6 and term_text not in seen_terms:
                seen_terms.add(term_text)
                deduped.append(t)
        terms = deduped

        comp_score = 0.0
        phys_score = 0.0

        comp_terms = []
        phys_terms = []

        for t in terms:
            term = t["term"].lower()
            weight = t.get("importance", 5)

            if self._match(term, self._comp_patterns):
                comp_score += weight
                comp_terms.append(term)

            elif self._match(term, self._phys_patterns):
                phys_score += weight
                phys_terms.append(term)

        # Also score claims terms directly (without adding to terms list)
        for ct in claims_terms:
            ct_lower = ct.lower()
            if ct_lower not in seen_terms:
                if self._match(ct_lower, self._comp_patterns):
                    comp_score += 5
                    comp_terms.append(f"[claims] {ct_lower}")
                    seen_terms.add(ct_lower)
                elif self._match(ct_lower, self._phys_patterns):
                    phys_score += 5
                    phys_terms.append(f"[claims] {ct_lower}")
                    seen_terms.add(ct_lower)

        # -------------------------
        # STABLE TCR COMPUTATION (v3.0 — no artifactual floors)
        # -------------------------
        # TCR is comp_score / phys_score with these rules:
        #   phys_score == 0 → TCR based on comp_score magnitude (discriminative, not capped)
        #   comp_score == 0 → TCR based on phys_score magnitude
        #   both present    → ratio capped at [0.1, 10.0]

        if comp_score == 0 and phys_score == 0:
            tcr = 1.0
        elif phys_score == 0:
            tcr = min(comp_score * 0.5 + 1.0, 10.0)
        elif comp_score == 0:
            tcr = max(0.1, 1.0 - phys_score * 0.05)
        else:
            tcr = min(max(comp_score / phys_score, 0.1), 10.0)

        # -------------------------
        # TCR BIAS (normalized asymmetry)
        # -------------------------
        total = comp_score + phys_score
        if total > 0:
            tcr_bias = (comp_score - phys_score) / total  # Range: -1.0 to +1.0
        else:
            tcr_bias = 0.0

        # -------------------------
        # FORCE FLAG (SAFE THRESHOLDS)
        # -------------------------

        if tcr > 2.0:
            force_mode = "FORCE_SOFTWARE_CORE"
        elif tcr < 0.5:
            force_mode = "FORCE_DOMAIN_CORE"
        else:
            force_mode = "HYBRID_INVENTION"

        # -------------------------
        # CONFIDENCE (v2.2 — stricter thresholds)
        # -------------------------
        total_evidence = len(comp_terms) + len(phys_terms)
        if total_evidence >= 8:
            confidence = 0.95
        elif total_evidence >= 5:
            confidence = 0.90
        elif total_evidence >= 3:
            confidence = 0.75
        elif total_evidence >= 2:
            confidence = 0.50
        elif total_evidence >= 1:
            confidence = 0.30  # Lowered from 0.50 — 1 match is weak evidence
        else:
            confidence = 0.0  # No evidence — neutral fallback

        # -------------------------
        # OVERRIDE (role-based adjustment)
        # -------------------------
        override_applied = False
        if role == "SUPPORT" and role_confidence >= 0.6:
            # SUPPORT inventions (monitoring, logging, UI) are rarely pure software
            # cores — soften the force_mode to avoid over-biasing Phase 2 search.
            if force_mode == "FORCE_SOFTWARE_CORE":
                force_mode = "HYBRID_INVENTION"
                override_applied = True
                logger.info(
                    "TCR override: SUPPORT role (conf=%.2f) → downgraded "
                    "FORCE_SOFTWARE_CORE to HYBRID_INVENTION",
                    role_confidence,
                )
        elif role == "APPLICATION" and role_confidence >= 0.6:
            # APPLICATION inventions apply tech to a domain — domain context matters,
            # so a pure software force is too strong.
            if force_mode == "FORCE_SOFTWARE_CORE" and tcr_bias < 0.95:
                force_mode = "HYBRID_INVENTION"
                override_applied = True
                logger.info(
                    "TCR override: APPLICATION role (conf=%.2f, bias=%.2f) → "
                    "downgraded FORCE_SOFTWARE_CORE to HYBRID_INVENTION",
                    role_confidence,
                    tcr_bias,
                )

        # -------------------------
        # DETERMINISTIC OUTPUT
        # -------------------------

        return {
            # Core TCR metrics
            "tcr": round(tcr, 3),
            "tcr_bias": round(tcr_bias, 3),
            # Force mode
            "force_mode": force_mode,
            # Component scores
            "computational_score": round(comp_score, 2),
            "physical_score": round(phys_score, 2),
            # Confidence
            "confidence": confidence,
            # Override tracking
            "override_applied": override_applied,
            # Detailed breakdown (for debugging)
            "computational_terms": comp_terms,
            "physical_terms": phys_terms,
            # Term counts (for debugging)
            "comp_term_count": len(comp_terms),
            "phys_term_count": len(phys_terms),
            # Legacy alias removed in v3.0 — use force_mode exclusively
            "dominant_bucket": "computational"
            if comp_score >= phys_score
            else "physical",
            "analysis_mode": "stable_rule_based_v2.2",
        }

    # -------------------------
    # HELPERS
    # -------------------------

    def _extract_claims_terms(self, phase1_data: Dict[str, Any]) -> List[str]:
        """Extract individual terms from raw claims text."""
        claims_terms: List[str] = []
        claims_text = (
            phase1_data.get("claims_text", "")
            or phase1_data.get("labeled_claims", "")
            or ""
        )
        if not claims_text:
            return claims_terms
        # Strip claim labels
        claims_clean = re.sub(r"\[(INDEPENDENT|DEPENDENT[^\]]*)\]", "", claims_text)
        # Split into individual words/phrases
        parts = re.split(r'[,;.\n"()\[\]{}]+', claims_clean)
        for p in parts:
            p = p.strip().lower()
            words = p.split()
            # Reject full sentences (more than 6 words) — single vocabulary terms only
            if 2 < len(p) and len(words) <= 6:
                claims_terms.append(p)
        return claims_terms

    def _extract_terms(self, phase1_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract terms from multiple possible Phase 1 keys."""
        # 1. Base terms
        terms = (
            phase1_data.get("terms", []) or phase1_data.get("essential_terms", []) or []
        )

        # 2. Evidence Table (Fix 1)
        evidence_table = phase1_data.get("evidence_table", [])
        if evidence_table:
            for row in evidence_table:
                if isinstance(row, dict) and row.get("term"):
                    terms.append({
                        "term": row["term"],
                        "importance": row.get("weight", 5)
                    })

        # 3. CPC Terms (Fix 2)
        cpc_terms = phase1_data.get("cpc_terms", [])
        if cpc_terms:
            for ct in cpc_terms:
                if isinstance(ct, str):
                    terms.append({"term": ct, "importance": 7})

        # 4. Core Function Generalized (Fix 3)
        generalized_fns = phase1_data.get("core_function_generalized", [])
        if generalized_fns:
            for gfn in generalized_fns:
                if isinstance(gfn, str):
                    # Split into sub-parts (max 4 words each) for wider matching
                    parts = re.split(r"[,;.\n]", gfn)
                    for p in parts:
                        p = p.strip()
                        if p and len(p) > 3:
                            terms.append({"term": p, "importance": 8})

        # 5. Disambiguated terms fallback
        if not terms:
            disamb = phase1_data.get("disambiguated_terms", [])
            if disamb:
                terms = [
                    {"term": d.get("term", ""), "importance": 7}
                    for d in disamb
                    if d.get("term")
                ]

        cleaned = []
        seen = set()
        for t in terms:
            if isinstance(t, dict):
                term_text = t.get("term", "").strip().lower()
                importance = t.get("importance", t.get("weight", 5))
                if term_text and len(term_text) > 2 and term_text not in seen:
                    seen.add(term_text)
                    cleaned.append(
                        {
                            "term": t.get("term", "").strip(),
                            "importance": importance
                            if isinstance(importance, (int, float))
                            else 5,
                        }
                    )
        return cleaned

    def _match(self, term: str, patterns: List[re.Pattern]) -> bool:
        """Match term against pre-compiled word-boundary patterns."""
        for pattern in patterns:
            if pattern.search(term):
                return True
        return False


# =========================
# PUBLIC API
# =========================


def apply_technical_weight_analysis(
    phase1_data: Dict[str, Any],
    phase1_5_result: Optional[Dict[str, Any]] = None,
    domain_signals: Optional[List[Dict[str, Any]]] = None,
    analyzer: Optional[TechnicalWeightAnalyzer] = None,
) -> Dict[str, Any]:
    """Public API for TCR analysis."""
    if analyzer is None:
        analyzer = TechnicalWeightAnalyzer()

    return analyzer.analyze(
        phase1_data,
        phase1_5_result=phase1_5_result,
        domain_signals=domain_signals,
    )
