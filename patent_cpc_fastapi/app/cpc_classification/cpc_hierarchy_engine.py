"""
cpc_hierarchy_engine.py - Phase 3.6: Universal CPC Hierarchy CORRECTION ENGINE

PURPOSE:
Select the most accurate CPC subgroup by enforcing TECHNICAL CONTRIBUTION
PRIORITY within ANY CPC family. This phase OVERRIDES all semantic/keyword
bias from previous phases.

CORE PRINCIPLE:
CPC classification is contribution-driven, NOT keyword-driven or domain-driven.
Correct order: 1. What does it DO? → 2. What level is it? → 3. Which CPC matches?

HARD RULES:
- If A or B signals exist → NEVER select C, D, E, or F as primary
- Must choose LOWEST numbered priority category available
- Domain refines but NEVER overrides contribution type
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ── UNIVERSAL CONTRIBUTION TYPES (CORRECTION ENGINE) ──
# Priority: A > B > C > E > D > F

CONTRIBUTION_TYPES = {
    "A": {
        "name": "parameter_model_optimization",
        "label": "A - Parameter/Model Optimization",
        "priority": 1,  # HIGHEST
        "signals": [
            "optimization",
            "parameter",
            "tuning",
            "calibration",
            "adaptation",
            "adjustment",
            "weight",
            "gradient",
            "learning rate",
            "momentum",
            "regularization",
            "hyperparameter",
            "configuration",
            "setting",
            "control parameter",
            "coefficient",
            "variable adjustment",
            "structural optimization",
            "design optimization",
            "performance tuning",
            "feedback control",
            "adaptive control",
            "self-tuning",
            "auto-tuning",
            "weight clipping",
            "gradient clipping",
            "batch normalization",
            "layer normalization",
            "dropout",
            "weight decay",
            "initialization",
            "model improvement",
            "training refinement",
        ],
        "description": "Weight tuning, parameter adjustment, calibration, model improvement, training refinement, performance optimization",
    },
    "B": {
        "name": "compression_efficiency_reduction",
        "label": "B - Compression/Efficiency/Reduction",
        "priority": 2,
        "signals": [
            "compression",
            "quantization",
            "pruning",
            "distillation",
            "reduction",
            "simplification",
            "compact",
            "sparse",
            "sparsity",
            "low-bit",
            "bitwidth",
            "int8",
            "fp16",
            "mixed precision",
            "model compression",
            "weight pruning",
            "knowledge distillation",
            "tensor decomposition",
            "low rank",
            "dimensionality reduction",
            "data compression",
            "signal compression",
            "image compression",
            "bandwidth reduction",
            "memory reduction",
            "size reduction",
            "efficient",
            "efficiency",
            "memory efficient",
            "compute efficient",
            "energy efficient",
            "power reduction",
            "resource optimization",
            "precision reduction",
            "encoding reduction",
            "error correction",
            "channel coding",
            "forward error correction",
            "FEC",
            "ARQ",
            "checksum",
            "parity",
            "redundancy",
        ],
        "description": "Quantization, pruning, compression, memory reduction, computational reduction, precision reduction",
    },
    "C": {
        "name": "system_architecture_design",
        "label": "C - System Architecture/Design",
        "priority": 3,
        "signals": [
            "architecture",
            "structure",
            "design",
            "framework",
            "pipeline",
            "system design",
            "network architecture",
            "model architecture",
            "layer design",
            "topology",
            "configuration",
            "arrangement",
            "organization",
            "scheme",
            "methodology",
            "approach",
            "hardware design",
            "circuit design",
            "protocol design",
            "structural",
            "organizational",
            "systematic",
        ],
        "description": "Structural design of systems, pipelines, hardware/software architecture",
    },
    "E": {
        "name": "signal_data_transformation",
        "label": "E - Signal/Data Transformation",
        "priority": 4,
        "signals": [
            "filtering",
            "transformation",
            "encoding",
            "decoding",
            "modulation",
            "demodulation",
            "conversion",
            "mapping",
            "signal processing",
            "data processing",
            "image processing",
            "feature extraction",
            "preprocessing",
            "postprocessing",
            "normalization",
            "standardization",
            "scaling",
            "transform",
            "translation",
            "rotation",
            "warping",
            "morphing",
            "rendering",
            "synthesis",
            "reconstruction",
        ],
        "description": "Filtering, encoding, transformation, or mapping of data/signals",
    },
    "D": {
        "name": "operation_execution_inference",
        "label": "D - Operation/Execution/Inference",
        "priority": 5,  # LOWER than E
        "signals": [
            "inference",
            "execution",
            "runtime",
            "operation",
            "process",
            "forward pass",
            "prediction",
            "classification",
            "detection",
            "recognition",
            "segmentation",
            "generation",
            "synthesis",
            "reasoning",
            "decision",
            "action",
            "behavior",
            "performance",
            "throughput",
            "latency",
            "response time",
            "real-time",
            "online",
            "streaming",
            "batch processing",
            "deployment",
            "serving",
            "inference engine",
            "runtime system",
        ],
        "description": "Runtime behavior, execution processes, inference, or usage",
    },
    "F": {
        "name": "abstract_modeling_logic_reasoning",
        "label": "F - Abstract Modeling/Logic/Reasoning",
        "priority": 6,  # LOWEST
        "signals": [
            "reasoning",
            "logic",
            "cognitive",
            "semantic",
            "ontology",
            "knowledge representation",
            "expert system",
            "rule-based",
            "fuzzy logic",
            "fuzzy set",
            "membership function",
            "decision support",
            "cognitive model",
            "mental model",
            "conceptual",
            "abstract",
            "theoretical",
            "formal",
            "symbolic",
            "logical inference",
            "deduction",
            "induction",
            "belief",
            "intention",
            "goal",
            "planning",
        ],
        "description": "Reasoning systems, cognitive models, abstract logic, decision systems",
    },
}

# Priority order: lower number = higher priority
PRIORITY_ORDER = ["A", "B", "C", "E", "D", "F"]

# ── CPC FAMILY STRUCTURE PATTERNS ──
# Maps contribution types to typical CPC subgroup patterns
# These are PATTERNS, not fixed codes - they guide selection

FAMILY_STRUCTURE_PATTERNS = {
    # AI / Neural Networks
    "G06N": {
        "A": ["G06N3/045", "G06N3/048", "G06N3/044"],  # Parameter optimization
        "B": ["G06N3/063", "G06N3/065", "G06N3/067"],  # Compression
        "C": ["G06N3/08", "G06N3/02", "G06N3/098"],  # Architecture
        "E": ["G06N3/084", "G06N3/088"],  # Data transformation
        "D": ["G06N5/04", "G06N5/046", "G06N5/048"],  # Inference
        "F": ["G06N5/08", "G06N5/06", "G06N7/"],  # Reasoning
    },
    # Computing / Data Processing
    "G06F": {
        "A": ["G06F9/", "G06F11/", "G06F12/"],  # Optimization
        "B": ["G06F16/", "G06F17/"],  # Compression/encoding
        "C": ["G06F15/", "G06F13/"],  # Architecture
        "E": ["G06F7/", "G06F5/"],  # Data transformation
        "D": ["G06F1/", "G06F3/"],  # Execution/operation
        "F": ["G06F17/30", "G06F17/40"],  # Abstract modeling
    },
    # Image Processing
    "G06T": {
        "A": ["G06T5/", "G06T7/10"],  # Optimization/enhancement
        "B": ["G06T9/", "G06T5/"],  # Compression
        "C": ["G06T17/", "G06T19/"],  # 3D structure
        "E": ["G06T3/", "G06T5/"],  # Transformation
        "D": ["G06T15/", "G06T11/"],  # Rendering/execution
        "F": ["G06T13/", "G06T15/"],  # Abstract graphics
    },
    # Telecom
    "H04L": {
        "A": ["H04L47/", "H04L45/"],  # Protocol adaptation
        "B": ["H04L1/", "H04L25/"],  # Error correction/compression
        "C": ["H04L29/", "H04L45/", "H04L49/"],  # Architecture
        "E": ["H04L27/", "H04L25/"],  # Signal transformation
        "D": ["H04L67/", "H04L69/"],  # Services/execution
        "F": ["H04L9/", "H04L63/"],  # Security/abstract
    },
    # Mechanical
    "F16": {
        "A": ["F16F", "F16K31/", "F16H61/"],  # Control/optimization
        "B": ["F16H55/", "F16H57/"],  # Efficiency
        "C": ["F16J", "F16K", "F16H", "F16B", "F16C"],  # Components
        "E": ["F16H", "F16D"],  # Transmission/transformation
        "D": ["F16P", "F16S"],  # Safety/operation
        "F": ["F16S", "F16P"],  # Abstract systems
    },
    # Medical
    "A61": {
        "A": ["A61B5/", "A61B8/"],  # Adaptive diagnostics
        "B": ["A61B5/055", "A61B6/03"],  # Imaging compression
        "C": ["A61B1/", "A61B17/", "A61B18/"],  # Device design
        "E": ["A61B5/", "A61B8/"],  # Signal transformation
        "D": ["A61B90/", "A61B34/"],  # Operation/execution
        "F": ["A61B5/16", "A61B5/24"],  # Abstract monitoring
    },
    # Default pattern for unknown families
    "DEFAULT": {
        "A": ["/0", "/1", "/2"],  # Optimization prefixes
        "B": ["/3", "/4", "/5"],  # Compression prefixes
        "C": ["/6", "/7"],  # Architecture prefixes
        "E": ["/8", "/9"],  # Transformation prefixes
        "D": ["/10", "/11", "/12"],  # Execution prefixes
        "F": ["/13", "/14", "/15"],  # Abstract prefixes
    },
}


class UniversalCPCHierarchyEngine:
    """
    Phase 3.6: Universal CPC Hierarchy Selection Layer.

    Selects CPC subgroups based on technical contribution type,
    not domain or keyword. Works across ALL CPC families.
    """

    def __init__(
        self,
        match_boost: float = 2.5,
        mismatch_penalty: float = 0.15,
        higher_priority_boost: float = 1.5,
    ):
        self.match_boost = match_boost
        self.mismatch_penalty = mismatch_penalty
        self.higher_priority_boost = higher_priority_boost
        self.rules_log: List[Dict[str, Any]] = []

    def apply_hierarchy(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        primary_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply universal hierarchy CORRECTION ENGINE to CPC candidates.

        HARD RULES:
        - If A or B signals exist → ONLY allow A or B as primary
        - Must choose LOWEST numbered priority category available
        - Domain refines but NEVER overrides contribution type

        Args:
            candidates: Phase 3.5 adjusted candidates
            phase1_data: Phase 1 extraction data
            primary_domain: Detected primary domain (e.g., "ai", "telecom")

        Returns:
            Dict with refined candidates, contribution types, and rules log
        """
        if not candidates:
            return self._empty_result()

        self.rules_log = []
        candidates = [dict(c) for c in candidates]  # Copy

        # Step 1: Detect contribution types from patent text
        detected_types = self._detect_contribution_types(phase1_data)

        if not detected_types:
            logger.info("Phase 3.6: No contribution types detected")
            return {
                "phase36_candidates": candidates,
                "phase36_types": {},
                "phase36_primary_type": None,
                "phase36_secondary_types": [],
                "phase36_primary_code": None,
                "phase36_rules_log": [],
                "phase36_adjustments": 0,
            }

        # Step 2: Sort by priority (A=1 highest)
        sorted_types = sorted(detected_types.items(), key=lambda x: x[1]["priority"])

        primary_type = sorted_types[0][0]
        primary_config = sorted_types[0][1]
        secondary_types = [t[0] for t in sorted_types[1:]]

        # ── HARD SELECTION RULE ──
        # If A or B exists, ONLY A or B can be primary
        # This enforces that optimization/compression ALWAYS beats inference/reasoning
        has_high_priority = any(t in detected_types for t in ["A", "B"])
        if has_high_priority and primary_type not in ["A", "B"]:
            # Force primary to be the highest A or B available
            for t in ["A", "B"]:
                if t in detected_types:
                    primary_type = t
                    primary_config = detected_types[t]
                    logger.warning(
                        "HARD RULE ENFORCED: Switched primary from %s to %s "
                        "(A/B signals detected)",
                        sorted_types[0][0],
                        t,
                    )
                    break

        logger.info(
            "Phase 3.6: Primary contribution=%s (priority=%d), signals=%s",
            primary_type,
            primary_config["priority"],
            primary_config["matched_signals"][:3],
        )

        # Step 3: Map contribution type to CPC patterns for this domain
        family_patterns = self._get_family_patterns(primary_domain)

        # Step 4: Adjust candidate scores based on contribution match
        candidates = self._adjust_by_contribution(
            candidates,
            primary_type,
            secondary_types,
            family_patterns,
            detected_types,
        )

        # Step 5: Apply invalid class filter
        candidates = self._apply_invalid_filter(
            candidates, primary_type, detected_types, family_patterns
        )

        # Step 6: Re-sort and normalize
        candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[
            :10
        ]

        return {
            "phase36_candidates": candidates,
            "phase36_types": {
                k: v["matched_signals"][:3] for k, v in detected_types.items()
            },
            "phase36_primary_type": CONTRIBUTION_TYPES[primary_type]["label"],
            "phase36_secondary_types": [
                CONTRIBUTION_TYPES[t]["label"] for t in secondary_types
            ],
            "phase36_primary_code": primary_type,
            "phase36_rules_log": self.rules_log,
            "phase36_adjustments": len(self.rules_log),
        }

    def _detect_contribution_types(
        self, phase1_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Detect which universal contribution types are present in the patent."""
        # Gather all text
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )
        core_function = phase1_data.get("core_function", "").lower()
        technical_object = phase1_data.get("technical_object", "").lower()
        system_context = phase1_data.get("system_context", "").lower()
        all_text = f"{all_terms} {core_function} {technical_object} {system_context}"

        detected = {}
        for type_code, config in CONTRIBUTION_TYPES.items():
            matched = []
            for signal in config["signals"]:
                if signal in all_text:
                    matched.append(signal)

            if matched:
                detected[type_code] = {
                    "priority": config["priority"],
                    "name": config["name"],
                    "matched_signals": matched,
                    "signal_count": len(matched),
                }
                logger.debug(
                    "Phase 3.6: Detected contribution %s with %d signals",
                    type_code,
                    len(matched),
                )

        return detected

    def _get_family_patterns(
        self, primary_domain: Optional[str]
    ) -> Dict[str, List[str]]:
        """Get CPC pattern mapping for the detected domain."""
        if not primary_domain:
            return FAMILY_STRUCTURE_PATTERNS["DEFAULT"]

        # Map domain name to CPC family code
        family_code = self._domain_to_family_code(primary_domain)

        if family_code and family_code in FAMILY_STRUCTURE_PATTERNS:
            return FAMILY_STRUCTURE_PATTERNS[family_code]

        # Try to find partial match
        for fam_code in FAMILY_STRUCTURE_PATTERNS:
            if fam_code != "DEFAULT" and primary_domain.lower().startswith(
                fam_code.lower()
            ):
                return FAMILY_STRUCTURE_PATTERNS[fam_code]

        # Fall back to default
        return FAMILY_STRUCTURE_PATTERNS["DEFAULT"]

    def _domain_to_family_code(self, domain: str) -> Optional[str]:
        """Map domain name to CPC family code."""
        mapping = {
            "ai": "G06N",
            "machine learning": "G06N",
            "neural networks": "G06N",
            "general computing": "G06F",
            "data processing": "G06F",
            "image": "G06T",
            "image processing": "G06T",
            "telecom": "H04L",
            "telecommunications": "H04L",
            "mechanical": "F16",
            "medical": "A61",
        }
        return mapping.get(domain.lower())

    def _adjust_by_contribution(
        self,
        candidates: List[Dict[str, Any]],
        primary_type: str,
        secondary_types: List[str],
        family_patterns: Dict[str, List[str]],
        detected_types: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Adjust candidate scores based on contribution type match."""
        # Get patterns for primary and secondary types
        primary_patterns = family_patterns.get(primary_type, [])
        secondary_patterns = []
        for st in secondary_types[:2]:  # Top 2 secondary types
            secondary_patterns.extend(family_patterns.get(st, []))

        # Get patterns for all lower-priority types (to penalize)
        all_lower_patterns = []
        primary_priority = CONTRIBUTION_TYPES[primary_type]["priority"]
        for type_code, config in CONTRIBUTION_TYPES.items():
            if config["priority"] > primary_priority:
                all_lower_patterns.extend(family_patterns.get(type_code, []))

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family3 = symbol[:3] if len(symbol) >= 3 else symbol

            # Check if candidate matches primary contribution patterns
            matches_primary = any(symbol.startswith(p) for p in primary_patterns)

            # Check if candidate matches secondary contribution patterns
            matches_secondary = any(symbol.startswith(p) for p in secondary_patterns)

            # Check if candidate matches lower-priority patterns (to penalize)
            matches_lower = any(symbol.startswith(p) for p in all_lower_patterns)

            if matches_primary:
                # Strong boost for matching primary contribution type
                old_score = score
                score *= self.match_boost
                self._log_rule(
                    "CONTRIBUTION_PRIMARY",
                    symbol,
                    old_score,
                    score,
                    f"Matches primary contribution '{CONTRIBUTION_TYPES[primary_type]['name']}'",
                )
            elif matches_secondary:
                # Moderate boost for secondary contribution type
                old_score = score
                score *= self.higher_priority_boost
                self._log_rule(
                    "CONTRIBUTION_SECONDARY",
                    symbol,
                    old_score,
                    score,
                    f"Matches secondary contribution type",
                )
            elif matches_lower:
                # Strong penalty for lower-priority contribution type
                # e.g., inference/reasoning when optimization exists
                old_score = score
                score *= self.mismatch_penalty
                self._log_rule(
                    "CONTRIBUTION_LOWER_PRIORITY",
                    symbol,
                    old_score,
                    score,
                    f"Lower priority contribution when '{CONTRIBUTION_TYPES[primary_type]['name']}' exists",
                )
            else:
                # Candidate doesn't match any specific pattern - neutral
                # But we still want to penalize if it's in a known lower-priority area
                pass

            candidate["score"] = round(score, 4)
            candidate["contribution_match"] = (
                "primary"
                if matches_primary
                else (
                    "secondary"
                    if matches_secondary
                    else ("lower" if matches_lower else "neutral")
                )
            )
            adjusted.append(candidate)

        return adjusted

    def _apply_invalid_filter(
        self,
        candidates: List[Dict[str, Any]],
        primary_type: str,
        detected_types: Dict[str, Dict[str, Any]],
        family_patterns: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Apply invalid class filter.

        HARD RULES:
        - If A or B exists → penalize C, D, E, F heavily
        - Must choose lowest numbered priority available
        """
        has_a_or_b = "A" in detected_types or "B" in detected_types
        primary_priority = CONTRIBUTION_TYPES[primary_type]["priority"]

        if not has_a_or_b or primary_type in ["A", "B"]:
            # No need for extra filtering - already correct
            return candidates

        # If we get here, something went wrong - A or B exists but primary is lower
        # This should NOT happen due to hard rule enforcement, but just in case
        logger.warning(
            "INVALID FILTER: A/B exists but primary is %s (priority=%d). "
            "Applying emergency penalty.",
            primary_type,
            primary_priority,
        )

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            match_type = candidate.get("contribution_match", "neutral")

            # Check if candidate is in a forbidden lower-priority category
            if has_a_or_b and match_type in ["lower", "neutral"]:
                # Heavy penalty for lower-priority candidates when A/B exists
                old_score = score
                score *= 0.05  # Very strong penalty
                self._log_rule(
                    "INVALID_CLASS_FILTER",
                    symbol,
                    old_score,
                    score,
                    f"HARD RULE: A/B exists → penalizing lower-priority candidate",
                )
                candidate["score"] = round(score, 4)

            adjusted.append(candidate)

        return adjusted

    def _log_rule(
        self,
        rule: str,
        symbol: str,
        before: float,
        after: float,
        reason: str,
    ) -> None:
        """Log a rule application."""
        self.rules_log.append(
            {
                "rule": rule,
                "symbol": symbol,
                "score_before": round(before, 4),
                "score_after": round(after, 4),
                "reason": reason,
            }
        )

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result."""
        return {
            "phase36_candidates": [],
            "phase36_types": {},
            "phase36_primary_type": None,
            "phase36_secondary_types": [],
            "phase36_primary_code": None,
            "phase36_rules_log": [],
            "phase36_adjustments": 0,
        }


def apply_universal_hierarchy(
    candidates: List[Dict[str, Any]],
    phase1_data: Dict[str, Any],
    primary_domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Quick function to apply Phase 3.6 universal hierarchy."""
    engine = UniversalCPCHierarchyEngine()
    return engine.apply_hierarchy(candidates, phase1_data, primary_domain)
