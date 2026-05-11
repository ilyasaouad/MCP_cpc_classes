"""
cpc_cross_domain_validator.py - Phase 3.6: Cross-Domain Validation Layer

PURPOSE:
Prevent domain collapse caused by overly aggressive abstraction layers.
Validate that selected CPC candidates actually match the patent's domain signals.

CORE PRINCIPLE:
CPC classification is NOT about abstract contribution types.
It is about: Technical Domain → Sub-domain → Specific Function → Application

This phase acts as a GUARDRAIL, not a classifier.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ── DOMAIN SIGNAL REQUIREMENTS ──
# Each CPC family requires specific signals to be present

DOMAIN_REQUIREMENTS = {
    "G06N": {
        "required_signals": [
            "neural",
            "network",
            "learning",
            "model",
            "training",
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "llm",
            "large language model",
            "transformer",
        ],
        "forbidden_signals": [],  # No forbidden signals for G06N
        "min_signal_count": 1,
    },
    "G06F": {
        "required_signals": [
            "computing",
            "computer",
            "processing",
            "software",
            "program",
            "data processing",
            "information processing",
            "digital",
            "algorithm",
            "code",
            "application",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "G06T": {
        "required_signals": [
            "image",
            "video",
            "pixel",
            "graphics",
            "visual",
            "picture",
            "photo",
            "rendering",
            "texture",
            "frame",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "G10L": {
        "required_signals": [
            "speech",
            "audio",
            "voice",
            "sound",
            "acoustic",
            "utterance",
            "phonetic",
            "speaker",
            "microphone",
            "audio signal",
            "sound wave",
            "vocal",
        ],
        "forbidden_signals": [],
        "min_signal_count": 2,  # Speech requires stronger signals
    },
    "H04L": {
        "required_signals": [
            "network",
            "communication",
            "protocol",
            "transmission",
            "telecom",
            "wireless",
            "data transmission",
            "packet",
            "routing",
            "switching",
            "bandwidth",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "H04W": {
        "required_signals": [
            "wireless",
            "mobile",
            "wifi",
            "bluetooth",
            "cellular",
            "radio",
            "spectrum",
            "base station",
            "access point",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "F16": {
        "required_signals": [
            "mechanical",
            "engine",
            "pump",
            "valve",
            "gear",
            "bearing",
            "seal",
            "housing",
            "shaft",
            "coupling",
            "structural",
            "mechanism",
            "assembly",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "A61B": {
        "required_signals": [
            "medical",
            "diagnosis",
            "diagnostic",
            "patient",
            "clinical",
            "surgical",
            "therapy",
            "treatment",
            "healthcare",
            "biomedical",
            "imaging",
            "medical device",
            "endoscopy",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "A61K": {
        "required_signals": [
            "pharmaceutical",
            "drug",
            "medication",
            "vaccine",
            "compound",
            "therapeutic",
            "biochemical",
            "molecule",
            "formulation",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
    "G01": {
        "required_signals": [
            "measurement",
            "sensor",
            "sensing",
            "detecting",
            "testing",
            "instrument",
            "gauge",
            "meter",
            "optical measurement",
            "calibration",
            "metrology",
        ],
        "forbidden_signals": [],
        "min_signal_count": 1,
    },
}

# ── CONTEXTUAL ENTITY MAPPINGS ──
# Prevent keyword traps by mapping ambiguous terms to correct domains

ENTITY_MAPPINGS = {
    # NLP / Text processing terms
    "prompt": {
        "correct_domain": ["G06F", "G06N"],
        "wrong_domain": ["G10L"],
        "context": "text nlp query input",
        "wrong_context": "speech utterance command",
    },
    "utterance": {
        "correct_domain": ["G10L"],
        "wrong_domain": ["G06F", "G06N"],
        "context": "speech audio vocal",
        "wrong_context": "text nlp statement",
    },
    "context": {
        "correct_domain": ["G06F", "G06N"],
        "wrong_domain": ["G10L"],
        "context": "text conversation history",
        "wrong_context": "audio acoustic environment",
    },
    "interaction": {
        "correct_domain": ["G06F"],
        "wrong_domain": ["G10L"],
        "context": "software system user interface",
        "wrong_context": "audio speech dialogue",
    },
    "response": {
        "correct_domain": ["G06F", "G06N"],
        "wrong_domain": ["G10L"],
        "context": "text output reply",
        "wrong_context": "audio playback",
    },
    "query": {
        "correct_domain": ["G06F", "G06N"],
        "wrong_domain": ["G10L"],
        "context": "text search question",
        "wrong_context": "voice command",
    },
    "dialogue": {
        "correct_domain": ["G06F", "G06N"],
        "wrong_domain": ["G10L"],
        "context": "text conversation chat",
        "wrong_context": "spoken conversation",
    },
    "token": {
        "correct_domain": ["G06N", "G06F"],
        "wrong_domain": ["G10L"],
        "context": "text unit nlp",
        "wrong_context": "speech unit phoneme",
    },
    "embedding": {
        "correct_domain": ["G06N"],
        "wrong_domain": ["G10L"],
        "context": "vector representation neural",
        "wrong_context": "audio feature extraction",
    },
    "encoding": {
        "correct_domain": ["G06F", "G06N", "H04L"],
        "wrong_domain": [],
        "context": "data representation compression",
        "wrong_context": "audio codec",
    },
}

# ── ANTI-DOMAIN COLLAPSE RULES ──
# Strong penalties for cross-domain leakage

ANTI_COLLAPSE_RULES = {
    "G10L": {
        "description": "Speech/Audio domain requires explicit audio/speech signals",
        "required_context": [
            "audio",
            "sound",
            "speech",
            "voice",
            "acoustic",
            "microphone",
            "speaker",
        ],
        "penalty_multiplier": 0.05,  # Very strong penalty if audio context missing
        "boost_multiplier": 2.0,  # Strong boost if audio context present
    },
    "G06T": {
        "description": "Image/Video domain requires explicit visual signals",
        "required_context": [
            "image",
            "pixel",
            "visual",
            "graphics",
            "frame",
            "camera",
            "photo",
            "video",
        ],
        "penalty_multiplier": 0.1,
        "boost_multiplier": 2.0,
    },
    "G06V": {
        "description": "Computer vision requires explicit vision signals",
        "required_context": [
            "vision",
            "object detection",
            "image recognition",
            "pattern recognition",
        ],
        "penalty_multiplier": 0.1,
        "boost_multiplier": 2.0,
    },
    "G01": {
        "description": "Measurement requires explicit sensor/measurement signals",
        "required_context": [
            "sensor",
            "measure",
            "detect",
            "gauge",
            "instrument",
            "metrology",
        ],
        "penalty_multiplier": 0.1,
        "boost_multiplier": 2.0,
    },
    "H01L": {
        "description": "Semiconductors require explicit chip/circuit signals",
        "required_context": [
            "semiconductor",
            "transistor",
            "integrated circuit",
            "chip",
            "wafer",
        ],
        "penalty_multiplier": 0.1,
        "boost_multiplier": 2.0,
    },
}


# ── CONTEXTUAL ENTITY CONSISTENCY TABLE (Prompt 3 Fix) ──
# Extended entity mappings for multi-layer patent validation

CONTEXTUAL_ENTITY_TABLE = {
    # Vehicle + Database combinations
    ("vehicle state", "database query"): {
        "anchor": ["G06F", "B60W"],
        "block": ["B60Q", "H02"],
        "description": "Vehicle state queries map to data/control, NOT signaling/power",
    },
    ("vehicle", "database"): {
        "anchor": ["G06F16", "B60W"],
        "block": ["B60Q"],
        "description": "Vehicle with database → data management + control",
    },
    ("driving state", "knowledge graph"): {
        "anchor": ["G06F16", "B60W"],
        "block": ["B60Q", "H02"],
        "description": "Driving state with knowledge graph → G06F16 data layer",
    },
    # LLM + Speech combinations
    ("llm", "speech"): {
        "anchor": ["G10L", "G06N"],
        "block": [],
        "description": "LLM for speech → speech recognition + NLP",
    },
    ("large language model", "voice"): {
        "anchor": ["G10L", "G06F40"],
        "block": [],
        "description": "Voice + LLM → speech + text processing",
    },
    ("neural network", "utterance"): {
        "anchor": ["G06N", "G10L"],
        "block": [],
        "description": "Neural network processing utterance → G06N + G10L",
    },
    # Control + Software combinations
    ("control", "graph"): {
        "anchor": ["G05B", "G06F16"],
        "block": ["H02"],
        "description": "Control with graph data → control orchestration + data",
    },
    ("state", "management"): {
        "anchor": ["G05B", "G06F"],
        "block": ["H02P"],
        "description": "State management → control/software, NOT motor switches",
    },
    # Window/Vehicle control specific
    ("window", "controlling"): {
        "anchor": ["B60W"],
        "block": ["B60Q", "H02"],
        "description": "Window controlling → B60W vehicle control, NOT signaling/power",
    },
    ("window", "vehicle"): {
        "anchor": ["B60W"],
        "block": ["B60Q"],
        "description": "Vehicle window → B60W, NOT B60Q signaling",
    },
}


# ── LAYER-BASED VALIDATION (Prompt 3 Fix) ──
# Each technical layer requires specific domain signals

LAYER_DOMAIN_REQUIREMENTS = {
    "application": {
        "min_signals": 2,
        "signal_keywords": [
            "vehicle",
            "automotive",
            "car",
            "medical",
            "healthcare",
            "industrial",
            "manufacturing",
            "agricultural",
            "financial",
        ],
        "required_families": ["B60", "A61", "B23", "A01", "G06Q"],
    },
    "data_reasoning": {
        "min_signals": 1,
        "signal_keywords": [
            "database",
            "graph",
            "query",
            "SPARQL",
            "ontology",
            "knowledge",
            "semantic",
            "data storage",
            "retrieval",
        ],
        "required_families": ["G06F16", "G06F40", "G06F18"],
    },
    "interaction": {
        "min_signals": 2,
        "signal_keywords": [
            "speech",
            "voice",
            "audio",
            "NLP",
            "text",
            "LLM",
            "dialogue",
            "conversation",
            "utterance",
            "display",
        ],
        "required_families": ["G10L", "G06F40", "G06N", "G06K"],
    },
    "control": {
        "min_signals": 1,
        "signal_keywords": [
            "control",
            "orchestration",
            "scheduling",
            "state",
            "management",
            "automation",
            "feedback",
            "workflow",
        ],
        "required_families": ["G05B", "G05D", "B60W"],
    },
}


class CrossDomainValidator:
    """
    Phase 3.6: Cross-Domain Validation Layer.

    Prevents domain collapse by validating CPC candidates against
    actual domain signals and contextual entity consistency.
    """

    def __init__(self):
        self.rules_log: List[Dict[str, Any]] = []

    def validate(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        phase35_result: Dict[str, Any] = None,
        layer_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Validate CPC candidates against domain signals.

        Args:
            candidates: CPC candidates from Phase 3
            phase1_data: Phase 1 output
            phase35_result: Optional Phase 3.5 output
            layer_data: Optional Phase 2A layer decomposition (for layer-aware validation)

        Returns:
            Dict with validated candidates, rejected candidates, and rules log.
        """
        if not candidates:
            return self._empty_result()

        self.rules_log = []
        candidates = [dict(c) for c in candidates]

        # Step 1: Domain Anchor Check
        candidates = self._domain_anchor_check(candidates, phase1_data)

        # Step 2: Anti-Domain Collapse
        candidates = self._anti_domain_collapse(candidates, phase1_data)

        # Step 3: Contextual Entity Consistency
        candidates = self._entity_consistency_check(candidates, phase1_data)

        # Step 3.5: Contextual Entity Consistency Table (Prompt 3 Fix)
        candidates = self._contextual_entity_table_check(candidates, phase1_data)

        # Step 4: Layer-based validation (if layer_data provided)
        if layer_data:
            candidates = self._layer_based_validation(
                candidates, phase1_data, layer_data
            )

        # Step 5: Final Family Lock (≥2 signals required)
        candidates = self._final_family_lock(candidates, phase1_data)

        # Sort by score
        candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        return {
            "phase36_candidates": candidates,
            "phase36_rules_log": self.rules_log,
            "phase36_adjustments": len(self.rules_log),
            "phase36_domain_verified": True,
            "phase36_contextual_check_applied": True,
        }

    def _extract_family(self, symbol: str) -> str:
        """Extract CPC family code (e.g., G06N from G06N3/045)."""
        # CPC family is first 4 characters: Section(1) + Class(2) + Subclass(1)
        # e.g., G06N, G10L, H04L, A61B, F16J
        if len(symbol) >= 4:
            return symbol[:4]
        return symbol[:3] if len(symbol) >= 3 else symbol

    def _domain_anchor_check(
        self, candidates: List[Dict[str, Any]], phase1_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Verify each candidate's family has supporting domain signals.
        """
        # Extract all text from Phase 1
        all_text = self._extract_all_text(phase1_data)

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            # Get requirements for this family
            req = DOMAIN_REQUIREMENTS.get(family)
            if not req:
                adjusted.append(candidate)
                continue

            # Count matching signals
            matched_signals = [
                sig for sig in req["required_signals"] if sig.lower() in all_text
            ]
            signal_count = len(matched_signals)

            if signal_count >= req["min_signal_count"]:
                # Domain anchor confirmed
                old_score = score
                score *= 1.2
                self._log_rule(
                    "DOMAIN_ANCHOR_CONFIRMED",
                    symbol,
                    old_score,
                    score,
                    f"{family}: {signal_count} domain signals matched",
                )
            else:
                # Domain anchor weak
                old_score = score
                score *= 0.5
                self._log_rule(
                    "DOMAIN_ANCHOR_WEAK",
                    symbol,
                    old_score,
                    score,
                    f"{family}: only {signal_count}/{req['min_signal_count']} signals",
                )

            candidate["score"] = round(score, 4)
            candidate["domain_signals_matched"] = signal_count
            adjusted.append(candidate)

        return adjusted

    def _anti_domain_collapse(
        self, candidates: List[Dict[str, Any]], phase1_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Step 2: Apply anti-domain collapse rules.
        Prevent G10L without audio, G06T without image, etc.
        """
        all_text = self._extract_all_text(phase1_data)

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            rule = ANTI_COLLAPSE_RULES.get(family)
            if not rule:
                adjusted.append(candidate)
                continue

            # Check if required context exists
            has_context = any(ctx in all_text for ctx in rule["required_context"])

            if has_context:
                # Context present - boost
                old_score = score
                score *= rule["boost_multiplier"]
                self._log_rule(
                    "ANTI_COLLAPSE_BOOST",
                    symbol,
                    old_score,
                    score,
                    f"{family}: domain context confirmed",
                )
            else:
                # Context missing - heavy penalty
                old_score = score
                score *= rule["penalty_multiplier"]
                self._log_rule(
                    "ANTI_COLLAPSE_PENALTY",
                    symbol,
                    old_score,
                    score,
                    f"{family}: NO domain context (requires: {rule['required_context'][:3]})",
                )

            candidate["score"] = round(score, 4)
            candidate["domain_context_verified"] = has_context
            adjusted.append(candidate)

        return adjusted

    def _entity_consistency_check(
        self, candidates: List[Dict[str, Any]], phase1_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Check contextual entity consistency.
        Prevent 'prompt' → G10L (speech), force 'prompt' → G06F (NLP).
        """
        all_text = self._extract_all_text(phase1_data)

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            # Check each entity mapping
            for entity, mapping in ENTITY_MAPPINGS.items():
                if entity in all_text:
                    if family in mapping["wrong_domain"]:
                        # Wrong domain for this entity
                        old_score = score
                        score *= 0.2
                        self._log_rule(
                            "ENTITY_MISMATCH",
                            symbol,
                            old_score,
                            score,
                            f"'{entity}' → {family} is wrong (should be {mapping['correct_domain']})",
                        )
                    elif family in mapping["correct_domain"]:
                        # Correct domain for this entity
                        old_score = score
                        score *= 1.3
                        self._log_rule(
                            "ENTITY_MATCH",
                            symbol,
                            old_score,
                            score,
                            f"'{entity}' → {family} is correct",
                        )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _final_family_lock(
        self, candidates: List[Dict[str, Any]], phase1_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Final family lock requiring ≥2 independent signals.
        """
        all_text = self._extract_all_text(phase1_data)

        # Count signals per family
        family_signals = {}
        for family, req in DOMAIN_REQUIREMENTS.items():
            matched = [s for s in req["required_signals"] if s.lower() in all_text]
            family_signals[family] = matched

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            signals = family_signals.get(family, [])
            if len(signals) >= 2:
                # Strong lock
                old_score = score
                score *= 1.1
                self._log_rule(
                    "FAMILY_LOCKED",
                    symbol,
                    old_score,
                    score,
                    f"{family}: {len(signals)} independent signals",
                )
            elif len(signals) == 1:
                # Weak lock
                old_score = score
                score *= 0.9
                self._log_rule(
                    "FAMILY_WEAK",
                    symbol,
                    old_score,
                    score,
                    f"{family}: only 1 signal",
                )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _contextual_entity_table_check(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Step 3.5: CONTEXTUAL ENTITY CONSISTENCY TABLE (Prompt 3 Fix).

        Apply layer-specific rules to anchor/block CPC families.
        - "Vehicle State" + "Database Query" → anchor G06F/B60W, block B60Q
        - "LLM" + "Speech" → anchor G10L/G06N
        - "Window" + "Controlling" → anchor B60W, block B60Q/H02
        """
        all_text = self._extract_all_text(phase1_data)
        all_text_lower = all_text.lower()

        # Check if any contextual entity patterns match
        matched_rules = []
        for pattern_keys, rule_config in CONTEXTUAL_ENTITY_TABLE.items():
            key1, key2 = pattern_keys
            if key1 in all_text_lower and key2 in all_text_lower:
                matched_rules.append((pattern_keys, rule_config))
                logger.info(
                    "CONTEXTUAL_ENTITY: Pattern (%s, %s) matched - %s",
                    key1,
                    key2,
                    rule_config["description"],
                )

        if not matched_rules:
            return candidates

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            # Apply matched rules
            for pattern_keys, rule_config in matched_rules:
                anchors = rule_config["anchor"]
                blocks = rule_config["block"]

                # Anchor boost
                if any(family.startswith(a) for a in anchors):
                    old_score = score
                    score *= 1.3
                    self._log_rule(
                        "CONTEXTUAL_ANCHOR",
                        symbol,
                        old_score,
                        score,
                        f"Pattern {pattern_keys} → {rule_config['description']}",
                    )
                    break

                # Block penalty
                if any(family.startswith(b) for b in blocks):
                    old_score = score
                    score *= 0.2  # Strong penalty
                    self._log_rule(
                        "CONTEXTUAL_BLOCK",
                        symbol,
                        old_score,
                        score,
                        f"Blocked by pattern {pattern_keys}: {rule_config['description']}",
                    )
                    break

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _layer_based_validation(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        layer_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Step 4: LAYER-BASED VALIDATION (Prompt 3 Fix).

        Validate candidates against layer-specific requirements.
        Each layer requires specific domain signals for verification.
        """
        layers = layer_data.get("layers", {})
        all_text = self._extract_all_text(phase1_data)

        # Detect which layers are active
        active_layers = set()
        for layer_name, layer_candidates in layers.items():
            if layer_candidates:
                active_layers.add(layer_name)

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            family = self._extract_family(symbol)

            # Check if candidate matches any active layer
            layer_matched = False
            for layer_name in active_layers:
                layer_config = LAYER_DOMAIN_REQUIREMENTS.get(layer_name, {})
                if not layer_config:
                    continue

                required_families = layer_config.get("required_families", [])

                # Check if candidate family matches this layer's requirements
                if any(family.startswith(rf) for rf in required_families):
                    layer_matched = True

                    # Check signal requirements for this layer
                    signal_keywords = layer_config.get("signal_keywords", [])
                    min_signals = layer_config.get("min_signals", 1)
                    matched_signals = sum(1 for kw in signal_keywords if kw in all_text)

                    if matched_signals >= min_signals:
                        # Validated
                        old_score = score
                        score *= 1.1
                        self._log_rule(
                            "LAYER_VALIDATED",
                            symbol,
                            old_score,
                            score,
                            f"{layer_name} layer: {matched_signals} signals (min {min_signals})",
                        )
                    else:
                        # Insufficient signals for this layer
                        old_score = score
                        score *= 0.7
                        self._log_rule(
                            "LAYER_UNVERIFIED",
                            symbol,
                            old_score,
                            score,
                            f"{layer_name} layer: only {matched_signals} signals (need {min_signals})",
                        )
                    break

            # Mark as UNVERIFIED if no layer match
            if not layer_matched:
                candidate["domain_context_verified"] = False

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _extract_all_text(self, phase1_data: Dict[str, Any]) -> str:
        """Extract all text from Phase 1 data."""
        parts = []

        # Core fields
        for field in ["technical_object", "core_function", "system_context"]:
            val = phase1_data.get(field, "")
            if val:
                parts.append(val.lower())

        # Terms
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        for t in terms:
            if isinstance(t, dict):
                parts.append(t.get("term", "").lower())
            else:
                parts.append(str(t).lower())

        # Domain signals
        domain_signals = phase1_data.get("domain_signals", [])
        for ds in domain_signals:
            if isinstance(ds, dict):
                parts.append(ds.get("name", "").lower())

        return " ".join(parts)

    def _log_rule(
        self, rule: str, symbol: str, before: float, after: float, reason: str
    ) -> None:
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
        return {
            "phase36_candidates": [],
            "phase36_rules_log": [],
            "phase36_adjustments": 0,
            "phase36_domain_verified": False,
        }


def apply_cross_domain_validation(
    candidates: List[Dict[str, Any]],
    phase1_data: Dict[str, Any],
    phase35_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Quick function to apply cross-domain validation."""
    validator = CrossDomainValidator()
    return validator.validate(candidates, phase1_data, phase35_result)
