"""
cpc_role_classifier.py - Phase 1.5: Structural role tagging.

Outputs role + confidence only. Machine-readable. No explanatory text.
"""

import logging
from typing import Dict, Any, List, Optional
from search_core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ── Inventive contribution signals (determines role label) ──
INVENTIVE_CONTRIBUTION_SIGNALS = {
    "CORE_TECH": [
        "training",
        "clustering",
        "initializ",
        "optimiz",
        "classif",
        "self-supervised",
        "pre-training",
        "fine-tuning",
        "embedding",
        "neural network",
        "transformer",
        "attention mechanism",
        "novel method",
        "new approach",
        "proposed algorithm",
        "loss function",
        "objective function",
        "gradient",
    ],
    "SYSTEM": [
        "communicat",
        "coordinat",
        "orchestrat",
        "schedul",
        "load balanc",
        "failover",
        "routing",
        "distribut",
        "pipeline management",
        "workflow",
    ],
    "APPLICATION": [
        "user interface",
        "end-user",
        "application layer",
        "business process",
        "service platform",
    ],
    "SUPPORT": [
        "monitor",
        "log",
        "diagnos",
        "test",
        "evaluat",
        "benchmark",
        "report",
    ],
}


class CPCRoleClassifier:
    """Phase 1.5: Structural role tagging. Machine-readable only."""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm

    def classify_role(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        if not phase1_data:
            return self._default_result("SYSTEM", 0.5)

        technical_object = phase1_data.get("technical_object", "")
        problem_solved = phase1_data.get("problem_solved", "")
        core_function = phase1_data.get("core_function", "")

        # "claims" is a list in phase1; "claims_text" key does not exist
        claims_raw = phase1_data.get("claims", phase1_data.get("claims_text", ""))
        if isinstance(claims_raw, list):
            claims_text = " ".join(str(c) for c in claims_raw)
        else:
            claims_text = str(claims_raw) if claims_raw else ""

        combined_text = (
            f"{technical_object} {core_function} {claims_text} {problem_solved}"
        ).lower()

        scores = {role: 0 for role in INVENTIVE_CONTRIBUTION_SIGNALS}

        for role, signals in INVENTIVE_CONTRIBUTION_SIGNALS.items():
            for signal in signals:
                if signal.lower() in combined_text:
                    scores[role] += 1

        top_role = max(scores, key=scores.get)
        top_score = scores[top_role]
        total_signals = sum(scores.values())

        raw_confidence = top_score / total_signals if total_signals > 0 else 0.5

        # Only cap when total signal count is low AND consensus is weak.
        # 100% consensus (all signals point to one role) is still informative.
        if total_signals < 3 and top_score < total_signals:
            raw_confidence = min(raw_confidence, 0.75)

        raw_confidence = min(raw_confidence, 0.92)

        # Build reasoning from top matched signals
        matched = [
            sig for sig in INVENTIVE_CONTRIBUTION_SIGNALS[top_role]
            if sig.lower() in combined_text
        ]
        reasoning = (
            f"{top_role} selected: {top_score}/{total_signals} signals matched "
            f"({', '.join(matched[:5])})"
        )

        result = {
            "role": top_role,
            "confidence": round(raw_confidence, 2),
            "reasoning": reasoning,
            "signal_scores": scores,
        }

        logger.info("Phase 1.5: Role=%s (conf=%.2f) — %s", top_role, raw_confidence, reasoning)
        return result

    def _format_terms(self, terms: List[Any], max_terms: int = 15) -> str:
        if not terms:
            return "No terms extracted"

        lines = []
        for t in terms[:max_terms]:
            if isinstance(t, dict):
                term = t.get("term", "")
                importance = t.get("importance", 5)
                source = t.get("source_section", t.get("source", "unknown"))
                lines.append(f"- {term} (importance: {importance}, source: {source})")
            elif isinstance(t, str):
                lines.append(f"- {t}")

        return "\n".join(lines) if lines else "No terms extracted"

    def _rule_based_classification(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        combined_text = " ".join(
            [
                phase1_data.get("technical_object", "").lower(),
                phase1_data.get("core_function", "").lower(),
                phase1_data.get("system_context", "").lower(),
                *[
                    t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                    for t in (
                        phase1_data.get("terms", phase1_data.get("essential_terms", []))
                    )
                ],
            ]
        )

        scores = {"CORE_TECH": 0.0, "SYSTEM": 0.0, "APPLICATION": 0.0, "SUPPORT": 0.0}

        for signal in [
            "algorithm",
            "architecture",
            "layer",
            "training method",
            "optimization method",
            "weight",
            "parameter",
            "quantization",
            "pruning",
            "distillation",
            "compression algorithm",
            "encoding scheme",
            "filter design",
            "model structure",
            "network architecture",
            "activation function",
            "loss function",
            "gradient",
            "backpropagation",
            "neural network layer",
        ]:
            if signal in combined_text:
                scores["CORE_TECH"] += 1.0

        for signal in [
            "pipeline",
            "orchestrating",
            "coordinating",
            "workflow",
            "data flow",
            "control flow",
            "component interaction",
            "multi-component",
            "interface",
            "API",
            "protocol",
            "routing",
            "dispatching",
            "orchestration",
        ]:
            if signal in combined_text:
                scores["SYSTEM"] += 1.0

        for signal in [
            "medical",
            "healthcare",
            "diagnosis",
            "treatment",
            "patient",
            "automotive",
            "vehicle",
            "driving",
            "financial",
            "banking",
            "industrial",
            "manufacturing",
            "energy",
        ]:
            if signal in combined_text:
                scores["APPLICATION"] += 1.0

        for signal in [
            "logging",
            "monitoring",
            "tracking",
            "visualization",
            "user interface",
            "display",
            "dashboard",
            "storing",
            "caching",
            "configuration",
            "metadata",
        ]:
            if signal in combined_text:
                scores["SUPPORT"] += 1.0

        for signal in [
            "using a neural network",
            "using machine learning",
            "applying neural network",
            "deploying AI",
        ]:
            if signal in combined_text:
                scores["CORE_TECH"] -= 1.5
                scores["SYSTEM"] += 0.5
                scores["APPLICATION"] += 0.5

        role, top_score = max(scores.items(), key=lambda x: x[1])
        if top_score < 0.5:
            role = "SYSTEM"
            confidence = 0.5
        else:
            confidence = min(top_score / max(sum(scores.values()), 1), 1.0)

        return {"role": role, "confidence": round(confidence, 2)}

    def _default_result(self, role: str, confidence: float) -> Dict[str, Any]:
        return {"role": role, "confidence": confidence}


def get_role_family_boosts(role: str) -> Dict[str, float]:
    return {}


def apply_role_scoring(
    family_scores: Dict[str, float], role: str, phase1_data: Dict[str, Any]
) -> Dict[str, float]:
    return family_scores
