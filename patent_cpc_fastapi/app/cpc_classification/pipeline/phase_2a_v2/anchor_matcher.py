"""
anchor_matcher.py — Anchor-to-CPC mapping layer.

Deterministic mapping from Phase 1 anchors (structural, action, domain)
to CPC family scores using keyword and label matching against KG metadata.
No LLM — purely deterministic.
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# ── Keyword → CPC family mapping (targeted, non-exhaustive) ──
_KEYWORD_TO_CPC: Dict[str, str] = {
    # Speech / Audio → G10L
    "speech": "G10L",
    "audio": "G10L",
    "acoustic": "G10L",
    "phoneme": "G10L",
    "voice": "G10L",
    "utterance": "G10L",
    "speaker": "G10L",
    "mfcc": "G10L",
    "spectral": "G10L",
    # Neural networks / ML → G06N
    "neural network": "G06N",
    "training": "G06N",
    "inference": "G06N",
    "lstm": "G06N",
    "transformer": "G06N",
    "encoder": "G06N",
    "decoder": "G06N",
    "attention": "G06N",
    "embedding": "G06N",
    "gradient": "G06N",
    "backpropagation": "G06N",
    "clustering": "G06N",
    "classification": "G06N",
    "regression": "G06N",
    "self-supervised": "G06N",
    "pre-training": "G06N",
    "loss function": "G06N",
    "optimization": "G06N",
    # Image / Video → G06T / G06V
    "image": "G06T",
    "pixel": "G06T",
    "camera": "G06T",
    "visual": "G06T",
    "photograph": "G06T",
    "video": "G06V",
    "frame": "G06V",
    # Computing / Data processing → G06F
    "processor": "G06F",
    "memory": "G06F",
    "database": "G06F",
    "program": "G06F",
    "interface": "G06F",
    "computing": "G06F",
    "storage": "G06F",
    "cache": "G06F",
    # Networking → H04L
    "network": "H04L",
    "packet": "H04L",
    "transmission": "H04L",
    "protocol": "H04L",
    "routing": "H04L",
    # Wireless → H04W
    "wireless": "H04W",
    "radio": "H04W",
    "antenna": "H04W",
    "base station": "H04W",
    # Medical → A61B
    "patient": "A61B",
    "medical": "A61B",
    "diagnosis": "A61B",
    "surgical": "A61B",
    # Vehicle → B60W
    "vehicle": "B60W",
    "driving": "B60W",
    "steering": "B60W",
    # Electric → B60L
    "battery": "B60L",
    "motor": "B60L",
    "propulsion": "B60L",
    # Control → G05B
    "control system": "G05B",
    "automation": "G05B",
    "actuator": "G05B",
    # Mechanical → F16J/F16K
    "seal": "F16J",
    "valve": "F16K",
    "fluid": "F16K",
}


def extract_anchors(phase1: dict) -> dict:
    """
    Extract high-quality structural, action, and domain anchors.
    Prioritizes disambiguated audit terms over raw evidence terms.
    """
    anchors = {
        "structural": [],
        "actions": [],
        "domain": [],
    }
    seen: Set[str] = set()

    # Priority 1: Disambiguated terms (system of record)
    for dt in phase1.get("disambiguated_terms", []):
        if not isinstance(dt, dict):
            continue
        term = dt.get("term", "").strip().lower()
        if not term or len(term) < 3:
            continue
        
        seen.add(term)
        category = dt.get("category", "").lower()
        if category == "structural":
            anchors["structural"].append(term)
        elif category in ("functional", "action"):
            anchors["actions"].append(term)
        elif category in ("legal", "domain"):
            anchors["domain"].append(term)

    # Priority 2: High-importance evidence terms (validated anchors)
    terms = phase1.get("terms", phase1.get("essential_terms", []))
    for t in terms:
        if not isinstance(t, dict):
            continue
        term = t.get("term", "").strip().lower()
        # Stricter filter for evidence-based anchors to avoid noise
        if not term or len(term) < 4 or term in seen:
            continue
            
        importance = t.get("importance", t.get("weight", 5))
        if importance < 8:
            continue
            
        seen.add(term)
        source = t.get("source_section", t.get("source", "")).lower()
        if "claim" in source:
            anchors["domain"].append(term)
        else:
            # High-weight non-claim terms are usually functional steps
            anchors["actions"].append(term)

    return anchors


def match_anchors_to_cpc(anchors: dict) -> Dict[str, float]:
    """
    Map anchors to CPC family scores using keyword matching.

    Each matched anchor contributes to the family's score.
    Multi-map is allowed (e.g., "clustering" → G06N and G10L).

    Returns: {cpc_family: accumulated_score}
    """
    scores: Dict[str, float] = {}
    weights = {
        "domain": 1.0,
        "structural": 0.7,
        "actions": 0.5,
    }

    for anchor_type, anchor_list in anchors.items():
        weight = weights.get(anchor_type, 0.5)
        for anchor in anchor_list:
            for keyword, cpc_family in _KEYWORD_TO_CPC.items():
                if keyword in anchor:
                    scores[cpc_family] = scores.get(cpc_family, 0) + weight
                    # Don't break — multi-map allowed

    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        for family in scores:
            scores[family] = round(scores[family] / max_score, 3)

    return scores
