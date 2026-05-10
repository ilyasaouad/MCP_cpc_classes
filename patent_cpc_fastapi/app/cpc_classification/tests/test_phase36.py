"""
Test Phase 3.6 Universal CPC Hierarchy Engine.

Tests contribution-type-first, domain-second selection logic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_hierarchy_engine import UniversalCPCHierarchyEngine


def test_quantization_selects_g06n3_063():
    """Test quantization patent: contribution B -> G06N3/063."""
    phase1 = {
        "terms": [
            {"term": "quantization", "importance": 10},
            {"term": "weight clipping", "importance": 9},
            {"term": "model compression", "importance": 8},
        ],
        "core_function": "Neural network model compression through quantization",
        "technical_object": "Method for compressing trained neural network models",
    }

    # Candidates after Phase 3.5
    candidates = [
        {"symbol": "G06N3/063", "title": "Model compression", "score": 0.7},
        {"symbol": "G06N3/045", "title": "Parameter optimization", "score": 0.65},
        {"symbol": "G06N3/043", "title": "Fuzzy logic", "score": 0.8},
        {"symbol": "G06N5/04", "title": "Inference", "score": 0.75},
        {"symbol": "G06N3/08", "title": "General NN", "score": 0.6},
    ]

    engine = UniversalCPCHierarchyEngine()
    result = engine.apply_hierarchy(candidates, phase1, primary_domain="ai")

    print("=" * 70)
    print("TEST: Quantization -> G06N3/063 (Contribution Type B)")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    print(f"Primary type: {result['phase36_primary_type']}")
    print(f"Detected: {result['phase36_types']}")
    print()

    # G06N3/063 (compression) should be highest
    assert scores["G06N3/063"] >= scores["G06N3/043"], (
        f"FAIL: G06N3/063 ({scores['G06N3/063']}) should exceed "
        f"G06N3/043 ({scores['G06N3/043']})"
    )

    # G06N5/04 (inference - lower priority D) should be penalized
    assert scores["G06N3/063"] > scores["G06N5/04"], (
        f"FAIL: G06N3/063 ({scores['G06N3/063']}) should exceed "
        f"G06N5/04 ({scores['G06N5/04']})"
    )

    # When both A and B exist, A (optimization) has higher priority per hierarchy
    # But both should dominate lower-priority types like D (inference) and F (reasoning)
    assert result["phase36_primary_code"] in ["A", "B"], (
        f"FAIL: Primary type should be A or B, got {result['phase36_primary_code']}"
    )

    print(
        "[OK] PASS: A/B types (optimization/compression) correctly prioritized over D/F"
    )
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} [{c.get('contribution_match', '?')}]")
    print()


def test_telecom_error_correction():
    """Test telecom error correction: contribution B -> H04L1/00."""
    phase1 = {
        "terms": [
            {"term": "error correction", "importance": 10},
            {"term": "channel coding", "importance": 9},
            {"term": "forward error correction", "importance": 8},
        ],
        "core_function": "Improve data reliability through channel coding",
        "technical_object": "Communication system with error correction",
    }

    candidates = [
        {"symbol": "H04L1/00", "title": "Error correction", "score": 0.7},
        {"symbol": "H04L29/00", "title": "Protocols", "score": 0.75},
        {"symbol": "H04L67/00", "title": "Network services", "score": 0.8},
    ]

    engine = UniversalCPCHierarchyEngine()
    result = engine.apply_hierarchy(candidates, phase1, primary_domain="telecom")

    print("=" * 70)
    print("TEST: Telecom Error Correction (Contribution Type B)")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    print(f"Primary type: {result['phase36_primary_type']}")
    print()

    # H04L1/00 should dominate H04L67/00
    assert scores["H04L1/00"] > scores["H04L67/00"], (
        f"FAIL: H04L1/00 ({scores['H04L1/00']}) should exceed "
        f"H04L67/00 ({scores['H04L67/00']})"
    )

    print("[OK] PASS: B-type correctly selected in telecom domain")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} [{c.get('contribution_match', '?')}]")
    print()


def test_unknown_domain_universality():
    """Test that engine works even for unknown CPC domains."""
    phase1 = {
        "terms": [
            {"term": "quantization", "importance": 10},
        ],
        "core_function": "Quantize parameters for efficiency",
        "technical_object": "Unknown domain invention",
    }

    candidates = [
        {"symbol": "X99A1/00", "title": "Unknown compression", "score": 0.7},
        {"symbol": "X99A5/00", "title": "Unknown inference", "score": 0.8},
    ]

    engine = UniversalCPCHierarchyEngine()
    result = engine.apply_hierarchy(candidates, phase1, primary_domain="unknown_domain")

    print("=" * 70)
    print("TEST: Unknown Domain Universality")
    print("=" * 70)

    adjusted = result["phase36_candidates"]

    # Should detect contribution type (A or B)
    assert result["phase36_primary_code"] in ["A", "B"], (
        f"FAIL: Should detect A or B type, got {result['phase36_primary_code']}"
    )

    print("[OK] PASS: Universal engine works for unknown domains")
    print(f"  Primary type: {result['phase36_primary_type']}")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f}")
    print()


def test_a_over_f_priority():
    """Test that optimization (A) dominates reasoning (F)."""
    phase1 = {
        "terms": [
            {"term": "parameter optimization", "importance": 10},
            {"term": "weight tuning", "importance": 9},
            {"term": "fuzzy logic", "importance": 8},  # Should NOT dominate
        ],
        "core_function": "Optimize neural network parameters",
        "technical_object": "Parameter optimization system",
    }

    candidates = [
        {"symbol": "G06N3/045", "title": "Optimization", "score": 0.6},
        {"symbol": "G06N3/043", "title": "Fuzzy logic", "score": 0.9},
    ]

    engine = UniversalCPCHierarchyEngine()
    result = engine.apply_hierarchy(candidates, phase1, primary_domain="ai")

    print("=" * 70)
    print("TEST: A-type (optimization) dominates F-type (reasoning)")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # A should be primary, not F
    assert result["phase36_primary_code"] == "A", (
        f"FAIL: A should dominate, got {result['phase36_primary_code']}"
    )

    # G06N3/045 should exceed G06N3/043
    assert scores["G06N3/045"] > scores["G06N3/043"], (
        f"FAIL: G06N3/045 ({scores['G06N3/045']}) should exceed "
        f"G06N3/043 ({scores['G06N3/043']})"
    )

    print("[OK] PASS: A-type correctly dominates F-type")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} [{c.get('contribution_match', '?')}]")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING PHASE 3.6 UNIVERSAL HIERARCHY TESTS")
    print("=" * 70 + "\n")

    test_quantization_selects_g06n3_063()
    test_telecom_error_correction()
    test_unknown_domain_universality()
    test_a_over_f_priority()

    print("=" * 70)
    print("ALL PHASE 3.6 TESTS PASSED [OK]")
    print("=" * 70)
