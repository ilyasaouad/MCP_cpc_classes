"""
Test Phase 3.5 hierarchy priority (intra-domain fix).

Tests that structural/optimization signals select correct subgroups
within a domain, preventing drift to semantic/usage subgroups.

Example: quantization should go to G06N3/063, not G06N5/04.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_decision_tree import CPCDecisionTreeConstraint


def test_g06n_hierarchy_quantization():
    """Test quantization patent selects G06N3/063, not G06N5/04."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.95,
        },
        "terms": [
            {"term": "weight clipping", "importance": 10},
            {"term": "quantization", "importance": 10},
            {"term": "model compression", "importance": 9},
            {"term": "low-bit precision", "importance": 8},
            {"term": "inference latency", "importance": 6},  # Should NOT dominate
        ],
        "core_function": "Neural network model compression through weight clipping and quantization for efficient inference",
    }

    candidates = [
        {
            "symbol": "G06N3/063",
            "title": "Model compression/quantization",
            "score": 0.75,
        },
        {"symbol": "G06N3/045", "title": "Parameter optimization", "score": 0.70},
        {
            "symbol": "G06N3/043",
            "title": "Fuzzy logic / neuro-fuzzy",
            "score": 0.85,
        },  # Should be penalized - wrong contribution type
        {
            "symbol": "G06N5/04",
            "title": "Inference models",
            "score": 0.80,
        },  # Should be penalized
        {"symbol": "G06N5/08", "title": "Reasoning systems", "score": 0.65},
        {"symbol": "G06N3/08", "title": "General neural networks", "score": 0.60},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 70)
    print("TEST: G06N Hierarchy - Quantization Patent")
    print("=" * 70)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    print(
        f"Detected domain: {result['phase35_domain']} (conf={result['phase35_domain_confidence']})"
    )
    print(f"Rules applied: {result['phase35_adjustments']}")
    print()

    # G06N3/063 (Level 2 - compression) should be highest
    assert scores["G06N3/063"] >= scores["G06N5/04"], (
        f"FAIL: G06N3/063 ({scores['G06N3/063']}) should dominate "
        f"G06N5/04 ({scores['G06N5/04']})"
    )

    # G06N3/045 (Level 1 - optimization) should also be high
    assert scores["G06N3/045"] >= scores["G06N5/04"], (
        f"FAIL: G06N3/045 ({scores['G06N3/045']}) should dominate "
        f"G06N5/04 ({scores['G06N5/04']})"
    )

    # G06N5/04 (Level 4 - inference) should be heavily penalized
    assert scores["G06N5/04"] < 0.5, (
        f"FAIL: G06N5/04 should be penalized, got {scores['G06N5/04']}"
    )

    print("[OK] PASS: Hierarchy correctly prioritizes structural over semantic")
    for c in adjusted:
        level = "?"
        if c["symbol"].startswith("G06N3/045"):
            level = "L1-opt"
        elif c["symbol"].startswith("G06N3/063"):
            level = "L2-comp"
        elif c["symbol"].startswith("G06N3/08"):
            level = "L3-gen"
        elif c["symbol"].startswith("G06N5/04"):
            level = "L4-inf"
        elif c["symbol"].startswith("G06N5/08"):
            level = "L4-reason"
        print(f"  {c['symbol']}: {c['score']:.4f} [{level}] - {c['title']}")
    print()

    # Show hierarchy and contribution rules
    for rule in result["phase35_rules_log"]:
        if "HIERARCHY" in rule["rule"] or "CONTRIBUTION" in rule["rule"]:
            print(
                f"  {rule['rule']}: {rule['symbol']} {rule['score_before']:.3f} -> {rule['score_after']:.3f}"
            )
    print()


def test_g06n_contribution_filter():
    """Test contribution filter penalizes fuzzy logic when quantization detected."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.95,
        },
        "terms": [
            {"term": "quantization", "importance": 10},
            {"term": "model compression", "importance": 9},
        ],
        "core_function": "Quantize neural network weights for efficient deployment",
    }

    candidates = [
        {"symbol": "G06N3/063", "title": "Quantization", "score": 0.7},
        {"symbol": "G06N3/045", "title": "Optimization", "score": 0.6},
        {
            "symbol": "G06N3/043",
            "title": "Fuzzy logic",
            "score": 0.9,
        },  # Should be penalized
        {"symbol": "G06N3/08", "title": "General NN", "score": 0.5},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 70)
    print("TEST: G06N Contribution Filter")
    print("=" * 70)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06N3/063 should dominate G06N3/043
    assert scores["G06N3/063"] > scores["G06N3/043"], (
        f"FAIL: G06N3/063 ({scores['G06N3/063']}) should exceed "
        f"G06N3/043 ({scores['G06N3/043']})"
    )

    # Check that contribution filter rules were applied
    contrib_rules = [
        r for r in result["phase35_rules_log"] if "CONTRIBUTION" in r["rule"]
    ]
    assert len(contrib_rules) > 0, "FAIL: No contribution filter rules applied"

    print("[OK] PASS: Contribution filter correctly penalizes fuzzy logic")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()

    for rule in contrib_rules:
        print(
            f"  {rule['rule']}: {rule['symbol']} {rule['score_before']:.3f} -> {rule['score_after']:.3f}"
        )
    print()


def test_g06t_hierarchy_filtering():
    """Test image filtering patent selects G06T5/00, not G06T15/00."""
    phase1 = {
        "primary_domain": {
            "name": "Image Processing",
            "cpc_class": "G06T",
            "confidence": 0.9,
        },
        "terms": [
            {"term": "image filtering", "importance": 10},
            {"term": "edge detection", "importance": 9},
            {"term": "denoising", "importance": 8},
        ],
        "core_function": "Filter and enhance images using edge-preserving denoising",
    }

    candidates = [
        {"symbol": "G06T5/00", "title": "Image filtering/enhancement", "score": 0.7},
        {"symbol": "G06T7/00", "title": "Image analysis", "score": 0.6},
        {"symbol": "G06T15/00", "title": "Rendering/graphics", "score": 0.8},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 70)
    print("TEST: G06T Hierarchy - Image Filtering")
    print("=" * 70)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06T5/00 (Level 2 - filtering) should be highest
    assert scores["G06T5/00"] >= scores["G06T15/00"], (
        f"FAIL: G06T5/00 ({scores['G06T5/00']}) should dominate "
        f"G06T15/00 ({scores['G06T15/00']})"
    )

    print("[OK] PASS: Hierarchy correctly prioritizes filtering over rendering")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()


def test_h04l_hierarchy_error_correction():
    """Test telecom patent selects H04L1/00, not H04L67/00."""
    phase1 = {
        "primary_domain": {
            "name": "Telecommunications",
            "cpc_class": "H04L",
            "confidence": 0.9,
        },
        "terms": [
            {"term": "error correction", "importance": 10},
            {"term": "channel coding", "importance": 9},
            {"term": "forward error correction", "importance": 8},
        ],
        "core_function": "Improve data reliability through advanced channel coding",
    }

    candidates = [
        {"symbol": "H04L1/00", "title": "Error correction/coding", "score": 0.7},
        {"symbol": "H04L29/00", "title": "Network protocols", "score": 0.75},
        {"symbol": "H04L67/00", "title": "Network services", "score": 0.8},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 70)
    print("TEST: H04L Hierarchy - Error Correction")
    print("=" * 70)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # H04L1/00 (Level 2 - error correction) should dominate H04L67 (Level 4)
    assert scores["H04L1/00"] >= scores["H04L67/00"], (
        f"FAIL: H04L1/00 ({scores['H04L1/00']}) should dominate "
        f"H04L67/00 ({scores['H04L67/00']})"
    )

    print("[OK] PASS: Hierarchy correctly prioritizes error correction over services")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING HIERARCHY PRIORITY TESTS")
    print("=" * 70 + "\n")

    test_g06n_hierarchy_quantization()
    test_g06n_contribution_filter()
    test_g06t_hierarchy_filtering()
    test_h04l_hierarchy_error_correction()

    print("=" * 70)
    print("ALL HIERARCHY TESTS PASSED [OK]")
    print("=" * 70)
