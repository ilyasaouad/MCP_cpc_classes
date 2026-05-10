"""
Test Phase 3.5 Decision Tree Constraint Layer.

Tests the constraint layer with various domain scenarios.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_decision_tree import CPCDecisionTreeConstraint


def test_ai_domain_constraints():
    """Test AI patent gets G06N boost and G06T penalty."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.95,
        },
        "domain_signals": [
            {
                "name": "neural network",
                "confidence": 0.95,
                "cpc_family": "G06N",
                "role": "primary",
            },
        ],
        "terms": [
            {"term": "weight clipping", "importance": 10},
            {"term": "quantization", "importance": 10},
        ],
        "core_function": "Neural network parameter optimization via weight clipping and quantization",
    }

    candidates = [
        {"symbol": "G06N3/045", "title": "Parameter optimization", "score": 0.8},
        {"symbol": "G06N3/063", "title": "Model compression", "score": 0.75},
        {
            "symbol": "G06T15/30",
            "title": "Image clipping",
            "score": 0.7,
        },  # Should be penalized
        {
            "symbol": "G06F2207/00",
            "title": "BCD encoding",
            "score": 0.6,
        },  # Should be penalized
        {
            "symbol": "G10K11/00",
            "title": "Acoustic processing",
            "score": 0.5,
        },  # Should be penalized
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 60)
    print("TEST: AI Domain Constraints")
    print("=" * 60)
    print(
        f"Detected domain: {result['phase35_domain']} (conf={result['phase35_domain_confidence']})"
    )
    print(f"Rules applied: {result['phase35_adjustments']}")
    print()

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06N should be boosted (before normalization)
    # Check rules log for boost evidence
    rules = result["phase35_rules_log"]
    boost_rules = [
        r for r in rules if "G06N" in r["symbol"] and r["rule"] == "DOMAIN_BOOST"
    ]
    penalty_rules_g06t = [
        r for r in rules if "G06T" in r["symbol"] and r["rule"] == "DOMAIN_PENALTY"
    ]
    penalty_rules_g06f = [
        r for r in rules if "G06F2207" in r["symbol"] and r["rule"] == "INVALID_FILTER"
    ]
    penalty_rules_g10k = [
        r for r in rules if "G10K" in r["symbol"] and r["rule"] == "DOMAIN_PENALTY"
    ]

    assert len(boost_rules) >= 2, f"Expected G06N boost rules, got {len(boost_rules)}"
    assert len(penalty_rules_g06t) >= 1, (
        f"Expected G06T penalty rules, got {len(penalty_rules_g06t)}"
    )
    assert len(penalty_rules_g06f) >= 1, (
        f"Expected G06F2207 penalty rules, got {len(penalty_rules_g06f)}"
    )
    assert len(penalty_rules_g10k) >= 1, (
        f"Expected G10K penalty rules, got {len(penalty_rules_g10k)}"
    )

    # Check relative ordering: G06N should be highest, penalized should be lowest
    g06n_scores = [scores[s] for s in scores if s.startswith("G06N")]
    penalized_scores = [
        scores[s] for s in ["G06T15/30", "G06F2207/00", "G10K11/00"] if s in scores
    ]

    assert min(g06n_scores) > max(penalized_scores), (
        f"G06N scores ({min(g06n_scores)}) should exceed penalized ({max(penalized_scores)})"
    )

    print("[OK] PASS: AI domain constraints applied correctly")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()

    # Show rules log
    for rule in result["phase35_rules_log"]:
        print(
            f"  {rule['rule']}: {rule['symbol']} {rule['score_before']:.3f} -> {rule['score_after']:.3f}"
        )
    print()

    # Show rules log
    for rule in result["phase35_rules_log"]:
        print(
            f"  {rule['rule']}: {rule['symbol']} {rule['score_before']:.3f} -> {rule['score_after']:.3f}"
        )
    print()


def test_weight_clipping_disambiguation():
    """Test 'weight clipping' maps to G06N not G06T."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.95,
        },
        "domain_signals": [
            {
                "name": "neural network",
                "confidence": 0.95,
                "cpc_family": "G06N",
                "role": "primary",
            },
        ],
        "terms": [
            {"term": "weight clipping", "importance": 10},
            {"term": "neural network", "importance": 9},
        ],
        "core_function": "Optimize neural network weights",
    }

    candidates = [
        {"symbol": "G06N3/045", "title": "Parameter optimization", "score": 0.7},
        {
            "symbol": "G06T15/30",
            "title": "Image clipping",
            "score": 0.8,
        },  # Should be penalized
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 60)
    print("TEST: Weight Clipping Disambiguation")
    print("=" * 60)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06N should now exceed G06T
    assert scores["G06N3/045"] > scores["G06T15/30"], (
        f"G06N3/045 ({scores['G06N3/045']}) should exceed G06T15/30 ({scores['G06T15/30']})"
    )

    print("[OK] PASS: Weight clipping correctly mapped to G06N")
    print(f"  G06N3/045: {scores['G06N3/045']:.4f}")
    print(f"  G06T15/30: {scores['G06T15/30']:.4f}")
    print()


def test_functional_boosting():
    """Test functional boosting for quantization."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.9,
        },
        "terms": [
            {"term": "quantization", "importance": 10},
        ],
        "core_function": "Model compression through quantization and low-bit representation",
    }

    candidates = [
        {"symbol": "G06N3/063", "title": "Quantization methods", "score": 0.7},
        {"symbol": "G06N3/045", "title": "Parameter optimization", "score": 0.6},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 60)
    print("TEST: Functional Boosting (Quantization)")
    print("=" * 60)

    adjusted = result["phase35_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06N3/063 (quantization) should be boosted more than G06N3/045
    assert scores["G06N3/063"] > scores["G06N3/045"], (
        f"G06N3/063 ({scores['G06N3/063']}) should exceed G06N3/045 ({scores['G06N3/045']})"
    )

    print("[OK] PASS: Functional boosting applied correctly")
    print(f"  G06N3/063: {scores['G06N3/063']:.4f} (quantization)")
    print(f"  G06N3/045: {scores['G06N3/045']:.4f} (parameter optimization)")
    print()


def test_multi_domain_support():
    """Test multi-domain inventions preserve both domains."""
    phase1 = {
        "primary_domain": {
            "name": "AI / Machine Learning",
            "cpc_class": "G06N",
            "confidence": 0.8,
        },
        "domain_signals": [
            {
                "name": "neural network",
                "confidence": 0.8,
                "cpc_family": "G06N",
                "role": "primary",
            },
            {
                "name": "image processing",
                "confidence": 0.7,
                "cpc_family": "G06T",
                "role": "secondary",
            },
        ],
        "terms": [
            {"term": "neural network", "importance": 9},
            {"term": "image segmentation", "importance": 8},
        ],
        "core_function": "AI-based image segmentation",
    }

    candidates = [
        {"symbol": "G06N3/045", "title": "AI model", "score": 0.8},
        {"symbol": "G06T7/00", "title": "Image segmentation", "score": 0.7},
    ]

    constraint = CPCDecisionTreeConstraint()
    result = constraint.apply_constraints(candidates, phase1)

    print("=" * 60)
    print("TEST: Multi-Domain Support")
    print("=" * 60)

    adjusted = result["phase35_candidates"]

    # Both should be preserved
    assert any(c["symbol"].startswith("G06N") for c in adjusted), (
        "G06N should be preserved"
    )
    assert any(c["symbol"].startswith("G06T") for c in adjusted), (
        "G06T should be preserved"
    )

    print("[OK] PASS: Multi-domain candidates preserved")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING PHASE 3.5 DECISION TREE TESTS")
    print("=" * 60 + "\n")

    test_ai_domain_constraints()
    test_weight_clipping_disambiguation()
    test_functional_boosting()
    test_multi_domain_support()

    print("=" * 60)
    print("ALL PHASE 3.5 TESTS PASSED [OK]")
    print("=" * 60)
