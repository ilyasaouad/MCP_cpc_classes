"""
Test Phase 8 CPC Role Labeling (3-Layer Explanation Model).

Tests that candidates are correctly assigned roles:
- CORE, SUPPORT, CONTEXT, LEGAL_COVERAGE
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_role_labeling import CPCRoleLabeling


def test_role_labeling_basic():
    """Test basic role assignment."""
    phase1 = {
        "technical_object": "Neural network optimization system",
        "core_function": "Optimize weights through clipping",
        "system_context": "AI deployment pipeline",
    }

    phase36 = {
        "phase36_primary_code": "A",
        "phase36_types": {
            "A": ["weight", "clipping"],
            "B": ["compression"],
        },
    }

    candidates = [
        {
            "symbol": "G06N3/045",
            "title": "Parameter optimization",
            "score": 0.9,
            "contribution_match": "primary",
        },
        {
            "symbol": "G06N3/063",
            "title": "Model compression",
            "score": 0.8,
            "contribution_match": "secondary",
        },
        {
            "symbol": "G06N3/08",
            "title": "General NN",
            "score": 0.6,
            "contribution_match": "neutral",
        },
        {
            "symbol": "G06N3/",
            "title": "Neural networks",
            "score": 0.5,
            "contribution_match": "neutral",
        },
    ]

    labeler = CPCRoleLabeling()
    result = labeler.label_roles(candidates, phase1, phase36)

    print("=" * 70)
    print("TEST: Role Labeling Basic")
    print("=" * 70)

    core = result["layer1_core"]
    support = result["layer2_support"]
    context = result["layer2_context"]
    coverage = result["layer3_coverage"]

    # CORE should have G06N3/045 (primary match)
    assert any(c["symbol"] == "G06N3/045" for c in core), "G06N3/045 should be CORE"
    assert all(c["role"] == "CORE" for c in core), (
        "All CORE items should have role=CORE"
    )

    # SUPPORT should have G06N3/063 (secondary match)
    assert any(c["symbol"] == "G06N3/063" for c in support), (
        "G06N3/063 should be SUPPORT"
    )

    # COVERAGE should have G06N3/ (broader class)
    assert any(c["symbol"] == "G06N3/" for c in coverage), "G06N3/ should be COVERAGE"

    print("[OK] PASS: Roles assigned correctly")
    print(f"  CORE: {[c['symbol'] for c in core]}")
    print(f"  SUPPORT: {[c['symbol'] for c in support]}")
    print(f"  CONTEXT: {[c['symbol'] for c in context]}")
    print(f"  COVERAGE: {[c['symbol'] for c in coverage]}")
    print()


def test_role_labeling_no_secondary():
    """Test when no secondary matches exist."""
    phase1 = {
        "technical_object": "Simple invention",
        "core_function": "Do one thing",
        "system_context": "General computing",
    }

    phase36 = {
        "phase36_primary_code": "A",
        "phase36_types": {"A": ["optimization"]},
    }

    candidates = [
        {
            "symbol": "G06F9/",
            "title": "Computing",
            "score": 0.9,
            "contribution_match": "primary",
        },
    ]

    labeler = CPCRoleLabeling()
    result = labeler.label_roles(candidates, phase1, phase36)

    print("=" * 70)
    print("TEST: Role Labeling No Secondary")
    print("=" * 70)

    core = result["layer1_core"]
    assert len(core) == 1, "Should have 1 CORE"
    assert core[0]["symbol"] == "G06F9/"

    print("[OK] PASS: Single candidate correctly labeled as CORE")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING PHASE 8 ROLE LABELING TESTS")
    print("=" * 70 + "\n")

    test_role_labeling_basic()
    test_role_labeling_no_secondary()

    print("=" * 70)
    print("ALL PHASE 8 TESTS PASSED [OK]")
    print("=" * 70)
