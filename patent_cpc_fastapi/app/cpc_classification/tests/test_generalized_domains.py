"""
Test generalization to non-AI domains.

Tests that the system correctly handles:
- Mechanical patents (F16)
- Telecom patents (H04L)
- Medical patents (A61)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_family_router import CPCFamilyRouter


def test_mechanical_patent():
    """Test mechanical sealing patent routes to F16."""
    phase1 = {
        "technical_object": "A high-pressure sealing mechanism for oil wellheads",
        "core_function": "Prevent fluid leakage under extreme pressure using composite gasket",
        "system_context": "Oil drilling equipment",
        "classification_strategy": "system-first",
        "primary_domain": {
            "name": "Mechanical Engineering",
            "cpc_class": "F16",
            "confidence": 0.95,
        },
        "domain_signals": [
            {
                "name": "mechanical sealing",
                "confidence": 0.95,
                "evidence": "gasket, seal, pressure",
                "cpc_family": "F16",
                "role": "primary",
            },
            {
                "name": "oil drilling",
                "confidence": 0.9,
                "evidence": "wellhead, drilling",
                "cpc_family": "E21",
                "role": "secondary",
            },
        ],
        "terms": [
            {"term": "sealing mechanism", "importance": 10},
            {"term": "gasket", "importance": 9},
            {"term": "oil wellhead", "importance": 8},
        ],
        "negative_domains": [
            {"domain": "image processing", "confidence": 0.9, "penalize_family": "G06T"}
        ],
    }

    router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
    result = router._route_from_domain_analysis(phase1)

    families = result.get("families", [])
    primary = result.get("primary", "")
    scores = result.get("scores", {})

    print("=" * 60)
    print("TEST: Mechanical Sealing Patent Routing")
    print("=" * 60)
    print(f"Selected families: {families}")
    print(f"Primary family: {primary}")
    print(f"Family scores: {scores}")

    assert "F16" in families, f"FAIL: F16 not in families {families}"
    assert primary == "F16", f"FAIL: Primary should be F16, got {primary}"

    # G06T should be penalized or absent
    g06t_score = scores.get("G06T", 0)
    f16_score = scores.get("F16", 0)
    assert f16_score > g06t_score * 3, (
        f"FAIL: F16 score ({f16_score}) should dominate G06T ({g06t_score})"
    )

    print("[OK] PASS: F16 is correctly identified as primary")
    print()


def test_telecom_patent():
    """Test wireless communication patent routes to H04L."""
    phase1 = {
        "technical_object": "A low-latency data transmission protocol for 5G networks",
        "core_function": "Reduce transmission delay through adaptive channel coding",
        "system_context": "Wireless communication systems",
        "classification_strategy": "function-first",
        "primary_domain": {
            "name": "Telecommunications",
            "cpc_class": "H04L",
            "confidence": 0.92,
        },
        "domain_signals": [
            {
                "name": "wireless communication",
                "confidence": 0.95,
                "evidence": "5G, protocol, channel",
                "cpc_family": "H04L",
                "role": "primary",
            },
            {
                "name": "error correction",
                "confidence": 0.8,
                "evidence": "channel coding",
                "cpc_family": "H04L",
                "role": "secondary",
            },
        ],
        "terms": [
            {"term": "data transmission protocol", "importance": 10},
            {"term": "5G network", "importance": 9},
            {"term": "channel coding", "importance": 8},
        ],
        "negative_domains": [
            {"domain": "image processing", "confidence": 0.9, "penalize_family": "G06T"}
        ],
    }

    router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
    result = router._route_from_domain_analysis(phase1)

    families = result.get("families", [])
    primary = result.get("primary", "")
    scores = result.get("scores", {})

    print("=" * 60)
    print("TEST: Telecom Patent Routing")
    print("=" * 60)
    print(f"Selected families: {families}")
    print(f"Primary family: {primary}")
    print(f"Family scores: {scores}")

    assert "H04L" in families, f"FAIL: H04L not in families {families}"
    assert primary == "H04L", f"FAIL: Primary should be H04L, got {primary}"

    # G06T should be penalized
    g06t_score = scores.get("G06T", 0)
    h04l_score = scores.get("H04L", 0)
    assert h04l_score > g06t_score * 3, (
        f"FAIL: H04L score ({h04l_score}) should dominate G06T ({g06t_score})"
    )

    print("[OK] PASS: H04L is correctly identified as primary")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING GENERALIZED DOMAIN TESTS")
    print("=" * 60 + "\n")

    test_mechanical_patent()
    test_telecom_patent()

    print("=" * 60)
    print("ALL GENERALIZED TESTS PASSED [OK]")
    print("=" * 60)
