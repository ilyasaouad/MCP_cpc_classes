"""
Test fixes for cross-domain misclassification.

Tests the specific case:
- Invention: Neural network weight quantization + clipping for LLM deployment
- WRONG before: G06T15/30 (image clipping)
- CORRECT after: G06N 3/045, 3/063, 3/08
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_family_router import CPCFamilyRouter, CPCDomainTaxonomy


def test_hard_constraints_nn_subject():
    """Test that NN subject matter forces G06N as primary."""
    taxonomy = CPCDomainTaxonomy()

    # Simulate Phase 1 data for LLM quantization patent
    phase1 = {
        "technical_object": "A method for quantizing trained large language models",
        "core_function": "Model compression through weight clipping and quantization",
        "system_context": "Neural network deployment systems",
        "classification_strategy": "function-first",
        "domain_signals": [
            {
                "name": "neural network quantization",
                "confidence": 0.9,
                "evidence": "weight clipping, quantization",
            },
            {
                "name": "large language model",
                "confidence": 0.9,
                "evidence": "LLM deployment",
            },
            {
                "name": "model compression",
                "confidence": 0.85,
                "evidence": "reducing model size",
            },
        ],
        "terms": [
            {"term": "weight clipping", "importance": 10, "source_section": "claims"},
            {"term": "quantization", "importance": 10, "source_section": "claims"},
            {
                "term": "large language model",
                "importance": 9,
                "source_section": "summary",
            },
            {"term": "neural network", "importance": 9, "source_section": "summary"},
            {
                "term": "model compression",
                "importance": 8,
                "source_section": "detailed_description",
            },
            {"term": "inference latency", "importance": 7, "source_section": "claims"},
        ],
        "negative_signals": [
            {"term": "image processing", "confidence": 0.8},
            {"term": "computer graphics", "confidence": 0.7},
        ],
        "negative_domains": [
            {"domain": "image processing", "confidence": 0.9},
            {"domain": "computer vision", "confidence": 0.8},
        ],
        "negative_reasoning": "This patent is about neural network internals, not images or vision",
    }

    # Test domain analysis routing (no KG needed)
    router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
    result = router._route_from_domain_analysis(phase1)

    families = result.get("families", [])
    primary = result.get("primary", "")
    scores = result.get("scores", {})

    print("=" * 60)
    print("TEST: LLM Quantization Patent Routing")
    print("=" * 60)
    print(f"Selected families: {families}")
    print(f"Primary family: {primary}")
    print(f"Family scores: {scores}")
    print(f"Source: {result.get('source', '')}")
    print()

    # ASSERTIONS
    assert "G06N" in families, f"FAIL: G06N not in families {families}"
    assert primary == "G06N", f"FAIL: Primary should be G06N, got {primary}"

    # G06T should be heavily penalized or absent
    g06t_score = scores.get("G06T", 0)
    g06n_score = scores.get("G06N", 0)
    assert g06n_score > g06t_score * 3, (
        f"FAIL: G06N score ({g06n_score}) should dominate G06T ({g06t_score})"
    )

    print("[OK] PASS: G06N is correctly identified as primary")
    print(f"   G06N score: {g06n_score:.3f}")
    print(f"   G06T score: {g06t_score:.3f}")
    print()

    return result


def test_object_aware_keyword_mapping():
    """Test that 'weight clipping' maps to G06N, not G06T."""
    router = CPCFamilyRouter(knowledge_graph=None)

    # Case 1: Weight clipping (NN subject)
    phase1_nn = {
        "technical_object": "Weight clipping for neural networks",
        "core_function": "Clip weights to reduce model size",
        "system_context": "Neural network training",
        "terms": [
            {"term": "weight clipping", "importance": 10},
            {"term": "neural network", "importance": 9},
        ],
        "negative_domains": [{"domain": "image processing", "confidence": 0.8}],
    }

    scores_nn = router._apply_hard_constraints(phase1_nn, {})
    print("=" * 60)
    print("TEST: Object-Aware Keyword Mapping")
    print("=" * 60)
    print(
        f"'weight clipping' + NN context -> G06N boosted: {scores_nn.get('G06N', 0):.3f}"
    )

    # Case 2: Image clipping (image subject)
    phase1_img = {
        "technical_object": "Image clipping for graphics",
        "core_function": "Clip image regions",
        "system_context": "Computer graphics",
        "terms": [
            {"term": "image clipping", "importance": 10},
            {"term": "graphics", "importance": 9},
        ],
        "negative_domains": [],
    }

    scores_img = router._apply_hard_constraints(phase1_img, {})
    print(f"'image clipping' + graphics context -> G06T not penalized")
    print()

    print("[OK] PASS: Object-aware mapping works correctly")
    print()


def test_cross_domain_leakage_penalty():
    """Test that G06T is penalized when no image signals exist."""
    router = CPCFamilyRouter(knowledge_graph=None)

    phase1 = {
        "terms": [
            {"term": "quantization", "importance": 10},
            {"term": "neural network", "importance": 9},
            {"term": "weight", "importance": 8},
        ],
        "negative_domains": [{"domain": "image processing", "confidence": 0.9}],
    }

    scores = router._apply_hard_constraints(phase1, {"G06T": 2.0, "G06N": 1.0})

    print("=" * 60)
    print("TEST: Cross-Domain Leakage Penalty")
    print("=" * 60)
    print(f"G06T before: 2.000, after: {scores.get('G06T', 0):.3f}")
    print(f"G06N before: 1.000, after: {scores.get('G06N', 0):.3f}")

    assert scores["G06T"] < 1.0, f"FAIL: G06T should be penalized, got {scores['G06T']}"
    assert scores["G06N"] > 2.0, f"FAIL: G06N should be boosted, got {scores['G06N']}"

    print("[OK] PASS: Cross-domain leakage correctly penalized")
    print()


def test_modality_correction_nn_internal():
    """Test modality correction with NN internal terms."""
    router = CPCFamilyRouter(knowledge_graph=None)

    phase1 = {
        "terms": [
            {"term": "weight quantization", "importance": 10},
            {"term": "model compression", "importance": 9},
            {"term": "inference latency", "importance": 8},
        ],
    }

    scores = router._apply_modality_correction(phase1, {"G06N": 1.0, "G06T": 1.5})

    print("=" * 60)
    print("TEST: Modality Correction with NN Internal Terms")
    print("=" * 60)
    print(f"G06N score: {scores.get('G06N', 0):.3f}")
    print(f"G06T score: {scores.get('G06T', 0):.3f}")

    assert scores["G06N"] > scores["G06T"], (
        f"FAIL: G06N ({scores['G06N']}) should exceed G06T ({scores['G06T']})"
    )

    print("[OK] PASS: NN internal terms correctly boost G06N")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING CROSS-DOMAIN MISCLASSIFICATION FIX TESTS")
    print("=" * 60 + "\n")

    test_hard_constraints_nn_subject()
    test_object_aware_keyword_mapping()
    test_cross_domain_leakage_penalty()
    test_modality_correction_nn_internal()

    print("=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
