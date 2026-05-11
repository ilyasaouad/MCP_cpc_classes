"""
Test Phase 3.6 Cross-Domain Validation Layer.

Tests that prevent domain collapse (e.g., prompt → G10L speech).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpc_cross_domain_validator import CrossDomainValidator


def test_prevent_g10l_without_audio():
    """Test that 'prompt' without audio signals does NOT go to G10L."""
    phase1 = {
        "technical_object": "LLM prompt optimization system",
        "core_function": "Optimize prompts for better text generation",
        "system_context": "NLP text processing pipeline",
        "terms": [
            {"term": "prompt", "importance": 10},
            {"term": "text generation", "importance": 9},
            {"term": "nlp", "importance": 8},
        ],
        "domain_signals": [
            {"name": "natural language processing", "confidence": 0.9},
        ],
    }

    # Candidates including a speech candidate that should be penalized
    candidates = [
        {"symbol": "G06F16/335", "title": "Text processing", "score": 0.8},
        {"symbol": "G06N20/00", "title": "AI learning", "score": 0.75},
        {
            "symbol": "G10L17/24",
            "title": "Speech prompt",
            "score": 0.9,
        },  # Should be penalized
        {"symbol": "H04L29/00", "title": "Network protocol", "score": 0.6},
    ]

    validator = CrossDomainValidator()
    result = validator.validate(candidates, phase1, {})

    print("=" * 70)
    print("TEST: Prevent G10L without audio signals")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    print(f"Domain verified: {result['phase36_domain_verified']}")
    print(f"Rules applied: {result['phase36_adjustments']}")
    print()

    # G10L should be heavily penalized (no audio context)
    assert scores["G10L17/24"] < scores["G06F16/335"], (
        f"FAIL: G10L ({scores['G10L17/24']}) should be penalized below G06F ({scores['G06F16/335']})"
    )

    # G06F and G06N should remain high
    assert scores["G06F16/335"] > 0.7, "G06F should remain high"
    assert scores["G06N20/00"] > 0.7, "G06N should remain high"

    # Check that entity mismatch rule was applied
    rules = result["phase36_rules_log"]
    entity_rules = [r for r in rules if "ENTITY" in r["rule"]]
    assert len(entity_rules) > 0, "Entity consistency rules should be applied"

    print("[OK] PASS: G10L correctly penalized without audio context")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()

    for rule in entity_rules:
        print(
            f"  {rule['rule']}: {rule['symbol']} {rule['score_before']:.3f} -> {rule['score_after']:.3f}"
        )
    print()


def test_allow_g10l_with_audio():
    """Test that G10L IS allowed when audio signals exist."""
    phase1 = {
        "technical_object": "Voice command system",
        "core_function": "Process audio commands",
        "system_context": "Speech recognition pipeline",
        "terms": [
            {"term": "speech", "importance": 10},
            {"term": "audio", "importance": 9},
            {"term": "voice", "importance": 8},
        ],
        "domain_signals": [
            {"name": "speech recognition", "confidence": 0.95},
        ],
    }

    candidates = [
        {"symbol": "G10L15/00", "title": "Speech recognition", "score": 0.8},
        {"symbol": "G06F3/00", "title": "Input processing", "score": 0.7},
    ]

    validator = CrossDomainValidator()
    result = validator.validate(candidates, phase1, {})

    print("=" * 70)
    print("TEST: Allow G10L with audio signals")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G10L should be boosted (audio context present)
    assert scores["G10L15/00"] > scores["G06F3/00"], (
        f"FAIL: G10L ({scores['G10L15/00']}) should exceed G06F ({scores['G06F3/00']}) with audio context"
    )

    print("[OK] PASS: G10L correctly boosted with audio context")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()


def test_prompt_maps_to_nlp_not_speech():
    """Test that 'prompt' maps to NLP (G06F/G06N), not speech (G10L)."""
    phase1 = {
        "technical_object": "Text prompt system",
        "core_function": "Process text prompts",
        "system_context": "NLP system",
        "terms": [
            {"term": "prompt", "importance": 10},
            {"term": "text", "importance": 9},
            {"term": "language model", "importance": 8},
        ],
    }

    candidates = [
        {"symbol": "G06F16/335", "title": "NLP processing", "score": 0.8},
        {"symbol": "G10L17/00", "title": "Speech prompt", "score": 0.85},
    ]

    validator = CrossDomainValidator()
    result = validator.validate(candidates, phase1, {})

    print("=" * 70)
    print("TEST: Prompt maps to NLP not speech")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    scores = {c["symbol"]: c["score"] for c in adjusted}

    # G06F should exceed G10L
    assert scores["G06F16/335"] > scores["G10L17/00"], (
        f"FAIL: G06F ({scores['G06F16/335']}) should exceed G10L ({scores['G10L17/00']}) for text prompts"
    )

    print("[OK] PASS: 'prompt' correctly mapped to NLP domain")
    for c in adjusted:
        print(f"  {c['symbol']}: {c['score']:.4f} - {c['title']}")
    print()


def test_two_signal_lock():
    """Test that ≥2 signals are required to lock a family."""
    phase1 = {
        "technical_object": "Neural network system",
        "core_function": "Optimize neural networks",
        "system_context": "AI pipeline",
        "terms": [
            {"term": "neural network", "importance": 10},
            {"term": "deep learning", "importance": 9},
        ],
    }

    candidates = [
        {"symbol": "G06N3/045", "title": "NN optimization", "score": 0.8},
        {"symbol": "G06F9/00", "title": "Computing", "score": 0.7},
    ]

    validator = CrossDomainValidator()
    result = validator.validate(candidates, phase1, {})

    print("=" * 70)
    print("TEST: Two-signal family lock")
    print("=" * 70)

    adjusted = result["phase36_candidates"]
    g06n = next(c for c in adjusted if c["symbol"] == "G06N3/045")

    # G06N should have ≥2 signals
    assert g06n.get("domain_signals_matched", 0) >= 2, (
        f"FAIL: G06N should have ≥2 signals, got {g06n.get('domain_signals_matched', 0)}"
    )

    print("[OK] PASS: Family locked with 2+ signals")
    print(f"  G06N signals matched: {g06n.get('domain_signals_matched', 0)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING CROSS-DOMAIN VALIDATION TESTS")
    print("=" * 70 + "\n")

    test_prevent_g10l_without_audio()
    test_allow_g10l_with_audio()
    test_prompt_maps_to_nlp_not_speech()
    test_two_signal_lock()

    print("=" * 70)
    print("ALL CROSS-DOMAIN TESTS PASSED [OK]")
    print("=" * 70)
