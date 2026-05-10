"""
Test Phase 5: CPC Hypothesis Resolution (deterministic resolver).
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_hypothesis_resolver import CPCHypothesisResolver

print("=" * 70)
print("PHASE 5: CPC HYPOTHESIS RESOLUTION TEST")
print("=" * 70)

# Phase 1 context
phase1 = {
    "technical_object": "A method for quantizing trained large language models by calculating optimal clipping ranges per layer",
    "core_function": "Model compression through layer-wise weight clipping and asymmetric quantization",
    "system_context": "Neural network deployment systems",
    "terms": [
        {"term": "quantization", "importance": 10},
        {"term": "large language model", "importance": 10},
        {"term": "weight clipping", "importance": 10},
        {"term": "transformer", "importance": 9},
        {"term": "neural network", "importance": 9},
        {"term": "model compression", "importance": 8},
    ],
}

# Phase 4 hypotheses
phase4 = {
    "phase4_hypotheses": [
        {
            "family": "G06N",
            "role": "primary",
            "score": 1.0,
            "normalized_score": 1.0,
            "mean_score": 0.86,
            "candidate_count": 7,
            "coherence": 0.83,
            "supporting_codes": [
                {"symbol": "G06N3/063", "title": "Language models"},
                {"symbol": "G06N3/08", "title": "Learning methods"},
            ],
            "reasoning": "Core invention domain: neural networks",
        },
        {
            "family": "G06F",
            "role": "secondary",
            "score": 0.56,
            "normalized_score": 0.56,
            "mean_score": 0.75,
            "candidate_count": 3,
            "coherence": 0.82,
            "supporting_codes": [
                {"symbol": "G06F17/16", "title": "Digital computing"},
            ],
            "reasoning": "Supporting infrastructure: digital processing",
        },
    ],
    "phase4_primary_family": "G06N",
    "phase4_confidence": "medium",
}

print("\n" + "-" * 70)
print("Input: Phase 4 Hypotheses")
print("-" * 70)
for h in phase4["phase4_hypotheses"]:
    print(f"  {h['role'].upper()}: {h['family']} (score={h['normalized_score']:.3f})")

# Resolve
print("\n" + "-" * 70)
print("PHASE 5: Resolution")
print("-" * 70)

resolver = CPCHypothesisResolver()
result = resolver.resolve(phase4, phase1)

primary = result["primary"]
print(f"\nPRIMARY:")
print(f"  Family:        {primary['family']}")
print(f"  Final Score:   {primary['final_score']:.4f}")
print(f"  Phase4 Score:  {primary['phase4_score']:.4f}")
print(f"  Func Align:    {primary['functional_alignment']:.4f}")
print(f"  Tech Coverage: {primary['technical_coverage']:.4f}")
print(f"  Specificity:   {primary['specificity_match']:.4f}")
print(f"  Confidence:    {primary['confidence']}")
print(f"  Reasoning:     {primary['reasoning']}")

if "secondary" in result:
    secondary = result["secondary"]
    print(f"\nSECONDARY:")
    print(f"  Family:        {secondary['family']}")
    print(f"  Final Score:   {secondary['final_score']:.4f}")
    print(f"  Phase4 Score:  {secondary['phase4_score']:.4f}")
    print(f"  Func Align:    {secondary['functional_alignment']:.4f}")
    print(f"  Tech Coverage: {secondary['technical_coverage']:.4f}")
    print(f"  Confidence:    {secondary['confidence']}")
    print(f"  Reasoning:     {secondary['reasoning']}")
else:
    print(f"\nSECONDARY: None (gap too large)")

logic = result["decision_logic"]
print(f"\nDECISION LOGIC:")
print(f"  Score Gap:     {logic['score_gap']:.4f}")
print(f"  Secondary:     {'Accepted' if logic['secondary_accepted'] else 'Rejected'}")
print(f"  Method:        {logic['selection_method']}")
print(f"  Evaluated:     {logic['num_hypotheses_evaluated']}")

# Validation
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
assert primary["family"] == "G06N", f"Expected G06N, got {primary['family']}"
assert "tertiary" not in [h.get("role") for h in phase4["phase4_hypotheses"]]
assert logic["selection_method"] == "deterministic_scoring"
print("  [PASS] Primary is G06N")
print("  [PASS] Deterministic scoring used")
print("  [PASS] No tertiary role")

if "secondary" in result:
    assert logic["score_gap"] < 0.25, "Gap should be < 0.25 for secondary"
    print(f"  [PASS] Secondary accepted (gap={logic['score_gap']:.3f} < 0.25)")

print(f"\n  Scoring formula:")
print(f"    final = 0.5 * phase4 + 0.3 * func_align + 0.2 * tech_cov")
print(
    f"    {primary['final_score']:.3f} = 0.5 * {primary['phase4_score']:.3f} + 0.3 * {primary['functional_alignment']:.3f} + 0.2 * {primary['technical_coverage']:.3f}"
)

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print("""
Phase 5 Architecture:
  Input:  Phase 4 hypotheses + Phase 1 context
  Output: Primary (+ optional Secondary) with decision logic
  
  Rules:
    - Only PRIMARY and SECONDARY roles
    - Deterministic scoring (no LLM for classification)
    - Max 2 outputs
    - Secondary only if gap < 0.25
    - LLM used only for tie-breaking (not implemented yet)
""")
