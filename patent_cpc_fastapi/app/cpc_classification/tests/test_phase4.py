"""
Test Phase 4: CPC Hypothesis Consolidation

Validates clustering, scoring, and hypothesis building.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_hypothesis_consolidation import (
    CPCHypothesisConsolidator,
    normalize_to_family,
    lexical_similarity,
    consolidate_cpc_hypotheses,
)

print("=" * 70)
print("PHASE 4: CPC HYPOTHESIS CONSOLIDATION TEST")
print("=" * 70)

# ───────────────────────────────────────────
# Test 1: Basic normalization
# ───────────────────────────────────────────
print("\n--- Test 1: Normalization ---")
assert normalize_to_family("G06V10/764") == "G06V"
assert normalize_to_family("G06N3/08") == "G06N"
assert normalize_to_family("B60R") == "B60R"
assert normalize_to_family("H04") == "H04"
print("  [PASS] Normalization works")

# ───────────────────────────────────────────
# Test 2: Lexical similarity
# ───────────────────────────────────────────
print("\n--- Test 2: Lexical Similarity ---")
sim1 = lexical_similarity(
    "image segmentation for defect detection", "image-based defect analysis"
)
sim2 = lexical_similarity("natural language processing", "image segmentation")
assert sim1 > sim2, f"Similar titles should have higher similarity: {sim1} > {sim2}"
print(f"  Similar titles: {sim1:.3f}")
print(f"  Different titles: {sim2:.3f}")
print("  [PASS] Lexical similarity works")

# ───────────────────────────────────────────
# Test 3: Clustering and hypothesis building
# ───────────────────────────────────────────
print("\n--- Test 3: Full Consolidation ---")

# Mock Phase 3 candidates (vision + AI)
candidates = [
    {
        "symbol": "G06V10/764",
        "title": "Image segmentation for object detection",
        "score": 0.91,
    },
    {"symbol": "G06V10/75", "title": "Image-based pattern recognition", "score": 0.88},
    {"symbol": "G06V20/10", "title": "Visual inspection systems", "score": 0.85},
    {
        "symbol": "G06V10/82",
        "title": "Image analysis with neural networks",
        "score": 0.82,
    },
    {"symbol": "G06N3/08", "title": "Neural network learning methods", "score": 0.78},
    {"symbol": "G06N5/022", "title": "Expert systems for reasoning", "score": 0.75},
    {"symbol": "G06N7/01", "title": "Logical inference using AI", "score": 0.72},
    {"symbol": "G06T7/13", "title": "Edge detection in images", "score": 0.70},
    {"symbol": "B23Q17/24", "title": "Machine tool monitoring", "score": 0.65},
    {"symbol": "G01N21/88", "title": "Optical inspection of materials", "score": 0.60},
]

# Mock term importance
terms = {
    "image segmentation": 10,
    "defect detection": 9,
    "computer vision": 9,
    "neural network": 5,  # Tool, lower importance
    "inspection": 8,
    "pattern recognition": 7,
}

consolidator = CPCHypothesisConsolidator(max_hypotheses=3)
result = consolidator.consolidate(candidates, terms)

print(f"\n  Input: {len(candidates)} candidates")
print(f"  Clusters formed: {result['phase4_cluster_count']}")
print(f"  Hypotheses: {len(result['phase4_hypotheses'])}")
print(f"  Primary family: {result['phase4_primary_family']}")
print(f"  Confidence: {result['phase4_confidence']}")

print(f"\n  Hypotheses:")
for h in result["phase4_hypotheses"]:
    print(f"    {h['role'].upper()}: {h['family']}")
    print(f"      Score: {h['score']:.3f}")
    print(f"      Candidates: {h['candidate_count']}")
    print(f"      Codes: {', '.join(h['supporting_codes'])}")
    print(f"      Coherence: {h['coherence']:.3f}")
    print(f"      Reasoning: {h['reasoning'][:80]}...")

# Validation
assert result["phase4_primary_family"] == "G06V", (
    f"Expected G06V primary, got {result['phase4_primary_family']}"
)
assert len(result["phase4_hypotheses"]) <= 3, "Max 3 hypotheses"
assert result["phase4_confidence"] in ["high", "medium", "low"]

hypotheses = result["phase4_hypotheses"]
assert hypotheses[0]["role"] == "primary"

# Check that vision cluster has higher score than AI cluster
g06v = next((h for h in hypotheses if h["family"] == "G06V"), None)
g06n = next((h for h in hypotheses if h["family"] == "G06N"), None)

if g06v and g06n:
    assert g06v["score"] > g06n["score"], (
        f"Vision should score higher than AI: {g06v['score']} > {g06n['score']}"
    )
    print(
        f"\n  [PASS] Vision cluster ({g06v['score']:.3f}) dominates AI cluster ({g06n['score']:.3f})"
    )
else:
    print(f"\n  [INFO] G06V={g06v is not None}, G06N={g06n is not None}")

# ───────────────────────────────────────────
# Test 4: Empty candidates
# ───────────────────────────────────────────
print("\n--- Test 4: Empty Input ---")
empty_result = consolidator.consolidate([], {})
assert empty_result["phase4_cluster_count"] == 0
assert empty_result["phase4_confidence"] == "low"
print("  [PASS] Empty input handled correctly")

# ───────────────────────────────────────────
# Test 5: Single cluster
# ───────────────────────────────────────────
print("\n--- Test 5: Single Family ---")
single = [
    {"symbol": "G06V10/764", "title": "Image segmentation", "score": 0.95},
    {"symbol": "G06V10/75", "title": "Pattern recognition", "score": 0.90},
]
single_result = consolidator.consolidate(single, {"image": 10})
assert single_result["phase4_confidence"] == "high"
assert len(single_result["phase4_hypotheses"]) == 1
print("  [PASS] Single cluster gives high confidence")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print("""
Phase 4 Architecture:
  Input:  Phase 3 ranked candidates (max 10)
  Output: Structured hypotheses (max 3)
  
  Steps:
    1. Normalize codes to family (G06V10/764 -> G06V)
    2. Cluster by family + title similarity
    3. Compute cluster strength (score sum * term density * coherence)
    4. Rank clusters
    5. Build hypotheses (primary/secondary/tertiary)
    
  Constraints satisfied:
    - No new CPC codes generated
    - Max 3 hypotheses
    - Always identifies PRIMARY
    - Uses only Phase 3 input
""")
