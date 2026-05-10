"""
Full pipeline test (FAST - no model downloads)
Phase 1 (mock) → 2A → 3 → 4 → 5
"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import CPCFamilyRouter
from cpc_classification.cpc_hypothesis_consolidation import CPCHypothesisConsolidator
from cpc_classification.cpc_hypothesis_resolver import CPCHypothesisResolver

# Phase 1 mock
phase1 = {
    "technical_object": "A method for quantizing trained large language models",
    "core_function": "Model compression through weight clipping and asymmetric quantization",
    "system_context": "Neural network deployment systems",
    "classification_strategy": "function-first",
    "domain_signals": [
        {"name": "neural network quantization", "confidence": 0.95},
        {"name": "large language model", "confidence": 0.95},
        {"name": "model compression", "confidence": 0.90},
    ],
    "terms": [
        {"term": "quantization", "importance": 10},
        {"term": "large language model", "importance": 10},
        {"term": "weight clipping", "importance": 10},
        {"term": "transformer model", "importance": 9},
        {"term": "asymmetric quantization", "importance": 9},
    ],
}

print("=" * 60)
print("FULL PIPELINE: LLM Quantization Patent")
print("=" * 60)

# Phase 2A
print("\n[PHASE 2A] Family Router")
router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
phase2a = router.route(phase1)
print(f"  Primary:   {phase2a['primary']}")
print(f"  Secondary: {phase2a['secondary']}")
print(f"  Families:  {phase2a['families']}")

# Phase 3 candidates
candidates = [
    {"symbol": "G06N3/063", "title": "Language models", "score": 0.95},
    {
        "symbol": "G06N3/08",
        "title": "Learning methods for neural networks",
        "score": 0.92,
    },
    {
        "symbol": "G06N3/096",
        "title": "Neural network architectures for compression",
        "score": 0.90,
    },
    {
        "symbol": "G06N3/047",
        "title": "Quantisation of neural network parameters",
        "score": 0.89,
    },
    {
        "symbol": "G06N3/098",
        "title": "Neural network training with regularisation",
        "score": 0.85,
    },
    {
        "symbol": "G06N20/00",
        "title": "Machine learning using ensemble methods",
        "score": 0.82,
    },
    {
        "symbol": "G06F17/16",
        "title": "Digital computing for matrix operations",
        "score": 0.78,
    },
    {
        "symbol": "G06F7/483",
        "title": "Arithmetic operations with reduced word length",
        "score": 0.75,
    },
    {
        "symbol": "G06F9/30036",
        "title": "Instruction architectures for parallel processing",
        "score": 0.72,
    },
    {
        "symbol": "G06N3/006",
        "title": "Artificial life based on physical entities",
        "score": 0.70,
    },
]

print("\n[PHASE 3] Ranked Candidates (Top 10)")
for i, c in enumerate(candidates, 1):
    print(f"  {i}. {c['symbol']} | {c['title'][:40]:40s} | {c['score']:.2f}")

# Phase 4
print("\n[PHASE 4] Hypothesis Consolidation")
term_importance = {t["term"].lower(): float(t["importance"]) for t in phase1["terms"]}
consolidator = CPCHypothesisConsolidator(max_hypotheses=2)
phase4 = consolidator.consolidate(candidates, term_importance)

print(f"  Clusters:    {phase4['phase4_cluster_count']}")
print(f"  Strong:      {phase4['phase4_strong_clusters']}")
print(f"  Hypotheses:  {len(phase4['phase4_hypotheses'])}")
print(f"  Primary:     {phase4['phase4_primary_family']}")
print(f"  Support:     {phase4['phase4_support_weight']:.2f}")

for h in phase4["phase4_hypotheses"]:
    print(f"\n  [{h['role'].upper()}] {h['family']}")
    print(f"    Score:      {h['normalized_score']:.3f}")
    print(f"    Candidates: {h['candidate_count']}")
    print(f"    Codes:      {', '.join(h['supporting_codes'][:3])}")

# Phase 5
print("\n[PHASE 5] Hypothesis Resolution")
resolver = CPCHypothesisResolver()
phase5 = resolver.resolve(phase4, phase1)

primary = phase5["primary"]
print(f"\n  PRIMARY: {primary['family']}")
print(f"    Final Score:  {primary['final_score']:.4f}")
print(f"    Func Align:   {primary['functional_alignment']:.4f}")
print(f"    Tech Coverage:{primary['technical_coverage']:.4f}")
print(f"    Confidence:   {primary['confidence']}")

if "secondary" in phase5:
    sec = phase5["secondary"]
    print(f"\n  SECONDARY: {sec['family']}")
    print(f"    Final Score: {sec['final_score']:.4f}")
    print(f"    Confidence:  {sec['confidence']}")

logic = phase5["decision_logic"]
print(f"\n  Gap: {logic['score_gap']:.4f}")
print(f"  Method: {logic['selection_method']}")

# Final JSON
print("\n" + "=" * 60)
print("COMPLETE RESULT")
print("=" * 60)

result = {
    "phase2a": {
        "families": phase2a["families"],
        "primary": phase2a["primary"],
        "secondary": phase2a["secondary"],
    },
    "phase3": {
        "top_10": [{"symbol": c["symbol"], "score": c["score"]} for c in candidates]
    },
    "phase4": {
        "primary_family": phase4["phase4_primary_family"],
        "hypotheses_count": len(phase4["phase4_hypotheses"]),
        "support_weight": phase4["phase4_support_weight"],
    },
    "phase5": {
        "primary": {
            "family": primary["family"],
            "final_score": primary["final_score"],
            "confidence": primary["confidence"],
        },
        "secondary": {
            "family": phase5.get("secondary", {}).get("family", ""),
            "final_score": phase5.get("secondary", {}).get("final_score", 0),
        }
        if "secondary" in phase5
        else None,
        "decision_logic": {
            "score_gap": logic["score_gap"],
            "secondary_accepted": logic["secondary_accepted"],
            "method": logic["selection_method"],
        },
    },
}

print(json.dumps(result, indent=2))

print("\n" + "=" * 60)
print("ALL PHASES COMPLETE")
print("=" * 60)
