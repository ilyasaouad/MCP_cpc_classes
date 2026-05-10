"""
Phase 3 + Phase 4 output for LLM Quantization patent (standalone simulation).
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import CPCFamilyRouter
from cpc_classification.cpc_hypothesis_consolidation import CPCHypothesisConsolidator

# Phase 1: Semantic extraction (NO CPC classes)
phase1 = {
    "technical_object": "A method for quantizing trained large language models by calculating optimal clipping ranges per layer, clipping outlier weights, and mapping continuous weights to discrete values for efficient downstream deployment",
    "core_function": "Model compression through layer-wise weight clipping and asymmetric quantization of transformer-based LLMs",
    "system_context": "Neural network deployment systems requiring efficient memory usage and inference speed through weight quantization",
    "classification_strategy": "function-first",
    "domain_signals": [
        {
            "name": "neural network quantization",
            "confidence": 0.95,
            "evidence": "quantizing LLM, discrete values, weight mapping",
        },
        {
            "name": "large language model",
            "confidence": 0.95,
            "evidence": "LLM, transformer model, layers, weights",
        },
        {
            "name": "model compression",
            "confidence": 0.90,
            "evidence": "clipping, quantization, downstream processing",
        },
        {
            "name": "machine learning",
            "confidence": 0.85,
            "evidence": "trained model, weights, inference",
        },
        {
            "name": "digital computing",
            "confidence": 0.70,
            "evidence": "continuous to discrete mapping, processors",
        },
    ],
    "terms": [
        {"term": "quantization", "importance": 10},
        {"term": "large language model", "importance": 10},
        {"term": "weight clipping", "importance": 10},
        {"term": "transformer model", "importance": 9},
        {"term": "asymmetric quantization", "importance": 9},
        {"term": "optimal clipping range", "importance": 9},
        {"term": "discrete values", "importance": 8},
        {"term": "neural network layers", "importance": 8},
        {"term": "model weights", "importance": 8},
        {"term": "zero point", "importance": 7},
        {"term": "greedy search", "importance": 6},
        {"term": "max-absolute-error", "importance": 6},
        {"term": "resolution errors", "importance": 6},
        {"term": "clipping errors", "importance": 6},
    ],
}

print("=" * 70)
print("PHASE 2A + PHASE 3 + PHASE 4 OUTPUT")
print("LLM Quantization Patent")
print("=" * 70)

# Phase 2A: Family Router
print("\n" + "-" * 70)
print("PHASE 2A: CPC Family Router")
print("-" * 70)

router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
phase2a = router.route(phase1)

print(f"Selected Families: {phase2a['families']}")
print(f"PRIMARY:     {phase2a['primary']}")
print(f"Secondary:   {phase2a['secondary']}")
print(f"Modality:    {phase2a['modality']}")
print(f"Source:      {phase2a['source']}")
print(f"Scores:      {phase2a['scores']}")
print(f"\nReasoning: {phase2a['reasoning']}")

# Phase 3: Mock ranked candidates (simulating TF-IDF scoring)
print("\n" + "-" * 70)
print("PHASE 3: Ranked Candidates (simulated TF-IDF scoring)")
print("-" * 70)

candidates = [
    # G06N - Neural networks / AI (primary)
    {
        "symbol": "G06N3/063",
        "title": "Language models, e.g. large language models [LLM]",
        "score": 0.95,
    },
    {
        "symbol": "G06N3/08",
        "title": "Learning methods for neural networks",
        "score": 0.92,
    },
    {
        "symbol": "G06N3/096",
        "title": "Neural network architectures for compression or pruning",
        "score": 0.90,
    },
    {
        "symbol": "G06N3/047",
        "title": "Quantisation of neural network parameters",
        "score": 0.89,
    },
    {
        "symbol": "G06N3/098",
        "title": "Neural network training and learning with regularisation",
        "score": 0.85,
    },
    {
        "symbol": "G06N20/00",
        "title": "Machine learning using ensemble methods",
        "score": 0.82,
    },
    # G06F - Computing / Digital processing (secondary)
    {
        "symbol": "G06F17/16",
        "title": "Digital computing for matrix operations and linear algebra",
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
    # G06N - AI training
    {
        "symbol": "G06N3/006",
        "title": "Artificial life based on physical entities controlled by simulated neural networks",
        "score": 0.70,
    },
]

print(f"Total candidates: {len(candidates)}")
print(f"\nTop 7 ranked:")
for i, c in enumerate(candidates[:7], 1):
    bar = "#" * int(c["score"] * 20)
    print(f"  {i}. {c['symbol']} | {c['title'][:45]:45s} | {c['score']:.2f} {bar}")

# Phase 4: Hypothesis Consolidation
print("\n" + "-" * 70)
print("PHASE 4: CPC Hypothesis Consolidation")
print("-" * 70)

# Build term importance dict (convert to float)
term_importance = {t["term"].lower(): float(t["importance"]) for t in phase1["terms"]}

# Use max 2 hypotheses (no tertiary)
consolidator = CPCHypothesisConsolidator(max_hypotheses=2)
phase4 = consolidator.consolidate(candidates, term_importance)

print(f"Clusters formed:     {phase4['phase4_cluster_count']}")
print(f"Strong clusters:     {phase4['phase4_strong_clusters']}")
print(f"Discarded clusters:  {phase4['phase4_discarded_clusters']}")
print(f"Hypotheses:          {len(phase4['phase4_hypotheses'])}")
print(f"Primary family:      {phase4['phase4_primary_family']}")
print(f"Centroid family:     {phase4['phase4_centroid_family']}")
print(f"Support weight:      {phase4['phase4_support_weight']:.2f}")
print(f"Confidence:          {phase4['phase4_confidence']}")
print(f"\nReasoning: {phase4['phase4_reasoning']}")

print(f"\nStructured Hypotheses:")
for i, h in enumerate(phase4["phase4_hypotheses"], 1):
    role_label = h["role"].upper()
    print(f"\n  {i}. {role_label}: {h['family']}")
    print(f"      Raw Score:       {h['score']:.3f}")
    print(f"      Normalized:      {h['normalized_score']:.3f}")
    print(f"      Mean Score:      {h['mean_score']:.3f}")
    print(f"      Candidates:      {h['candidate_count']}")
    print(f"      Coherence:       {h['coherence']:.3f}")
    print(f"      Codes:           {', '.join(h['supporting_codes'])}")
    print(f"      Reasoning:       {h['reasoning']}")

# Validation
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

expected_primary = "G06N"
actual_primary = phase4["phase4_primary_family"]

print(f"Expected primary: {expected_primary} (AI/Neural networks)")
print(f"Actual primary:   {actual_primary}")
print(
    f"Status:           {'CORRECT' if actual_primary == expected_primary else 'WRONG'}"
)
print(f"Support weight:   {phase4['phase4_support_weight']:.2f}")

# Check roles
roles = [h["role"] for h in phase4["phase4_hypotheses"]]
print(f"Roles present:    {roles}")
print(f"Has tertiary?     {'tertiary' in roles}")

# Check G06N dominates
if phase4["phase4_hypotheses"]:
    primary_hyp = phase4["phase4_hypotheses"][0]
    if len(phase4["phase4_hypotheses"]) > 1:
        second_hyp = phase4["phase4_hypotheses"][1]
        ratio = (
            primary_hyp["normalized_score"] / second_hyp["normalized_score"]
            if second_hyp["normalized_score"] > 0
            else 999
        )
        print(f"\nPrimary/Secondary ratio: {ratio:.2f}x (normalized)")
        print(f"Confidence level:        {phase4['phase4_confidence']}")
    else:
        print(f"\nSingle hypothesis - strong unimodal invention")
        print(f"Confidence level:        {phase4['phase4_confidence']}")

print(f"\nArchitecture check:")
print(f"  Phase 1 has CPC classes?     NO")
print(f"  Phase 2A primary family?     {phase2a['primary']}")
print(f"  Phase 3 candidate count?     {len(candidates)}")
print(f"  Phase 4 clusters formed?     {phase4['phase4_cluster_count']}")
print(f"  Phase 4 strong clusters?     {phase4['phase4_strong_clusters']}")
print(f"  Phase 4 discarded?           {phase4['phase4_discarded_clusters']}")
print(f"  Phase 4 hypothesis count?    {len(phase4['phase4_hypotheses'])} (max 2)")
print(f"  Phase 4 has tertiary?        {'tertiary' in roles}")
print(
    f"  Phase 4 normalized scores?   {all('normalized_score' in h for h in phase4['phase4_hypotheses'])}"
)
print("=" * 70)
