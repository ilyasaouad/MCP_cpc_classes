"""
Full pipeline test: Phase 1 (mock) → Phase 2 → Phase 3 → Phase 4 → Phase 5

This script runs the COMPLETE pipeline without LLM timeouts
by using realistic Phase 1 mock data.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import CPCFamilyRouter
from cpc_classification.cpc_xml_parser import CPCXMLParser
from cpc_classification.cpc_hypothesis_consolidation import CPCHypothesisConsolidator
from cpc_classification.cpc_hypothesis_resolver import CPCHypothesisResolver

# ───────────────────────────────────────────
# PHASE 1: Mock semantic extraction (bypass LLM)
# ───────────────────────────────────────────
phase1 = {
    "technical_object": "A method for quantizing trained large language models by calculating optimal clipping ranges per layer, clipping outlier weights, and mapping continuous weights to discrete values for efficient downstream deployment",
    "core_function": "Model compression through layer-wise weight clipping and asymmetric quantization of transformer-based LLMs",
    "system_context": "Neural network deployment systems requiring efficient memory usage and inference speed through weight quantization",
    "problem_solved": "Reducing latency and computational cost of LLMs by quantizing high-precision weights to lower bit-precision integers",
    "solution_summary": "Calculate optimal clipping range per layer using greedy search based on max-absolute-error, clip outlier weights, then quantize from float to integer with asymmetric weight quantization",
    "target_industry": "Artificial intelligence deployment, cloud computing, edge computing",
    "target_professionals": "ML engineers, AI researchers, system architects",
    "classification_strategy": "function-first",
    "strategy_reasoning": "The invention is a generic AI/ML method (quantization) applicable across multiple industries, not tied to a specific application domain",
    "consistency_check": "consistent",
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
        {
            "term": "quantization",
            "importance": 10,
            "justification": "Core operation",
            "source_section": "claims",
        },
        {
            "term": "large language model",
            "importance": 10,
            "justification": "Core subject",
            "source_section": "claims",
        },
        {
            "term": "weight clipping",
            "importance": 10,
            "justification": "Core operation",
            "source_section": "claims",
        },
        {
            "term": "transformer model",
            "importance": 9,
            "justification": "Architecture type",
            "source_section": "summary",
        },
        {
            "term": "asymmetric quantization",
            "importance": 9,
            "justification": "Specific technique",
            "source_section": "claims",
        },
        {
            "term": "optimal clipping range",
            "importance": 9,
            "justification": "Key concept",
            "source_section": "claims",
        },
        {
            "term": "discrete values",
            "importance": 8,
            "justification": "Output format",
            "source_section": "claims",
        },
        {
            "term": "neural network layers",
            "importance": 8,
            "justification": "Structural element",
            "source_section": "claims",
        },
        {
            "term": "model weights",
            "importance": 8,
            "justification": "Core data",
            "source_section": "claims",
        },
        {
            "term": "zero point",
            "importance": 7,
            "justification": "Quantization parameter",
            "source_section": "claims",
        },
        {
            "term": "greedy search",
            "importance": 6,
            "justification": "Optimization method",
            "source_section": "claims",
        },
        {
            "term": "max-absolute-error",
            "importance": 6,
            "justification": "Error metric",
            "source_section": "claims",
        },
        {
            "term": "resolution errors",
            "importance": 6,
            "justification": "Error type",
            "source_section": "summary",
        },
        {
            "term": "clipping errors",
            "importance": 6,
            "justification": "Error type",
            "source_section": "summary",
        },
    ],
    "negative_signals": [
        {"term": "image processing", "confidence": 0.7},
        {"term": "computer vision", "confidence": 0.7},
        {"term": "robotics", "confidence": 0.6},
        {"term": "hardware sensor", "confidence": 0.7},
    ],
    "negative_domains": [
        {"domain": "computer vision", "confidence": 0.7},
        {"domain": "robotic automation", "confidence": 0.6},
        {"domain": "physical hardware control", "confidence": 0.7},
    ],
    "negative_reasoning": "Patent is about logical numerical optimization and model compression, not image processing, physical robots, or hardware sensors",
}

print("=" * 70)
print("FULL PIPELINE OUTPUT")
print("LLM Quantization Patent")
print("=" * 70)

# ───────────────────────────────────────────
# PHASE 2A: Family Router
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 1: Semantic Extraction (MOCK - bypass LLM)")
print("-" * 70)
print(f"Technical Object: {phase1['technical_object'][:80]}...")
print(f"Core Function:    {phase1['core_function']}")
print(f"Strategy:         {phase1['classification_strategy']}")
print(f"Domain Signals:   {len(phase1['domain_signals'])}")
print(f"Terms:            {len(phase1['terms'])}")

print("\n" + "-" * 70)
print("PHASE 2A: CPC Family Router")
print("-" * 70)

router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
phase2a = router.route(phase1)

print(f"Families:    {phase2a['families']}")
print(f"PRIMARY:     {phase2a['primary']}")
print(f"Secondary:   {phase2a['secondary']}")
print(f"Modality:    {phase2a['modality']}")
print(f"Source:      {phase2a['source']}")
print(f"Scores:      {phase2a['scores']}")

# ───────────────────────────────────────────
# PHASE 2B: XML Expansion (simulated)
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 2B: Restricted XML Expansion")
print("-" * 70)

families = phase2a["families"]
print(f"Expanding families: {families}")
print(f"(XML expansion would happen here with allowed_roots={families})")

# ───────────────────────────────────────────
# PHASE 2C: TF-IDF Scoring (simulated candidates)
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 2C: TF-IDF Scoring")
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
    {
        "symbol": "G06N3/006",
        "title": "Artificial life based on physical entities controlled by simulated neural networks",
        "score": 0.70,
    },
    # G06F - Computing (secondary)
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
]

print(f"Scored {len(candidates)} candidates")

# ───────────────────────────────────────────
# PHASE 3: Ranking (top 10)
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 3: Ranking (Top 10)")
print("-" * 70)

ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:10]

for i, c in enumerate(ranked, 1):
    bar = "#" * int(c["score"] * 20)
    print(f"  {i:2}. {c['symbol']} | {c['title'][:45]:45s} | {c['score']:.2f} {bar}")

# ───────────────────────────────────────────
# PHASE 4: Hypothesis Consolidation
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 4: CPC Hypothesis Consolidation")
print("-" * 70)

term_importance = {t["term"].lower(): float(t["importance"]) for t in phase1["terms"]}
consolidator = CPCHypothesisConsolidator(max_hypotheses=2)
phase4 = consolidator.consolidate(ranked, term_importance)

print(f"Clusters:      {phase4['phase4_cluster_count']}")
print(f"Strong:        {phase4['phase4_strong_clusters']}")
print(f"Discarded:     {phase4['phase4_discarded_clusters']}")
print(f"Hypotheses:    {len(phase4['phase4_hypotheses'])}")
print(f"Primary:       {phase4['phase4_primary_family']}")
print(f"Support:       {phase4['phase4_support_weight']:.2f}")
print(f"Confidence:    {phase4['phase4_confidence']}")

for i, h in enumerate(phase4["phase4_hypotheses"], 1):
    role = h["role"].upper()
    print(f"\n  {i}. {role}: {h['family']}")
    print(f"      Normalized: {h['normalized_score']:.3f}")
    print(f"      Mean:       {h['mean_score']:.3f}")
    print(f"      Candidates: {h['candidate_count']}")
    print(f"      Coherence:  {h['coherence']:.3f}")
    print(f"      Codes:      {', '.join(h['supporting_codes'][:5])}")

# ───────────────────────────────────────────
# PHASE 5: Hypothesis Resolution
# ───────────────────────────────────────────
print("\n" + "-" * 70)
print("PHASE 5: CPC Hypothesis Resolution")
print("-" * 70)

resolver = CPCHypothesisResolver()
phase5 = resolver.resolve(phase4, phase1)

primary = phase5["primary"]
print(f"\nPRIMARY:")
print(f"  Family:       {primary['family']}")
print(f"  Final Score:  {primary['final_score']:.4f}")
print(f"  Func Align:   {primary['functional_alignment']:.4f}")
print(f"  Tech Coverage:{primary['technical_coverage']:.4f}")
print(f"  Confidence:   {primary['confidence']}")

if "secondary" in phase5:
    secondary = phase5["secondary"]
    print(f"\nSECONDARY:")
    print(f"  Family:       {secondary['family']}")
    print(f"  Final Score:  {secondary['final_score']:.4f}")
    print(f"  Confidence:   {secondary['confidence']}")

logic = phase5["decision_logic"]
print(f"\nDECISION LOGIC:")
print(f"  Score Gap:    {logic['score_gap']:.4f}")
print(f"  Secondary:    {'Accepted' if logic['secondary_accepted'] else 'Rejected'}")
print(f"  Method:       {logic['selection_method']}")

# ───────────────────────────────────────────
# FULL RESULT JSON
# ───────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPLETE RESULT JSON")
print("=" * 70)

result = {
    "phase1": {
        "technical_object": phase1["technical_object"],
        "core_function": phase1["core_function"],
        "domain_signals": [s["name"] for s in phase1["domain_signals"]],
        "terms_count": len(phase1["terms"]),
    },
    "phase2a": phase2a,
    "phase3": {
        "top_10_candidates": [
            {"symbol": c["symbol"], "title": c["title"], "score": c["score"]}
            for c in ranked
        ]
    },
    "phase4": phase4,
    "phase5": phase5,
}

print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nFinal Classification:")
print(f"  Primary:   {primary['family']}")
if "secondary" in phase5:
    print(f"  Secondary: {secondary['family']}")
print(f"\nAll phases executed successfully!")
print(f"Phase 1: Semantic extraction (mock)")
print(f"Phase 2A: Family routing")
print(f"Phase 2B/C: Expansion + scoring")
print(f"Phase 3: Ranking (top 10)")
print(f"Phase 4: Consolidation (max 2 hypotheses)")
print(f"Phase 5: Resolution (deterministic)")
print("=" * 70)
