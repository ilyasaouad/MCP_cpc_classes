"""
Simulate Phase 2 output for the vehicle/LLM patent claims.
Uses a realistic Phase 1 mock (no LLM call) to show Phase 2A-2C behavior.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import (
    CPCFamilyRouter,
    extract_3char_families,
    FALLBACK_FAMILIES,
)
from cpc_classification.cpc_xml_parser import CPCXMLParser

# Realistic Phase 1 output based on patent analysis
phase1_mock = {
    "technical_object": "A method for evaluating and correcting vehicle embedded system functions using a large language model (LLM) that aggregates test results from multiple vehicles via wireless connection",
    "problem_solved": "How to efficiently evaluate and auto-correct embedded system functions in vehicles by leveraging distributed test data and LLM-based analysis",
    "solution_summary": "Collect test results through a touch-screen MMI form, transmit to remote device, aggregate with other vehicle data via LLM, evaluate function performance, and receive correction instructions",
    "system_context": "Vehicle embedded systems with human-machine interface (MMI) and wireless connectivity to remote server",
    "target_industry": "Automotive electronics and embedded software testing",
    "target_professionals": "Automotive software engineers, embedded system developers, vehicle diagnostic technicians",
    "core_function": "Distributed evaluation and autonomous correction of vehicle embedded functions using LLM aggregation of multi-vehicle test data",
    "claim_analysis": [
        {
            "claim_number": 1,
            "type": "METHOD",
            "core_function": "Evaluate vehicle function via LLM aggregation",
            "features": [
                "touch-screen MMI",
                "wireless transmission",
                "LLM aggregation",
                "correction instructions",
            ],
        },
        {
            "claim_number": 2,
            "type": "METHOD",
            "core_function": "Touch press input generation",
            "features": ["graphic object interaction"],
        },
        {
            "claim_number": 3,
            "type": "METHOD",
            "core_function": "Parameter correction via reinforcement learning",
            "features": ["RL-based parameter refinement"],
        },
        {
            "claim_number": 4,
            "type": "METHOD",
            "core_function": "Multi-modal test data collection",
            "features": ["text", "image", "score data"],
        },
        {
            "claim_number": 5,
            "type": "METHOD",
            "core_function": "RAG-based data aggregation",
            "features": ["enhanced recovery generation"],
        },
        {
            "claim_number": 6,
            "type": "METHOD",
            "core_function": "Vehicle identification",
            "features": ["vehicle ID step"],
        },
    ],
    "independent_claim_numbers": [1],
    "single_invention": True,
    "invention_groups": [
        {"claims": [1], "focus": "Vehicle function evaluation via LLM"}
    ],
    "classification_strategy": "system-first",
    "strategy_reasoning": "Invention is tightly coupled to vehicle embedded systems and automotive domain. The LLM is used specifically for vehicle function evaluation, not as a general AI component.",
    "consistency_check": "consistent",
    "class_hypotheses": [
        {"class": "G06N", "confidence": 0.85, "reasoning": "LLM and AI/ML evaluation"},
        {
            "class": "G06F",
            "confidence": 0.75,
            "reasoning": "Data processing and embedded systems",
        },
        {
            "class": "B60R",
            "confidence": 0.70,
            "reasoning": "Vehicle onboard systems and diagnostics",
        },
        {
            "class": "G06Q",
            "confidence": 0.55,
            "reasoning": "Data collection and transmission systems",
        },
        {
            "class": "H04L",
            "confidence": 0.50,
            "reasoning": "Wireless data transmission",
        },
    ],
    "cpc_classes": ["G06N", "G06F", "B60R"],
    "cpc_sections": ["G", "B"],
    "cpc_reasoning": "Primary: AI/ML (G06N) and computing (G06F) for LLM-based evaluation. Secondary: Vehicle systems (B60R) for automotive context.",
    "terms": [
        {
            "term": "large language model",
            "importance": 10,
            "justification": "Core AI component",
            "source_section": "claims",
        },
        {
            "term": "LLM",
            "importance": 10,
            "justification": "Core technology",
            "source_section": "claims",
        },
        {
            "term": "vehicle embedded system",
            "importance": 9,
            "justification": "Application domain",
            "source_section": "claims",
        },
        {
            "term": "human-machine interface",
            "importance": 8,
            "justification": "MMI/HMI component",
            "source_section": "claims",
        },
        {
            "term": "MMI",
            "importance": 8,
            "justification": "Interface abbreviation",
            "source_section": "claims",
        },
        {
            "term": "touch screen",
            "importance": 7,
            "justification": "Input device",
            "source_section": "claims",
        },
        {
            "term": "wireless connection",
            "importance": 7,
            "justification": "Data transmission",
            "source_section": "claims",
        },
        {
            "term": "test data aggregation",
            "importance": 9,
            "justification": "Core function",
            "source_section": "claims",
        },
        {
            "term": "reinforcement learning",
            "importance": 8,
            "justification": "Correction method",
            "source_section": "claims",
        },
        {
            "term": "parameter refinement",
            "importance": 7,
            "justification": "Optimization",
            "source_section": "claims",
        },
        {
            "term": "remote device",
            "importance": 6,
            "justification": "Cloud/server component",
            "source_section": "claims",
        },
        {
            "term": "function evaluation",
            "importance": 9,
            "justification": "Primary purpose",
            "source_section": "claims",
        },
        {
            "term": "correction instructions",
            "importance": 8,
            "justification": "Output",
            "source_section": "claims",
        },
        {
            "term": "RAG",
            "importance": 6,
            "justification": "Retrieval augmented generation",
            "source_section": "claims",
        },
        {
            "term": "multi-vehicle data",
            "importance": 7,
            "justification": "Distributed aspect",
            "source_section": "claims",
        },
    ],
    "negative_signals": [
        {"term": "pure software", "confidence": 0.6},
        {"term": "general AI", "confidence": 0.5},
    ],
    "negative_domains": [
        {"domain": "general machine learning", "confidence": 0.6},
        {"domain": "non-automotive embedded", "confidence": 0.5},
    ],
    "negative_reasoning": "Patent is specifically about vehicle systems, not general AI or non-automotive applications",
    "claim_classifications": [
        {
            "claim_number": 1,
            "claim_type": "independent",
            "parent_claim": None,
            "cpc_classes": ["G06N3/08", "B60R16/03"],
            "reasoning": "LLM evaluation + vehicle systems",
            "provisional": True,
        },
        {
            "claim_number": 2,
            "claim_type": "dependent",
            "parent_claim": 1,
            "cpc_classes": ["G06F3/048"],
            "reasoning": "Touch interface",
            "provisional": True,
        },
        {
            "claim_number": 3,
            "claim_type": "dependent",
            "parent_claim": 1,
            "cpc_classes": ["G06N3/098"],
            "reasoning": "RL parameter optimization",
            "provisional": True,
        },
    ],
}

print("=" * 70)
print("PHASE 2 PIPELINE OUTPUT (Simulation)")
print("=" * 70)

# Phase 2A: Family Router (no KG -> hypotheses fallback)
print("\n" + "-" * 70)
print("PHASE 2A: CPC Family Router")
print("-" * 70)

router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
phase2a = router.route(phase1_mock)

print(f"Selected Families: {phase2a['families']}")
print(f"Source: {phase2a['source']}")
print(f"Reasoning: {phase2a['reasoning']}")
print(f"Scores: {phase2a['scores']}")

# Phase 2B: Restricted Expansion
print("\n" + "-" * 70)
print("PHASE 2B: Restricted XML Expansion")
print("-" * 70)

xml_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "cpc_classification",
    "resources",
    "cpc_scheme_2026",
)

if os.path.exists(xml_dir):
    parser = CPCXMLParser(xml_dir)

    # Get combined classes from hypotheses
    hypotheses = phase1_mock.get("class_hypotheses", [])
    cpc_classes = [h["class"] for h in hypotheses if h.get("confidence", 0) > 0.3]

    print(f"Input classes from Phase 1: {cpc_classes}")
    print(f"Phase 2A families (filter): {phase2a['families']}")

    # Expand with family filtering
    all_subgroups = parser.expand_classes(
        cpc_classes,
        include_non_allocatable=False,
        allowed_roots=phase2a["families"],
    )

    print(f"Expanded candidates (with family filter): {len(all_subgroups)}")

    # Compare with unfiltered
    unfiltered = parser.expand_classes(cpc_classes, include_non_allocatable=False)
    print(f"Expanded candidates (without filter): {len(unfiltered)}")

    reduction = (1 - len(all_subgroups) / len(unfiltered)) * 100 if unfiltered else 0
    print(f"Search space reduction: {reduction:.1f}%")

    # Show sample codes
    print(f"\nSample expanded codes:")
    for sg in all_subgroups[:10]:
        print(f"  {sg['symbol']} | {sg['title'][:50]}...")

    # Phase 2C: Scoring (simplified)
    print("\n" + "-" * 70)
    print("PHASE 2C: TF-IDF Scoring (Simplified Preview)")
    print("-" * 70)
    print(f"Input to scoring: {len(all_subgroups)} candidates")
    print(f"Expected output: Top 7 scored candidates")
    print(f"Note: Full TF-IDF scoring requires running the complete pipeline")

else:
    print("XML directory not found, skipping expansion demo")
    print(f"Expected families: {phase2a['families']}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Phase 2A Families:       {phase2a['families']}")
print(f"Phase 2A Source:         {phase2a['source']}")
print(
    f"Phase 2B Candidate Count: {len(all_subgroups) if os.path.exists(xml_dir) else 'N/A'}"
)
print(f"Expected Phase 2C Output: Top 7 ranked CPC codes with scores")
print("\nExpected top CPC areas:")
print("  - G06N: AI/Neural networks (LLM evaluation)")
print("  - G06F: Computing (data processing)")
print("  - B60R: Vehicle systems (embedded diagnostics)")
print("  - G06Q: Data systems (transmission/collection)")
