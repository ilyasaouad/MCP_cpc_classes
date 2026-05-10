"""
Test modality-aware routing with two contrasting patents.

Case 1: Rule-based LLM integration (AI IS the purpose)
Case 2: LLM-based image segmentation for defect detection (AI is tool, vision IS purpose)

Expected:
- Case 1: G06N primary (AI/rule-based is the invention)
- Case 2: G06V primary (vision is the invention), G06N secondary (AI is tool)
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import CPCFamilyRouter

print("=" * 70)
print("MODALITY-AWARE ROUTING TEST")
print("=" * 70)

# ───────────────────────────────────────────
# CASE 1: Rule-based LLM (AI IS the purpose)
# ───────────────────────────────────────────
phase1_case1 = {
    "technical_object": "A method integrating rule-based decision trees with LLMs for automated rule checking",
    "core_function": "Structured rule-based evaluation using LLM-augmented logical inference",
    "system_context": "Rule-based AI systems with generative LLMs for automated decision support",
    "classification_strategy": "hybrid",
    "domain_signals": [
        {"name": "rule-based artificial intelligence", "confidence": 0.95},
        {"name": "large language model", "confidence": 0.95},
        {"name": "decision support system", "confidence": 0.85},
        {"name": "natural language processing", "confidence": 0.80},
        {"name": "expert system", "confidence": 0.75},
    ],
    "terms": [
        {"term": "rule tree", "importance": 10},
        {"term": "rulemap", "importance": 10},
        {"term": "large language model", "importance": 10},
        {"term": "logical links", "importance": 9},
        {"term": "system prompt", "importance": 9},
        {"term": "user prompt", "importance": 9},
        {"term": "query definition", "importance": 8},
        {"term": "rule-based checking", "importance": 9},
        {"term": "automated decision", "importance": 8},
        {"term": "AND OR XOR NOT", "importance": 7},
    ],
}

# ───────────────────────────────────────────
# CASE 2: Vision + LLM (Vision IS purpose, AI is tool)
# ───────────────────────────────────────────
phase1_case2 = {
    "technical_object": "A method for automated defect detection in manufacturing using image segmentation with a large language model that analyzes visual patterns to identify defects in production lines",
    "core_function": "Computer vision-based defect detection through image segmentation and visual pattern analysis",
    "system_context": "Manufacturing quality control systems using AI-powered visual inspection",
    "classification_strategy": "system-first",
    "domain_signals": [
        {"name": "computer vision", "confidence": 0.95},
        {"name": "image segmentation", "confidence": 0.95},
        {"name": "defect detection", "confidence": 0.90},
        {"name": "manufacturing inspection", "confidence": 0.85},
        {"name": "large language model", "confidence": 0.70},  # Tool, not purpose
        {"name": "quality control", "confidence": 0.80},
        {"name": "optical inspection", "confidence": 0.75},
    ],
    "terms": [
        {"term": "image segmentation", "importance": 10},
        {"term": "defect detection", "importance": 10},
        {"term": "computer vision", "importance": 9},
        {"term": "visual pattern", "importance": 9},
        {"term": "manufacturing", "importance": 8},
        {"term": "inspection", "importance": 8},
        {"term": "optical", "importance": 7},
        {"term": "quality control", "importance": 7},
        {"term": "large language model", "importance": 5},  # Lower - it's a tool
        {"term": "segmentation mask", "importance": 9},
        {"term": "pixel analysis", "importance": 8},
        {"term": "boundary detection", "importance": 7},
        {"term": "contour", "importance": 6},
        {"term": "texture", "importance": 6},
        {"term": "neural network", "importance": 4},  # Tool
    ],
}

router = CPCFamilyRouter(knowledge_graph=None, max_families=3)

for i, (name, phase1) in enumerate(
    [
        ("CASE 1: Rule-based LLM (AI is PURPOSE)", phase1_case1),
        ("CASE 2: Vision + LLM (Vision is PURPOSE, AI is TOOL)", phase1_case2),
    ],
    1,
):
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")

    result = router.route(phase1)

    print(f"\nPhase 2A Results:")
    print(f"  Families:    {result['families']}")
    print(f"  PRIMARY:     {result['primary']}  <-- CPC classifies by PURPOSE")
    print(f"  Secondary:   {result['secondary']}")
    print(f"  Modality:    {result['modality']}")
    print(f"  Source:      {result['source']}")
    print(f"  Scores:      {result['scores']}")
    print(f"\n  Reasoning:   {result['reasoning']}")

    # Validation
    if i == 1:
        # Case 1: AI should be primary
        expected_primary = "G06N"
        is_correct = result["primary"] == expected_primary
        status = "CORRECT" if is_correct else "WRONG"
        print(f"\n  Expected primary: {expected_primary}")
        print(f"  Actual primary:   {result['primary']}")
        print(f"  Status:           {status}")

    elif i == 2:
        # Case 2: Vision should be primary, NOT AI
        expected_primary = "G06V"
        is_correct = result["primary"] == expected_primary
        status = "CORRECT" if is_correct else "WRONG"
        ai_penalized = result["scores"].get("G06N", 0) < result["scores"].get("G06V", 0)
        print(f"\n  Expected primary: {expected_primary} (vision, not AI)")
        print(f"  Actual primary:   {result['primary']}")
        print(f"  AI penalized:     {ai_penalized}")
        print(f"  Status:           {status}")

        if not is_correct:
            print(f"\n  *** FIX NEEDED: Vision terms should dominate over AI terms ***")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print("""
Architecture principle:
  CPC classifies by TECHNICAL PURPOSE, not tools used.
  
  Case 1 (Rule-based LLM):
    Purpose = rule-based reasoning using LLM
    -> G06N primary (AI is the invention itself)
    
  Case 2 (Vision + LLM):
    Purpose = visual defect detection
    Tool = LLM for analysis
    -> G06V primary (vision is the invention)
    -> G06N secondary (AI is just a tool)
    
Key fix applied:
  - PURPOSE domains: weight=1.2 (computer vision, manufacturing)
  - TOOL domains: weight=0.35 (LLM, neural network)
  - Co-occurrence rule: vision + AI -> boost vision, penalize AI
  - Modality correction: vision_count >= ai_count -> boost G06V/G06T
""")
print(f"{'=' * 70}")
