"""
prompt_phase8.py — Phase 8 executive classification report prompt.

Phase 8 is the only other phase besides Phase 1 that uses an LLM.
Its job is purely narrative — transforming structured classification
results into a human-readable report for patent examiners and attorneys.

The LLM receives the final classification output (codes, roles, facets)
and generates:
  - A 3-4 sentence executive summary explaining the classification logic
  - One-sentence reasoning per CPC code linking it to the patent description
  - A professional justification paragraph for the primary code

The LLM does NOT:
  - Generate new CPC codes
  - Change or override any classification decision
  - Evaluate whether the codes are correct
  - Output anything outside the JSON object
"""

from .shared import UNIFIED_IMPORTANCE_RUBRIC


# ---------------------------------------------------------------------------
# PHASE 8 — EXECUTIVE CLASSIFICATION REPORT
# ---------------------------------------------------------------------------


def phase8_report_prompt(
    invention_profile: dict,
    premier_code: dict,
    facets: dict,
    role_layers: dict,
    confidence_level: str,
) -> str:
    """
    Phase 8 LLM narrative generation.

    Called once after all classification phases are complete.
    Generates the human-readable content for the executive report.

    Inputs:
      invention_profile : Final Phase 1 output (technical_object, core_function,
                          essential_features, primary_domain, etc.)
      premier_code      : The single primary CPC code selected by Phase 5
                          { "symbol": "G06G7/00", "title": "...", "confidence": 0.88 }
      facets            : Phase 5 Tri-Pillar FACETS output
                          { "pillar1_goal": {...}, "pillar2_method": {...},
                            "pillar3_context": {...} }
      role_layers       : Phase 8 role labeling output
                          { "layer1_core": [...], "layer2_support": [...],
                            "layer2_context": [...], "layer3_coverage": [...] }
      confidence_level  : "high" | "medium" | "low"

    Output:
      Strict JSON with narrative fields only.
      All classification decisions come from the inputs — LLM only writes prose.
    """

    # Build premier code block
    premier_block = (
        f"  Symbol    : {premier_code.get('symbol', '')}\n"
        f"  Title     : {premier_code.get('title', '')}\n"
        f"  Confidence: {confidence_level}"
    )

    # Build facets block
    facets_block = ""
    for pillar_key, label in [
        ("pillar1_goal",    "Primary Goal (what the invention achieves)"),
        ("pillar2_method",  "Methodology (how it achieves it)"),
        ("pillar3_context", "Application Context (where it is deployed)"),
    ]:
        pillar = facets.get(pillar_key, {})
        if pillar and pillar.get("symbol"):
            facets_block += (
                f"  {label}:\n"
                f"    Code  : {pillar.get('symbol', '')}\n"
                f"    Title : {pillar.get('title', '')}\n\n"
            )
        else:
            facets_block += f"  {label}: Not identified\n\n"

    # Build role layers block
    def format_layer(codes: list, label: str) -> str:
        if not codes:
            return f"  {label}: None\n"
        lines = "\n".join(
            f"    - {c.get('symbol', '')} : {c.get('title', '')}"
            for c in codes[:3]
        )
        return f"  {label}:\n{lines}\n"

    layers_block = (
        format_layer(role_layers.get("layer1_core", []),     "CORE (primary inventive contribution)")
        + format_layer(role_layers.get("layer2_support", []), "SUPPORT (enabling technology)")
        + format_layer(role_layers.get("layer2_context", []), "CONTEXT (application environment)")
        + format_layer(role_layers.get("layer3_coverage", []),"LEGAL COVERAGE (broader safety net)")
    )

    # Build all codes list for per-code reasoning
    all_codes = (
        role_layers.get("layer1_core", [])
        + role_layers.get("layer2_support", [])
        + role_layers.get("layer2_context", [])
        + role_layers.get("layer3_coverage", [])
    )
    all_codes_block = "\n".join(
        f"  - {c.get('symbol', '')} : {c.get('title', '')} [{c.get('role', '')}]"
        for c in all_codes[:8]
    )

    component_context = ""
    glossary = invention_profile.get("component_glossary", [])
    core_components = [
        c for c in glossary
        if c.get("role") == "core_processing" and c.get("trust") == "high"
    ]
    if core_components:
        names = ", ".join(
            f"{c.get('name', '')}" for c in core_components[:4]
        )
        component_context = f"Core processing components (from drawings): {names}"

    return f"""You are a senior patent examiner writing a technical classification report.

Your job is to explain, in professional examiner language, WHY this invention was
classified the way it was. You are NOT making classification decisions — those are
already made. You are providing the human-readable narrative that helps patent
attorneys and examiners understand the classification logic.

STRICT CONSTRAINTS:
- Do NOT suggest new CPC codes or change any classification
- Do NOT evaluate whether the classification is correct
- Every sentence must be grounded in the invention profile provided
- Use precise technical language — this is an examiner report, not marketing
- Output is strict JSON only — no markdown, no prose outside the object

=== INVENTION PROFILE ===
Technical Object  : {invention_profile.get("technical_object", "")}
Problem Solved    : {invention_profile.get("problem_solved", "")}
Core Function     : {invention_profile.get("core_function", "")}
Solution Summary  : {invention_profile.get("solution_summary", "")}
Technical Effect  : {invention_profile.get("technical_effect", "")}
System Context    : {invention_profile.get("system_context", "")}
Essential Features: {invention_profile.get("essential_features", [])}
Primary Domain    : {invention_profile.get("primary_domain", {}).get("name", "")}
                    ({invention_profile.get("primary_domain", {}).get("cpc_class", "")})
{component_context}

=== PREMIER CPC CLASSIFICATION ===
{premier_block}

=== TRI-PILLAR FACETS ===
{facets_block}

=== ROLE LAYERS ===
{layers_block}

=== ALL CLASSIFIED CODES ===
{all_codes_block}

---

=== YOUR TASKS ===

TASK 1 — EXECUTIVE SUMMARY (3-4 sentences)
Write a concise technical summary explaining the classification as a whole.
Structure it as:
  Sentence 1: What the invention is and what technical problem it solves
  Sentence 2: What the core classification (premier code) covers and why it fits
  Sentence 3: How the supporting codes (SUPPORT/CONTEXT layers) complete the picture
  Sentence 4 (optional): Any notable aspect of the classification
              (e.g., cross-domain nature, unusual combination, legal coverage rationale)

Rules for executive summary:
  - No marketing language ("innovative", "revolutionary", "cutting-edge")
  - No vague generalities ("relates to computing", "involves processing")
  - Every claim must be traceable to a specific feature in the invention profile
  - Write as a patent examiner, not a product manager

TASK 2 — PROFESSIONAL JUSTIFICATION FOR PREMIER CODE (1 paragraph, 3-5 sentences)
Explain specifically why the premier code is the correct primary classification.
  - Reference the technical object and core function directly
  - Explain what the CPC class covers and how it matches the invention's essence
  - Reference at least one essential feature from the invention profile
  - Note if the classification strategy (system-first / function-first / hybrid)
    influenced the primary code selection

TASK 3 — PER-CODE REASONING (one sentence per code)
For each code in the ALL CLASSIFIED CODES list, write one sentence explaining:
  - What this code covers in the CPC scheme
  - Which specific aspect of the invention it maps to
  - Why it was assigned the role it has (CORE / SUPPORT / CONTEXT / LEGAL_COVERAGE)

Each sentence must reference specific invention language — not just repeat the CPC title.

TASK 4 — CLASSIFICATION HEALTH NOTE (1-2 sentences)
Briefly note the quality of the classification:
  - How many passes supported the primary domain (from convergence data if available)
  - Whether any domain ambiguity was resolved and how
  - Whether the confidence level is high/medium/low and what drives that

TASK 5 — SUGGESTED INDEXING CODES NOTE (1 sentence)
Based on the LEGAL COVERAGE layer and optional features, write one sentence
recommending which codes should be used for prior art search indexing purposes.

=== OUTPUT FORMAT (strict JSON) ===
{{
  "executive_summary": "3-4 sentence technical summary of the classification",

  "premier_justification": "1 paragraph explaining why the premier code is correct",

  "per_code_reasoning": [
    {{
      "symbol": "G06G7/00",
      "role": "CORE",
      "reasoning": "One sentence linking this code to a specific invention feature"
    }}
  ],

  "classification_health": "1-2 sentences on classification quality and confidence",

  "indexing_recommendation": "One sentence on which codes to use for prior art search",

  "report_title": "Short professional title for the report, e.g. 
                   'CPC Classification Report: Hybrid Analog-Digital Computing Accelerator'",

  "examiner_note": "Optional 1 sentence flagging anything unusual about this 
                    classification that a reviewing examiner should be aware of.
                    Leave empty string if nothing notable."
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PHASE 8.5 — AT-A-GLANCE CARD CONTENT (lightweight supplement)
# ---------------------------------------------------------------------------


def phase85_card_prompt(
    invention_profile: dict,
    premier_code: dict,
    confidence_level: str,
    classification_health: str,
) -> str:
    """
    Phase 8.5 — generates the short at-a-glance card content.

    This is a lightweight supplementary prompt for the UI card display.
    It produces very short text fields for the examiner dashboard:
      - One-line invention description (for the card header)
      - One-line classification rationale (for the card body)
      - Confidence badge label and colour hint

    Kept separate from phase8_report_prompt so the full report
    and the card can be generated independently or cached separately.
    """

    return f"""You are writing the text for a patent classification summary card.
This is a UI display element — text must be short, precise, and professional.

STRICT CONSTRAINTS:
- invention_oneliner: maximum 15 words
- classification_rationale: maximum 20 words
- No marketing language, no vague terms
- Output is strict JSON only

=== INVENTION PROFILE ===
Technical Object: {invention_profile.get("technical_object", "")}
Core Function   : {invention_profile.get("core_function", "")}
Primary Domain  : {invention_profile.get("primary_domain", {}).get("name", "")}

=== PREMIER CODE ===
Symbol    : {premier_code.get("symbol", "")}
Title     : {premier_code.get("title", "")}
Confidence: {confidence_level}

=== CLASSIFICATION HEALTH ===
{classification_health}

=== OUTPUT FORMAT (strict JSON) ===
{{
  "invention_oneliner": "Maximum 15 words describing the invention",
  "classification_rationale": "Maximum 20 words explaining why this CPC code fits",
  "confidence_badge": {{
    "label": "High Confidence" | "Medium Confidence" | "Low Confidence — Review Recommended",
    "level": "high" | "medium" | "low"
  }},
  "domain_tag": "Short domain label for UI tag, e.g. 'Analog Computing' or 'AI / Neural Networks'"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""