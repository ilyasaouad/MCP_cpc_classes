"""
prompt_phase1.py — Multi-pass Phase 1 extraction prompts.

Passes:
  A — Function-first extraction (what does it DO?)
  B — Structure-first extraction (what IS it?)
  C — Problem-Solution extraction (EPO framework)
  D — Drawing description extraction (component vocabulary)

Reconciliation merges all passes into the final Phase 1 output.
Completeness scoring validates the result before Phase 2A.
"""

from .shared import UNIFIED_IMPORTANCE_RUBRIC


# ---------------------------------------------------------------------------
# PASS A — Function-first extraction
# ---------------------------------------------------------------------------


def phase1_pass_a_prompt(labeled_claims: str, description: str) -> str:
    return f"""You are a patent technical analyst performing a focused FUNCTION extraction.

Your task is to answer one question about this invention:
WHAT DOES IT DO? Focus exclusively on actions, operations, processes, and verbs.

Do NOT assign CPC classes. Do NOT describe physical structure unless structure IS the function.

=== CLAIMS (PRE-LABELED) ===
{labeled_claims}

=== DESCRIPTION ===
{description}

{UNIFIED_IMPORTANCE_RUBRIC}

---

EXTRACTION FOCUS: FUNCTIONS AND ACTIONS

Extract the following:

1. PRIMARY ACTION
   The single most important verb phrase describing what the invention does.
   Example: "dynamically adjusts weight parameters during neural network training"
   NOT: "a system comprising a processor"

2. SECONDARY ACTIONS (up to 4)
   Other significant operations performed by the invention.
   Each must be a verb phrase, not a noun description.

3. PROCESS STEPS (if method claims exist)
   List the ordered steps from independent method claims.
   Label each step with its claim number.

4. FUNCTIONAL TERMS
   Technical terms that describe operations (not components).
   Each term: {{ "term": "...", "importance": 8, "source_section": "claims", "is_verb_phrase": true }}

5. CORE TECHNICAL PROBLEM BEING SOLVED
   What operation was previously impossible, slow, inaccurate, or inefficient?
   Frame as: "The invention enables [action] which was previously [limitation]"

OUTPUT FORMAT (strict JSON):
{{
  "primary_action": "string",
  "secondary_actions": ["string"],
  "process_steps": [
    {{ "claim_number": 1, "step_number": 1, "action": "string" }}
  ],
  "functional_terms": [
    {{ "term": "string", "importance": 8, "source_section": "claims", "is_verb_phrase": true }}
  ],
  "core_problem": "string",
  "pass": "A"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PASS B — Structure-first extraction
# ---------------------------------------------------------------------------


def phase1_pass_b_prompt(labeled_claims: str, description: str) -> str:
    return f"""You are a patent technical analyst performing a focused STRUCTURE extraction.

Your task is to answer one question about this invention:
WHAT IS IT? Focus exclusively on components, architecture, modules, and physical/logical structure.

Do NOT assign CPC classes. Do NOT describe what the invention does — only what it IS made of.

=== CLAIMS (PRE-LABELED) ===
{labeled_claims}

=== DESCRIPTION ===
{description}

{UNIFIED_IMPORTANCE_RUBRIC}

---

EXTRACTION FOCUS: COMPONENTS AND ARCHITECTURE

Extract the following:

1. TECHNICAL OBJECT
   What physical or logical entity is claimed?
   Example: "a multi-layer transformer encoder with cross-attention modules"
   NOT: "a method for processing" (that is function, not structure)

2. CORE COMPONENTS (up to 6)
   The essential structural elements without which the invention cannot work.
   Each: {{ "name": "...", "role": "what this component contributes", "claim_number": 1 }}

3. SYSTEM ARCHITECTURE
   How components are connected or arranged.
   Describe the data/signal flow between components.

4. STRUCTURAL TERMS
   Technical nouns representing components, modules, data structures.
   Each: {{ "term": "...", "importance": 8, "source_section": "claims", "is_noun": true }}

5. CLAIM TYPE ANALYSIS
   For each independent claim:
   - Is it METHOD, APPARATUS/SYSTEM, or BOTH?
   - List the structural elements it introduces.

OUTPUT FORMAT (strict JSON):
{{
  "technical_object": "string",
  "core_components": [
    {{ "name": "string", "role": "string", "claim_number": 1 }}
  ],
  "system_architecture": "string",
  "structural_terms": [
    {{ "term": "string", "importance": 8, "source_section": "claims", "is_noun": true }}
  ],
  "claim_type_analysis": [
    {{ "claim_number": 1, "type": "METHOD|APPARATUS|BOTH", "structural_elements": ["string"] }}
  ],
  "pass": "B"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PASS C — Problem-Solution extraction (EPO framing)
# ---------------------------------------------------------------------------


def phase1_pass_c_prompt(labeled_claims: str, description: str) -> str:
    return f"""You are a patent technical analyst performing an EPO-style PROBLEM-SOLUTION extraction.

Your task is to identify the inventive contribution using the EPO problem-solution framework:
1. What existed before (closest prior art)?
2. What was the unsolved technical problem?
3. What is the solution (the invention)?
4. What technical effect or improvement results?

Do NOT assign CPC classes. Focus entirely on understanding the inventive step.

=== CLAIMS (PRE-LABELED) ===
{labeled_claims}

=== DESCRIPTION ===
{description}

{UNIFIED_IMPORTANCE_RUBRIC}

---

EXTRACTION FOCUS: EPO PROBLEM-SOLUTION APPROACH

Extract the following in order:

1. CLOSEST PRIOR ART
   What technology or system existed before this invention?
   Describe the state of the art that the invention improves upon.

2. OBJECTIVE TECHNICAL PROBLEM
   What specific technical problem was unsolved?
   Must be concrete and technical — not commercial or social.
   Example: "How to reduce neural network inference latency without accuracy loss"
   NOT: "How to improve user experience"

3. SOLUTION (THE INVENTIVE CONTRIBUTION)
   The specific mechanism, method, or structure that solves the problem.
   This is the essence of what makes the invention novel.
   Focus on the ESSENTIAL features — what must be present for the solution to work.

4. TECHNICAL EFFECT
   What measurable or observable improvement does the solution produce?
   Example: "40% reduction in model size with less than 1% accuracy degradation"

5. ESSENTIAL VS OPTIONAL FEATURES
   Essential: Features present in EVERY independent claim — without these the invention fails.
   Optional: Features present only in dependent claims — preferred embodiments.

6. DOMAIN SIGNALS FROM PROBLEM-SOLUTION CONTEXT
   What technical domains does the problem-solution pair belong to?
   Example: The problem of "latency in neural network inference" belongs to G06N (AI/ML).
   List up to 4 domain signals with confidence.

OUTPUT FORMAT (strict JSON):
{{
  "closest_prior_art": "string",
  "objective_technical_problem": "string",
  "solution_summary": "string",
  "technical_effect": "string",
  "essential_features": ["string"],
  "optional_features": ["string"],
  "domain_signals_from_problem": [
    {{ "domain": "string", "cpc_family": "G06N", "confidence": 0.9,
       "evidence": "problem involves neural network parameter optimization" }}
  ],
  "pass": "C"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PASS D — Drawing description extraction
# ---------------------------------------------------------------------------


def phase1_pass_d_prompt(drawing_descriptions: str) -> str:
    return f"""You are a patent technical analyst performing a DRAWING DESCRIPTION extraction.

Drawing descriptions are uniquely valuable because:
- Every component has a reference numeral AND a name → structured vocabulary
- Component relationships are explicit (e.g., "module 102 receives output from encoder 104")
- Data and signal flows are described step by step
- The hardware/software boundary is made explicit
- Alternative embodiments reveal the true technical scope

Your task: Extract the complete technical vocabulary and architecture from the figure descriptions.

CRITICAL RULE — FIGURE TRUST LEVELS:
Many patents show prior art in early figures before showing the invention.
You must identify each figure's trust level:

  HIGH TRUST (weight × 0.95):
  - Figures labeled "according to the present invention"
  - Figures labeled "embodiment of the invention"
  - Figures showing the inventive system/method
  - Figures 2, 3, 4... (usually invention after Figure 1 shows prior art)

  LOW TRUST (weight × 0.2 — treat as background):
  - Figures labeled "prior art", "conventional", "background"
  - Figures described as "existing system" or "known system"
  - Figure 1 in patents where the description says "Figure 1 shows the prior art"

  MEDIUM TRUST (weight × 0.6):
  - Figures with no explicit label
  - Figures showing optional embodiments only

=== DRAWING DESCRIPTIONS / DETAILED DESCRIPTION OF FIGURES ===
{drawing_descriptions}

---

EXTRACTION TASKS:

1. FIGURE INVENTORY
   List every figure mentioned with its trust level and subject.
   {{ "figure": "FIG. 1", "trust": "low", "subject": "prior art system", "reason": "described as conventional system" }}

2. REFERENCE NUMERAL GLOSSARY (COMPONENT VOCABULARY)
   Extract every named component with its reference number.
   This is the most precise technical vocabulary in the patent.
   Only include components from HIGH or MEDIUM trust figures.
   {{ "ref": "102", "name": "bidirectional LSTM encoder", "figure": "FIG. 2",
      "role": "core_processing", "trust": "high" }}

   Component roles:
   - core_processing    : The central computation (most important for CPC)
   - input_interface    : How data enters the system
   - output_interface   : How results leave the system
   - data_storage       : Memory, buffers, databases
   - control_logic      : Orchestration, scheduling, routing
   - communication      : Transmission, networking, protocols
   - application_context: The environment/domain where the invention operates

3. DATA / SIGNAL FLOW
   Describe how data moves between components in the invention figures.
   Format: "Component A (ref 102) → [transformation] → Component B (ref 104)"
   Focus on the main processing pipeline, not optional paths.

4. HARDWARE / SOFTWARE BOUNDARY
   Explicitly identify:
   - Which components are HARDWARE (physical, silicon, mechanical)
   - Which components are SOFTWARE (algorithms, models, processes)
   - Which are FIRMWARE or MIXED

5. INVENTIVE COMPONENTS (HIGH TRUST ONLY)
   From high-trust figures, identify the 2-3 components that represent the
   core inventive contribution — components NOT present in prior art figures
   but present in the invention figures.

6. CROSS-FIGURE CONSISTENCY
   Which components appear in MULTIPLE invention figures?
   Components appearing in ≥ 3 invention figures are essential (high importance).
   Components in only 1 figure may be optional embodiments (lower importance).

7. TECHNICAL TERMS FROM DRAWINGS (section-weighted)
   Extract precise technical terms from drawing descriptions.
   Apply HIGH TRUST weight for invention figures, LOW TRUST for prior art figures.
   {{ "term": "cross-attention layer", "importance": 9, "source_section": "drawing_description",
      "figure": "FIG. 3", "trust_level": "high", "ref_numeral": "204" }}

OUTPUT FORMAT (strict JSON):
{{
  "figure_inventory": [
    {{ "figure": "string", "trust": "high|medium|low", "subject": "string", "reason": "string" }}
  ],
  "component_glossary": [
    {{ "ref": "string", "name": "string", "figure": "string",
       "role": "core_processing|input_interface|output_interface|data_storage|control_logic|communication|application_context",
       "trust": "high|medium|low" }}
  ],
  "data_flow": "string — narrative description of processing pipeline",
  "hardware_software_boundary": {{
    "hardware_components": ["ref + name"],
    "software_components": ["ref + name"],
    "mixed_components": ["ref + name"]
  }},
  "inventive_components": [
    {{ "ref": "string", "name": "string", "why_inventive": "not present in prior art figures" }}
  ],
  "cross_figure_components": [
    {{ "name": "string", "appears_in": ["FIG. 2", "FIG. 3"], "importance": "essential|optional" }}
  ],
  "drawing_terms": [
    {{ "term": "string", "importance": 9, "source_section": "drawing_description",
       "figure": "string", "trust_level": "high|medium|low", "ref_numeral": "string" }}
  ],
  "pass": "D"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# RECONCILIATION — Merge passes A, B, C, D into final Phase 1 output
# ---------------------------------------------------------------------------


def phase1_reconciliation_prompt(
    pass_a: dict,
    pass_b: dict,
    pass_c: dict,
    pass_d: dict | None,
    labeled_claims: str,
) -> str:
    import json

    pass_a_str = json.dumps(pass_a, indent=2)
    pass_b_str = json.dumps(pass_b, indent=2)
    pass_c_str = json.dumps(pass_c, indent=2)
    pass_d_str = (
        json.dumps(pass_d, indent=2)
        if pass_d
        else "NOT PROVIDED — no drawing descriptions available"
    )

    drawing_section = (
        f"""
=== PASS D — DRAWING DESCRIPTION EXTRACTION ===
{pass_d_str}
"""
        if pass_d
        else """
=== PASS D — DRAWING DESCRIPTIONS ===
NOT PROVIDED. No drawing descriptions were available for this patent.
Do not penalize the extraction — proceed with passes A, B, C only.
"""
    )

    return f"""You are reconciling four independent patent extraction passes into a single
authoritative Phase 1 output.

Each pass approached the invention from a different angle:
  Pass A — Function-first (what it DOES)
  Pass B — Structure-first (what it IS)
  Pass C — Problem-Solution (EPO framework: prior art gap + inventive contribution)
  Pass D — Drawing descriptions (component vocabulary, data flow, figure inventory)
           [May be absent if no drawings section was provided]

Your job:
1. Find CONVERGENCE — terms and domain signals appearing in ≥ 2 passes are HIGH confidence
2. Find DIVERGENCE — terms in only 1 pass need disambiguation or lower confidence
3. Merge into a single authoritative technical profile
4. Assign final confidence per domain signal based on how many passes support it

IMPORTANT PRINCIPLES:
- Independent claim elements from Pass B have authority over description terms
- Domain signals supported by BOTH Pass C (problem domain) and Pass D (drawing components)
  are the strongest possible anchors for Phase 2A
- If Pass D identified prior-art figures, do NOT let those component names dominate
- If passes conflict on primary domain, note the conflict and assign lower confidence

=== PASS A — FUNCTION EXTRACTION ===
{pass_a_str}

=== PASS B — STRUCTURE EXTRACTION ===
{pass_b_str}

=== PASS C — PROBLEM-SOLUTION EXTRACTION ===
{pass_c_str}
{drawing_section}

=== CLAIMS (for final verification) ===
{labeled_claims}

---

RECONCILIATION TASKS:

TASK 1 — TERM RECONCILIATION
For each unique technical term across all passes:
- Count how many passes mention it (convergence_count)
- Assign final importance based on convergence AND section weight
- Flag ambiguous terms for disambiguation
- Mark terms only in Pass D prior-art figures as low trust

TASK 2 — DOMAIN SIGNAL RECONCILIATION
Merge domain signals from all passes:
- Pass A contributes: functional domain signals (what the action belongs to)
- Pass B contributes: structural domain signals (what the components belong to)
- Pass C contributes: problem domain signals (what field the problem is in)
- Pass D contributes: component domain signals (what the named components belong to)
A domain signal confirmed by ≥ 3 passes → confidence ≥ 0.9
A domain signal from only 1 pass → confidence ≤ 0.5 (flag for review)

TASK 3 — TECHNICAL OBJECT SYNTHESIS
Combine Pass B's structural technical_object with Pass A's primary_action:
Final technical_object = "[structural description] that [primary action]"
Example: "A multi-layer transformer encoder that dynamically adjusts attention weights
          based on contextual token density"

TASK 4 — SELF-CONSISTENCY CHECK (MANDATORY)
After determining primary_domain and classification_strategy:
a) Re-read the technical_object you just synthesized
b) Re-read the core_function from Pass A
c) Ask: Does the primary_domain match BOTH the technical_object AND core_function?
d) If YES → consistency_check = "consistent"
e) If NO → revise the conflicting field AND output the REVISED value in the JSON below
   (do not just note the revision — actually output the corrected field value)

TASK 5 — MULTI-INVENTION CHECK
If Pass B's claim_type_analysis shows independent claims with DIFFERENT technical objects:
- Flag as multi-invention
- Assign separate domain signals per invention group
- Note which claims belong to which invention

---

OUTPUT FORMAT (strict JSON — this is the final Phase 1 output consumed by Phase 2A):

{{
  "technical_object": "string — synthesised from Pass A + Pass B",
  "problem_solved": "string — from Pass C objective_technical_problem",
  "solution_summary": "string — from Pass C solution_summary",
  "technical_effect": "string — from Pass C technical_effect",
  "system_context": "string — industry/domain where invention operates",
  "target_industry": "string",
  "target_professionals": "string",
  "core_function": "string — primary_action from Pass A",
  "essential_features": ["string — from Pass C essential_features"],
  "optional_features": ["string — from Pass C optional_features"],

  "component_glossary": [
    {{
      "ref": "string", "name": "string", "role": "string",
      "trust": "high|medium|low", "figure": "string"
    }}
  ],

  "data_flow": "string — from Pass D if available, else from Pass A process_steps",

  "hardware_software_boundary": {{
    "hardware_components": ["string"],
    "software_components": ["string"],
    "mixed_components": ["string"]
  }},

  "claim_analysis": [
    {{
      "claim_number": 1,
      "type": "METHOD|APPARATUS|BOTH",
      "core_function": "string",
      "features": ["string"]
    }}
  ],

  "independent_claim_numbers": [1],
  "single_invention": true,
  "invention_groups": [
    {{
      "claims": [1],
      "focus": "string",
      "primary_domain": "string",
      "cpc_class": "G06N"
    }}
  ],

  "classification_strategy": "system-first|function-first|hybrid",
  "strategy_reasoning": "string",
  "consistency_check": "consistent|revised",
  "consistency_revision_details": "string — if revised, explain what changed and why",
  "technical_object_revised": "string — ONLY if consistency_check is revised",
  "core_function_revised": "string — ONLY if consistency_check is revised",

  "primary_domain": {{
    "name": "string",
    "cpc_class": "G06N",
    "confidence": 0.9,
    "supporting_passes": ["A", "B", "C", "D"],
    "reasoning": "string"
  }},

  "class_hypotheses": [
    {{
      "class": "G06N",
      "confidence": 0.9,
      "source": "multi-pass convergence",
      "passes_supporting": ["A", "C", "D"]
    }}
  ],

  "domain_signals": [
    {{
      "name": "string",
      "confidence": 0.9,
      "evidence": "string",
      "cpc_family": "G06N",
      "role": "primary|secondary|negative",
      "convergence_count": 3,
      "passes_supporting": ["A", "B", "C"]
    }}
  ],

  "disambiguated_terms": [
    {{
      "term": "string",
      "meaning": "string",
      "domain": "G06N",
      "avoid": ["G06T"],
      "convergence_count": 2,
      "source_passes": ["A", "B"]
    }}
  ],

  "terms": [
    {{
      "term": "string",
      "importance": 8,
      "justification": "string",
      "source_section": "claims|summary|detailed_description|drawing_description|abstract|background",
      "convergence_count": 3,
      "figure": "FIG. 2 — only for drawing_description terms",
      "trust_level": "high|medium|low — only for drawing_description terms"
    }}
  ],

  "negative_signals": [
    {{ "term": "string", "confidence": 0.8, "penalize_family": "G06T" }}
  ],
  "negative_domains": [
    {{ "domain": "string", "confidence": 0.9, "penalize_family": "G06V" }}
  ],
  "negative_reasoning": "string",

  "phase1_completeness_score": 0,
  "completeness_notes": "string — what is well covered and what is uncertain"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# COMPLETENESS SCORING — Rule-based check before passing to Phase 2A
# ---------------------------------------------------------------------------


def score_phase1_completeness(extraction: dict, labeled_claims: str) -> dict:
    score = 0
    issues = []

    obj = extraction.get("technical_object", "")
    if len(obj) > 20:
        score += 25
    else:
        issues.append("technical_object is missing or too vague")

    signals = [
        s
        for s in extraction.get("domain_signals", [])
        if s.get("confidence", 0) >= 0.5 and s.get("role") != "negative"
    ]
    if len(signals) >= 3:
        score += 25
    elif len(signals) >= 1:
        score += 10
        issues.append(f"Only {len(signals)} domain signal(s) with confidence >= 0.5")
    else:
        issues.append("No confident domain signals found")

    pd_conf = extraction.get("primary_domain", {}).get("confidence", 0)
    if pd_conf >= 0.7:
        score += 20
    elif pd_conf >= 0.5:
        score += 10
        issues.append(f"Primary domain confidence only {pd_conf:.2f}")
    else:
        issues.append("Primary domain confidence too low or missing")

    high_terms = [t for t in extraction.get("terms", []) if t.get("importance", 0) >= 7]
    if len(high_terms) >= 5:
        score += 15
    elif len(high_terms) >= 2:
        score += 8
        issues.append(f"Only {len(high_terms)} high-importance terms")
    else:
        issues.append("Too few high-importance terms")

    if extraction.get("essential_features"):
        score += 15
    else:
        issues.append(
            "No essential features extracted (EPO problem-solution pass may have failed)"
        )

    if score >= 80:
        status = "PASS"
    elif score >= 60:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "score": score,
        "status": status,
        "issues": issues,
        "recommendation": (
            "Proceed to Phase 2A"
            if status == "PASS"
            else "Proceed with caution — note low confidence"
            if status == "WARN"
            else "Re-run Phase 1 or flag for human review"
        ),
    }


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT — orchestrates all passes
# ---------------------------------------------------------------------------


def phase1_prompt(
    labeled_claims: str,
    description: str,
    drawing_descriptions: str = "",
) -> dict:
    has_drawings = bool(drawing_descriptions and drawing_descriptions.strip())

    return {
        "pass_a": phase1_pass_a_prompt(labeled_claims, description),
        "pass_b": phase1_pass_b_prompt(labeled_claims, description),
        "pass_c": phase1_pass_c_prompt(labeled_claims, description),
        "pass_d": phase1_pass_d_prompt(drawing_descriptions) if has_drawings else None,
        "has_drawings": has_drawings,
    }


# ---------------------------------------------------------------------------
# DOMAIN INFERENCE — Phase 1b (probabilistic, not hardcoded)
# ---------------------------------------------------------------------------


def domain_inference_prompt(phase1_data: dict) -> str:
    terms = phase1_data.get("terms", [])
    term_list = "\n".join(
        f"- {t.get('term', '')} (importance: {t.get('importance', 5)}, "
        f"source: {t.get('source_section', 'unknown')}, "
        f"convergence: {t.get('convergence_count', 1)} passes)"
        for t in sorted(terms, key=lambda x: x.get("importance", 0), reverse=True)[:15]
    )

    hypotheses = phase1_data.get("class_hypotheses", [])
    hypo_list = "\n".join(
        f"- {h.get('class', '')} (confidence: {h.get('confidence', 0.5)}, "
        f"passes supporting: {h.get('passes_supporting', [])})"
        for h in hypotheses
    )

    component_glossary = phase1_data.get("component_glossary", [])
    component_list = (
        "\n".join(
            f"- [{c.get('ref', '?')}] {c.get('name', '')} — role: {c.get('role', 'unknown')} "
            f"(trust: {c.get('trust', 'unknown')})"
            for c in component_glossary[:10]
            if c.get("trust") in ("high", "medium")
        )
        or "No component glossary available (no drawing descriptions provided)"
    )

    return f"""You are estimating the probability that various CPC domains are relevant to this invention.

Use the reconciled Phase 1 output — including multi-pass convergence data, component glossary
from drawing descriptions, and problem-solution framing — to estimate domain probabilities.

=== INVENTION PROFILE ===
Technical Object: {phase1_data.get("technical_object", "")}
Problem Solved: {phase1_data.get("problem_solved", "")}
Solution Summary: {phase1_data.get("solution_summary", "")}
System Context: {phase1_data.get("system_context", "")}
Core Function: {phase1_data.get("core_function", "")}
Classification Strategy: {phase1_data.get("classification_strategy", "")}
Essential Features: {phase1_data.get("essential_features", [])}

=== EXTRACTED TERMS (top 15 by importance, multi-pass convergence shown) ===
{term_list}

=== MULTI-PASS CLASS HYPOTHESES ===
{hypo_list}

=== COMPONENT GLOSSARY (from drawing descriptions, high/medium trust only) ===
{component_list}

=== DATA FLOW ===
{phase1_data.get("data_flow", "Not available")}

=== HARDWARE/SOFTWARE BOUNDARY ===
Hardware: {phase1_data.get("hardware_software_boundary", {}).get("hardware_components", [])}
Software: {phase1_data.get("hardware_software_boundary", {}).get("software_components", [])}

=== TASK ===
Estimate domain probabilities. Weight your estimates by:
1. Multi-pass convergence — domains supported by ≥ 3 passes get higher probability
2. Component roles from drawing descriptions — core_processing components anchor the domain
3. Problem-solution framing — the problem domain is a strong anchor
4. Hardware/software boundary — hardware components suggest H/B sections, software suggests G section

Estimate probabilities for these CPC domains (add others as relevant):
- G06F  (Computing / data processing)
- G06N  (AI / neural networks / ML)
- G06K  (Pattern recognition / data representation)
- G06V  (Image/video recognition)
- G06T  (Image data processing)
- G10L  (Speech / audio / acoustic processing)
- H04L  (Digital information transmission)
- H04W  (Wireless communication)
- E21B  (Earth drilling / mining)
- F16J  (Sealing)
- F16K  (Valves)
- F01P  (Cooling)
- B60L  (Electrical propulsion)
- B01D  (Separation)
- G01N  (Investigating materials)
- G05B  (Control systems / automation)
- B60W  (Road vehicle drive control)
- A61B  (Medical technology)

Rules:
- Base probabilities on technical function and context, NOT keyword matching alone
- Component roles from drawing descriptions are strong evidence:
  core_processing components anchor the primary domain
  application_context components suggest secondary domain
- Do NOT force inclusion of any domain
- Domains with 0 supporting passes should have probability < 0.2
- Domains with ≥ 3 supporting passes should have probability > 0.7

Output format (strict JSON):
{{
  "domain_probabilities": [
    {{
      "class": "G06N",
      "probability": 0.92,
      "reasoning": "Core processing components (ref 102, 104) are neural network layers; "
                   "problem is about model optimization; supported by passes A, B, C, D"
    }}
  ],
  "primary_domain": "G06N",
  "primary_confidence": 0.92,
  "reasoning": "Brief explanation of top domain selection"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""
