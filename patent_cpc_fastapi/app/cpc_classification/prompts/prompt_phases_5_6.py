"""
prompt_phases_5_6.py — Phase 5 + Phase 6 LLM prompts.

These two phases form a validation chain:
  Phase 5 (validation_prompt_single) — validates each CPC candidate individually
    against the invention profile.  Per-candidate quality gate — catches
    wrong-domain codes that deterministic scoring misses.

  Phase 6 (reconciliation_prompt) — updates per-claim CPC assignments after
    Phase 5 rejects some candidates.  Removes rejected codes and replaces
    them with validated ones so no claim is left with a rejected code.

The chain:  Phase 5 → Phase 6 → Phase 7
Phase 5 validates individually → produces rejected list
Phase 6 updates per-claim assignments using that rejected list → clean output
Phase 7 checks the cleaned set is coherent as a whole
"""


# =============================================================================
# PHASE 5 — SINGLE-CANDIDATE VALIDATION
# =============================================================================


def validation_prompt_single(
    phase1_data: dict,
    candidate: dict,
    score_margin: float,
    confidence_level: str,
) -> str:
    """
    Phase 5: Per-candidate semantic validation.

    Called for each CPC candidate before final selection.  The LLM evaluates
    whether the candidate's CPC title and description match the invention's
    technical profile — going beyond what TF-IDF scores can detect.

    This is the quality gate that catches wrong-domain codes (e.g., G06T
    image processing for a non-image invention) that passed through
    deterministic filtering with misleading scores.
    """
    strategy = phase1_data.get("classification_strategy", "unknown")
    claim_analysis = phase1_data.get("claim_analysis", [])

    method_claims = [c for c in claim_analysis if c.get("type") == "METHOD"]
    apparatus_claims = [c for c in claim_analysis if c.get("type") == "APPARATUS"]
    mixed_claims = [c for c in claim_analysis if c.get("type") == "BOTH"]

    if method_claims and not apparatus_claims:
        method_apparatus_note = (
            "NOTE: This patent contains ONLY method claims. "
            "The CPC class MUST cover methods/processes, not just apparatus."
        )
    elif apparatus_claims and not method_claims:
        method_apparatus_note = (
            "NOTE: This patent contains ONLY apparatus claims. "
            "The CPC class MUST cover systems/devices, not just methods."
        )
    elif mixed_claims or (method_claims and apparatus_claims):
        method_apparatus_note = (
            "NOTE: This patent contains both method and apparatus claims. "
            "The CPC class should cover both, or at least not contradict either."
        )
    else:
        method_apparatus_note = ""

    component_glossary = phase1_data.get("component_glossary", [])
    core_components = [
        c
        for c in component_glossary
        if c.get("role") == "core_processing" and c.get("trust") == "high"
    ]
    component_note = ""
    if core_components:
        names = ", ".join(f"[{c['ref']}] {c['name']}" for c in core_components[:4])
        component_note = f"""
=== CORE COMPONENTS FROM DRAWING DESCRIPTIONS ===
{names}
The candidate CPC class should describe systems or methods involving these component types.
"""

    return f"""You are validating a single CPC classification candidate against an invention's
full technical profile, including drawing-level component analysis.

=== INVENTION PROFILE ===
Technical Object:  {phase1_data.get("technical_object", "")}
Problem Solved:    {phase1_data.get("problem_solved", "")}
Core Function:     {phase1_data.get("core_function", "")}
System Context:    {phase1_data.get("system_context", "")}
Strategy:          {strategy}
Essential Features: {phase1_data.get("essential_features", [])}
Data Flow:         {phase1_data.get("data_flow", "Not available")}
{component_note}
=== SCORING CONTEXT ===
Confidence Level:   {confidence_level}
Score Margin (top1 - top2): {score_margin:.4f}
{method_apparatus_note}

=== CANDIDATE TO VALIDATE ===
Code:        {candidate.get("symbol", "")}
Title:       {candidate.get("title", "")}
Description: {candidate.get("full_context", candidate.get("title", ""))}

=== VALIDATION CRITERIA ===

1. FUNCTION ALIGNMENT (0.0-1.0)
   Does this CPC class describe the CORE FUNCTION of the invention?

2. CONTEXT ALIGNMENT (0.0-1.0)
   Does this CPC class fit the SYSTEM CONTEXT and INDUSTRY?

3. COMPONENT ALIGNMENT (0.0-1.0)
   Do the core processing components from drawing descriptions match
   what this CPC class describes?
   - 1.0: Class directly covers the type of components identified
   - 0.5: Class is adjacent to the component type
   - 0.0: Class describes completely different component types

4. CROSS-DOMAIN LEAKAGE CHECK
   Primary domain: {phase1_data.get("primary_domain", {}).get("name", "unknown")}
   ({phase1_data.get("primary_domain", {}).get("cpc_class", "unknown")})
   If candidate is in SAME domain as primary → STRONG PASS
   If candidate is in RELATED domain → CONDITIONAL PASS
   If candidate is in UNRELATED domain → STRONG FAIL

5. METHOD VS APPARATUS CHECK
   Does the candidate match the claim type (method/apparatus/both)?

6. SPECIFICITY FIT
   too_broad | appropriate | too_narrow

7. CONTRADICTION CHECK
   Does anything in the candidate description contradict the invention?
   Pay special attention to domain-specific terms that do NOT match the invention
   (e.g., image/video terms for a non-image invention).

=== DECISION RULES ===

PASS if:
- function_alignment >= 0.6 AND
- context_alignment >= 0.5 AND
- component_alignment >= 0.4 (or 0.0 if no drawing descriptions available) AND
- cross_domain_leakage == false AND
- method_apparatus_aligned == true AND
- specificity_fit != "too_broad"

FAIL if any critical check fails.
If score_margin < 0.1, mark confidence as LOW.

OUTPUT FORMAT (strict JSON):
{{
  "decision": "PASS|FAIL",
  "confidence": "high|medium|low",
  "scores": {{
    "function_alignment": 0.9,
    "context_alignment": 0.8,
    "component_alignment": 0.7,
    "cross_domain_leakage": false,
    "method_apparatus_aligned": true
  }},
  "specificity_fit": "appropriate",
  "contradictions": [],
  "reasoning": "One paragraph explaining the validation decision",
  "rejection_reason": "If FAIL, explain specifically why"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# =============================================================================
# PHASE 6 — PER-CLAIM RECONCILIATION
# =============================================================================


def reconciliation_prompt(
    validated_codes: list,
    rejected_codes: list,
    per_claim_classifications: list,
) -> str:
    """
    Phase 6: Per-claim reconciliation after Phase 5 validation.

    After Phase 5 produces a rejected-codes list, this prompt updates per-claim
    CPC assignments to remove rejected codes and replace them with the best
    matching validated alternative.

    Without this step, a claim could remain assigned a code that Phase 5
    already rejected — making Phase 5's work invisible to the final output.
    """
    validated_list = "\n".join(
        f"- {v.get('symbol', '')}: {v.get('title', '')}" for v in validated_codes
    )
    rejected_list = "\n".join(
        f"- {r.get('symbol', '')}: {r.get('rejection_reason', 'No reason given')}"
        for r in rejected_codes
    )
    per_claim_list = "\n".join(
        f"- Claim {p.get('claim_number', '')}: {', '.join(p.get('cpc_classes', []))}"
        for p in per_claim_classifications
    )

    return f"""You are reconciling claim-level CPC classifications with validated Phase 5 results.

=== VALIDATED CPC CODES (PASSED Phase 5) ===
{validated_list}

=== REJECTED CPC CODES (FAILED Phase 5) ===
{rejected_list}

=== CURRENT PER-CLAIM CLASSIFICATIONS ===
{per_claim_list}

=== TASK ===
For each claim:
1. Remove any CPC codes that were rejected in Phase 5
2. If removing a code leaves the claim with no CPC classes, replace with the
   best-matching validated code
3. Ensure alignment with claim type (method vs apparatus)
4. Keep provisional codes only if they match a validated code

Output format (strict JSON):
{{
  "reconciled_claims": [
    {{
      "claim_number": 1,
      "claim_type": "independent",
      "final_cpc": ["G06F16/00", "G06N3/08"],
      "reasoning": "Aligned with validated codes and claim function"
    }}
  ],
  "changes_made": [
    "Claim 2: Rejected G06V10/75, replaced with G06F16/00"
  ]
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""
