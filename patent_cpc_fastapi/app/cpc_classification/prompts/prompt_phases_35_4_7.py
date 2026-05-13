"""
prompt_phases_35_4_7.py

LLM prompts for three specific validation/control points in the pipeline:

  - Phase 3.5 : Tie-breaker between top candidates when score margin is too small
                for deterministic rules to separate them.

  - Phase 4   : Sanity check on clustering-produced hypotheses — binary YES/NO only.

  - Phase 7   : Primary consistency checker — are the final selected CPC codes
                semantically coherent with the invention?

DESIGN RULES for all three prompts:
  1. Constrained input  — LLM receives only the minimum context needed
  2. Narrow output      — binary decision, short reasoning, no free-form classification
  3. LLM never generates new CPC codes — it only evaluates what the pipeline produced
  4. LLM never sees the full candidate list — only the specific candidates in question
  5. Output is always strict JSON — no markdown, no prose outside the object
"""


# =============================================================================
# PHASE 3.5 — TIE-BREAKER
# =============================================================================

def phase35_tiebreaker_prompt(
    invention_profile: dict,
    tied_candidates: list,
) -> str:
    """
    Phase 3.5 LLM tie-breaker.

    Called ONLY when the decision tree produces two or three candidates
    whose scores are within a margin too small to separate deterministically
    (typically < 0.05 score difference).

    The LLM receives:
      - A minimal invention profile (technical object, core function, primary domain)
      - The tied candidates (maximum 3) with their CPC symbol and title
      - It does NOT receive scores, the full candidate list, or any pipeline internals

    The LLM must:
      - Select ONE winner
      - Provide a single-sentence justification grounded in the invention description
      - Assign a confidence level (high / medium / low)
      - If genuinely indistinguishable, say so explicitly rather than guessing

    What the LLM must NOT do:
      - Suggest new CPC codes not in the tied list
      - Re-rank the full candidate list
      - Comment on the scoring system
      - Output anything outside the JSON object
    """

    candidates_block = "\n".join(
        f"  Candidate {i + 1}:\n"
        f"    Code  : {c.get('symbol', '')}\n"
        f"    Title : {c.get('title', '')}\n"
        f"    Context: {c.get('full_context', c.get('title', ''))}"
        for i, c in enumerate(tied_candidates[:3])
    )

    return f"""You are acting as a tie-breaker in a patent CPC classification pipeline.

The deterministic scoring system has produced {len(tied_candidates[:3])} candidates with
scores too close to separate by rule. Your job is to read the invention description
and select the single best-matching candidate.

STRICT CONSTRAINTS:
- You must choose from the candidates listed below ONLY
- You must NOT suggest any CPC code not in this list
- You must NOT re-classify the invention from scratch
- Your justification must reference specific language from the invention profile
- If the candidates are genuinely equivalent, say so — do not guess

=== INVENTION PROFILE ===
Technical Object : {invention_profile.get('technical_object', '')}
Core Function    : {invention_profile.get('core_function', '')}
Problem Solved   : {invention_profile.get('problem_solved', '')}
Primary Domain   : {invention_profile.get('primary_domain', {}).get('name', '')}
                   ({invention_profile.get('primary_domain', {}).get('cpc_class', '')})
Essential Features: {invention_profile.get('essential_features', [])}

=== TIED CANDIDATES (maximum 3) ===
{candidates_block}

=== YOUR TASK ===

Step 1 — Read the technical object and core function carefully.

Step 2 — For each candidate, ask:
  a) Does the CPC title describe what this invention IS or DOES?
  b) Does the CPC title match the primary domain of the invention?
  c) Is the specificity level appropriate — not too broad, not too narrow?

Step 3 — Select the single best candidate, or declare a tie if indistinguishable.

Step 4 — Write one sentence of justification that references specific words
         from the technical object or core function above.

=== DECISION RULES ===

PREFER the candidate whose title:
  - Contains language closest to the technical object (noun match)
  - Contains language closest to the core function (verb match)
  - Belongs to the primary domain identified in Phase 1

REJECT a candidate if its title:
  - Describes a completely different domain from the primary domain
  - Is so broad it would apply to any computing invention
  - Introduces domain-specific terms not present in the invention profile
    (e.g., image/video terms for a non-image invention)

DECLARE TIE if:
  - All candidates are from the correct domain AND equally specific
  - No language in the invention profile distinguishes between them
  - Choosing would require speculation beyond the invention description

=== OUTPUT FORMAT (strict JSON) ===
{{
  "decision": "CANDIDATE_1" | "CANDIDATE_2" | "CANDIDATE_3" | "TIE",
  "winner_symbol": "G06G7/00" | null,
  "confidence": "high" | "medium" | "low",
  "justification": "One sentence referencing specific invention language",
  "tie_reason": "Only populated if decision is TIE — explain why candidates are indistinguishable"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# =============================================================================
# PHASE 4 — HYPOTHESIS SANITY CHECK
# =============================================================================

def phase4_sanity_check_prompt(
    invention_profile: dict,
    hypotheses: list,
) -> str:
    """
    Phase 4 LLM sanity check.

    Called AFTER the clustering algorithm has produced its hypotheses (maximum 2).
    The LLM answers one binary question per hypothesis:
    "Does this hypothesis label make technical sense for this invention?"

    This catches the specific failure mode where clustering produces a
    statistically strong but semantically absurd grouping — for example,
    a strong cluster of G06T (image processing) codes for an invention
    that has nothing to do with images.

    The LLM receives:
      - Minimal invention profile
      - The hypothesis labels and their top representative CPC codes (max 3 per hypothesis)

    The LLM must:
      - Answer YES or NO per hypothesis
      - If NO, identify what is wrong in one sentence
      - NOT suggest replacement hypotheses
      - NOT re-cluster or re-score

    What the LLM must NOT do:
      - Propose new hypotheses
      - Generate new CPC codes
      - Comment on scores or cluster sizes
      - Output anything outside the JSON object
    """

    hypotheses_block = ""
    for i, hyp in enumerate(hypotheses[:2]):
        label = hyp.get('label', f'Hypothesis {i + 1}')
        domain = hyp.get('domain', 'unknown')
        codes = hyp.get('representative_codes', [])
        codes_text = "\n    ".join(
            f"- {c.get('symbol', '')} : {c.get('title', '')}"
            for c in codes[:3]
        )
        hypotheses_block += f"""
Hypothesis {i + 1}:
  Label          : {label}
  Domain         : {domain}
  Representative codes:
    {codes_text}
"""

    return f"""You are performing a sanity check on patent classification hypotheses.

A clustering algorithm has grouped CPC candidates into the hypotheses below.
Your job is to check whether each hypothesis makes technical sense for this invention.
You are NOT re-classifying the invention. You are only checking for obvious errors.

STRICT CONSTRAINTS:
- Answer YES or NO per hypothesis — no partial answers
- Do NOT suggest replacement hypotheses or new CPC codes
- Your reasoning must reference specific language from the invention profile
- A hypothesis is WRONG only if it is clearly from the wrong domain —
  not just because it is imperfect

=== INVENTION PROFILE ===
Technical Object : {invention_profile.get('technical_object', '')}
Core Function    : {invention_profile.get('core_function', '')}
Primary Domain   : {invention_profile.get('primary_domain', {}).get('name', '')}
                   ({invention_profile.get('primary_domain', {}).get('cpc_class', '')})
System Context   : {invention_profile.get('system_context', '')}

=== HYPOTHESES TO CHECK ===
{hypotheses_block}

=== YOUR TASK ===

For EACH hypothesis, answer:

QUESTION: Does this hypothesis domain make technical sense for this invention?

YES if:
  - The hypothesis domain matches or is adjacent to the primary domain
  - The representative CPC titles describe something related to the invention
  - A patent examiner would consider this domain plausible for this invention

NO if:
  - The hypothesis domain is completely unrelated to the invention
  - The representative CPC titles describe technology absent from the invention
    (e.g., image processing titles for an invention about analog circuits)
  - The domain contradicts the primary domain with no supporting evidence

EXAMPLES of clearly wrong hypotheses to reject:
  - Invention is about analog computing → hypothesis is G06T (image processing) : NO
  - Invention is about speech recognition → hypothesis is E21B (drilling) : NO
  - Invention is about neural networks → hypothesis is F16J (sealing) : NO

EXAMPLES of acceptable hypotheses even if imperfect:
  - Invention is about AI → hypothesis includes both G06N and G06F : YES
    (G06F is adjacent and commonly co-occurs with G06N)
  - Invention is about automotive control → hypothesis includes B60W and G05B : YES
    (G05B control systems is adjacent to automotive control)

=== OUTPUT FORMAT (strict JSON) ===
{{
  "hypothesis_checks": [
    {{
      "hypothesis_number": 1,
      "label": "string — the hypothesis label as given",
      "verdict": "YES" | "NO",
      "reasoning": "One sentence referencing specific invention language",
      "severity": "acceptable" | "warning" | "reject"
    }}
  ],
  "overall_assessment": "PASS" | "WARN" | "FAIL",
  "overall_reasoning": "One sentence summarising the check result",
  "action_required": "proceed" | "discard_hypothesis_N" | "flag_for_review"
}}

Severity levels:
  acceptable : Hypothesis is plausible — proceed
  warning    : Hypothesis is questionable but not clearly wrong — proceed with caution
  reject     : Hypothesis is clearly from the wrong domain — discard it

Overall assessment:
  PASS  : All hypotheses are acceptable
  WARN  : At least one hypothesis has a warning
  FAIL  : At least one hypothesis must be rejected

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# =============================================================================
# PHASE 7 — CONSISTENCY CHECK (PRIMARY CHECKER)
# =============================================================================

def phase7_consistency_prompt(
    invention_profile: dict,
    selected_codes: list,
) -> str:
    """
    Phase 7 LLM consistency check.

    This is the primary consistency checker — not a supplementary rule check.
    It asks whether the final selected CPC codes are semantically coherent
    with each other and with the invention.

    This phase is better suited to LLM than deterministic rules because:
    - Coherence between codes is a language understanding task
    - Cross-domain inventions produce combinations that rule tables cannot anticipate
    - The LLM can reason about WHY codes fit or conflict, not just WHETHER they match

    The LLM receives:
      - Full invention profile (technical object, core function, essential features,
        component glossary from drawings if available)
      - The selected CPC codes with titles (maximum 6)

    The LLM must:
      - Assess coherence of the code set as a whole
      - Identify the primary code (the one that best represents the invention)
      - Flag any code from the wrong domain
      - Flag any redundant or contradictory pair
      - Recommend keep / demote / remove per code

    What the LLM must NOT do:
      - Generate new CPC codes
      - Re-run classification
      - Output anything outside the JSON object
    """

    codes_block = "\n".join(
        f"  Code {i + 1}: {c.get('symbol', '')} — {c.get('title', '')}"
        for i, c in enumerate(selected_codes[:6])
    )

    component_context = ""
    glossary = invention_profile.get('component_glossary', [])
    core_components = [
        c for c in glossary
        if c.get('role') == 'core_processing' and c.get('trust') == 'high'
    ]
    if core_components:
        names = ", ".join(
            f"{c.get('name', '')} [{c.get('ref', '')}]"
            for c in core_components[:4]
        )
        component_context = f"Core processing components (from drawing descriptions): {names}"

    return f"""You are performing the final consistency check on a set of CPC classification codes
selected for a patent invention.

Your job is to assess whether these codes are semantically coherent — do they logically
belong together and correctly represent the invention?

This is a language understanding task. Use your knowledge of what CPC sections cover
to reason about whether the code combination makes technical sense.

STRICT CONSTRAINTS:
- You must NOT generate new CPC codes or suggest replacements from memory
- You must only evaluate the codes provided
- Every recommendation must reference specific language from the invention profile
- Output is strict JSON only

=== INVENTION PROFILE ===
Technical Object  : {invention_profile.get('technical_object', '')}
Core Function     : {invention_profile.get('core_function', '')}
Problem Solved    : {invention_profile.get('problem_solved', '')}
System Context    : {invention_profile.get('system_context', '')}
Primary Domain    : {invention_profile.get('primary_domain', {}).get('name', '')}
                    ({invention_profile.get('primary_domain', {}).get('cpc_class', '')})
Essential Features: {invention_profile.get('essential_features', [])}
Data Flow         : {invention_profile.get('data_flow', 'Not available')}
{component_context}

=== SELECTED CPC CODES (maximum 6) ===
{codes_block}

=== YOUR TASK ===

STEP 1 — IDENTIFY THE PRIMARY CODE
Which single code best represents the core inventive contribution?
The primary code should match:
  - The technical object (what the invention IS)
  - The core function (what the invention DOES)
  - The primary domain from Phase 1

STEP 2 — CHECK EACH CODE INDIVIDUALLY
For each code ask:
  a) Is this code from the correct domain for this invention?
  b) Does this code describe something present in the invention?
  c) Is this code at the right level of specificity?
     (too broad = applies to any invention in the field;
      too narrow = more specific than what the invention claims)

STEP 3 — CHECK THE CODE SET AS A WHOLE
  a) Do these codes form a coherent technical picture of the invention?
  b) Are there any contradictory pairs?
     (e.g., one code for image processing and one for audio processing
      when the invention is about neither)
  c) Are any codes redundant?
     (e.g., two codes covering the same concept at different hierarchy levels
      where only one is needed)
  d) Does the set cover the invention's essential features?

STEP 4 — FLAG DOMAIN LEAKAGE
Domain leakage = a code from a completely unrelated domain that entered
the selection because of superficial keyword overlap.

Common leakage patterns to check:
  - G06T (image/video) in a non-image invention
  - G10L (speech/audio) in a non-audio invention
  - G06N (AI/neural networks) in a non-AI invention
  - H04L (telecommunications) in a non-comms invention
  - A61 (medical) in a non-medical invention
  - E21 (drilling/mining) in a non-industrial invention

If any code matches a leakage pattern AND the invention profile contains
no evidence for that domain → flag as leakage.

STEP 5 — MAKE RECOMMENDATIONS
For each code: keep | demote_to_secondary | remove
  keep              : Code is correct and at the right level
  demote_to_secondary : Code is correct domain but not the primary contribution
  remove            : Code is from wrong domain or is redundant

=== OUTPUT FORMAT (strict JSON) ===
{{
  "primary_code": "G06G7/00",
  "primary_code_reasoning": "One sentence explaining why this is primary",

  "code_assessments": [
    {{
      "symbol": "string",
      "domain_correct": true | false,
      "specificity": "too_broad" | "appropriate" | "too_narrow",
      "domain_leakage": true | false,
      "leakage_reason": "string or empty",
      "redundant_with": "symbol of other code or empty",
      "recommendation": "keep" | "demote_to_secondary" | "remove",
      "recommendation_reasoning": "One sentence referencing invention language"
    }}
  ],

  "set_coherence": "coherent" | "minor_issues" | "incoherent",
  "set_coherence_reasoning": "One sentence on the overall code set quality",

  "contradictions": [
    {{
      "code_a": "string",
      "code_b": "string",
      "reason": "Why these two codes contradict or do not belong together"
    }}
  ],

  "coverage_gaps": [
    "Essential feature not covered by any selected code — describe the gap"
  ],

  "final_verdict": "PASS" | "WARN" | "FAIL",
  "final_reasoning": "Two sentences maximum summarising the consistency assessment",

  "action": "proceed" | "remove_codes_and_proceed" | "flag_for_review"
}}

Final verdict:
  PASS : Codes are coherent, primary is correct, no leakage
  WARN : Minor issues (one borderline code, slight specificity mismatch) — proceed
  FAIL : Domain leakage detected, or primary code is wrong domain, or incoherent set

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""