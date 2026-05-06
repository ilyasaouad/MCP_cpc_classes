"""
prompts.py - Improved patent classification prompts.

Weaknesses addressed:
  W1  - Gate injection on LLM strategy confidence; probabilistic domain inference replaces hardcoded injection
  W2  - Calibrate domain multipliers; default to 1.2 for unknown domains
  W3  - Section-aware extraction with downweighted background/prior-art
  W4  - Term-density guard for specificity bonus
  W5  - Multi-pass validation (one candidate per prompt)
  W6  - Score margin and confidence level passed to Phase 5
  W7  - Per-claim reconciliation after Phase 5 rejection
  W8  - Method vs apparatus detection and routing
"""

import re


def label_claims(raw_claims_text: str) -> str:
    """
    Parse raw claims text and return a version where every claim is prefixed
    with [INDEPENDENT] or [DEPENDENT: ref <n>].
    """
    dependency_pattern = re.compile(
        r"\baccording to claim[s]?\s+([\d, ]+(?:or\s+\d+)?)"
        r"|\bof claim[s]?\s+([\d, ]+(?:or\s+\d+)?)",
        re.IGNORECASE,
    )

    lines = raw_claims_text.strip().splitlines()
    labeled_lines = []
    current_claim_lines = []
    current_claim_num = None

    def flush(claim_lines, claim_num):
        if not claim_lines:
            return
        block = "\n".join(claim_lines)
        match = dependency_pattern.search(block)
        if match:
            ref = (match.group(1) or match.group(2)).strip()
            labeled_lines.append(f"[DEPENDENT: ref claim {ref}]")
        else:
            labeled_lines.append("[INDEPENDENT]")
        labeled_lines.extend(claim_lines)
        labeled_lines.append("")

    claim_start = re.compile(r"^\s*(\d+)\.\s")

    for line in lines:
        m = claim_start.match(line)
        if m:
            flush(current_claim_lines, current_claim_num)
            current_claim_num = int(m.group(1))
            current_claim_lines = [line]
        else:
            current_claim_lines.append(line)

    flush(current_claim_lines, current_claim_num)
    return "\n".join(labeled_lines)


def _detect_sections(text: str) -> dict:
    """
    Detect patent sections and return a mapping with start indices.
    """
    section_markers = {
        "abstract": r"(?i)^\s*(abstract|summary of the invention|brief summary)",
        "background": r"(?i)^\s*(background|prior art|related art|field of the invention)",
        "summary": r"(?i)^\s*(summary|summary of invention|detailed description)",
        "claims": r"(?i)^\s*(claims?|we claim|what is claimed)",
        "detailed": r"(?i)^\s*(detailed description|description of.*embodiment|detailed description of.*invention)",
    }

    sections = {"full_text": text}
    lines = text.splitlines()

    for section_name, pattern in section_markers.items():
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                sections[section_name] = "\n".join(lines[i:])
                break

    return sections


SECTION_WEIGHTS = {
    "abstract": 0.6,
    "background": 0.2,
    "summary": 1.0,
    "detailed": 0.9,
    "claims": 1.2,
    "unknown": 0.5,
}


IMPORTANCE_RUBRIC = """
Importance score calibration (use this scale strictly, adjusted by section weight):
  Base scale:
    10 - Term appears in an independent claim AND is the core inventive feature
     9 - Term appears in an independent claim and is a key structural/functional element
     8 - Term is in the summary/detailed description AND directly enables the solution
     7 - Term is in the description and important context for the solution
     5 - Term is supporting/background context
     3 - Term is generic domain vocabulary (keep only if no better term)
     1 - Term is too generic to contribute (should be excluded)

  Section adjustments (multiply base by section weight, then round):
    Claims:           base × 1.2  (max 10)
    Summary:          base × 1.0  (no change)
    Detailed desc:    base × 0.9
    Abstract:         base × 0.6
    Background:       base × 0.2  (max 3 unless reinforced in claims)

  Never assign 10 to background-only terms.
  Never assign below 7 to independent-claim terms unless purely generic.
"""


def phase1_prompt(cpc_hints: str, labeled_claims: str, description: str) -> str:
    """
    Phase 1 prompt: Examiner-grade CPC classification with section-aware extraction.
    """

    return f"""You are a patent classification expert trained in CPC (Cooperative Patent Classification).

Your task is to analyze the patent description and claims and produce a structured JSON output
for CPC classification. Follow each step strictly and in order.

=== CPC REFERENCE (AUTHORITATIVE) ===
The following CPC hierarchy was retrieved from the EPO Linked Open Data API.
You MUST use it to verify every class you select. Only output classes that appear
in this reference. If your reasoning leads to a class not present here, revise your
choice to the closest ancestor that IS present.

{cpc_hints}

=== CLAIMS (PRE-LABELED) ===
The claims below have been pre-processed. Each claim is prefixed with either
[INDEPENDENT] or [DEPENDENT: ref claim N].

{labeled_claims}

=== DESCRIPTION ===
{description}

{IMPORTANCE_RUBRIC}

---

STEP 0 — SECTION WEIGHTING (CRITICAL)

The patent description contains sections with varying reliability:
- Abstract: 0.6 (summary, but may be broad)
- Background/Prior Art: 0.2 (often describes existing technology, NOT the invention)
- Summary of Invention: 1.0 (high signal)
- Detailed Description: 0.9 (implementation details)
- Claims: 1.2 (strongest signal, especially independent claims)

When extracting terms, DOWNWEIGHT background/prior-art terms unless they are
explicitly reused in claims or summary.

---

STEP 1 — TECHNICAL UNDERSTANDING

Extract:
- technical_object   : What is the invention? (1-2 sentences, concrete and specific)
- problem_solved     : What specific technical problem is addressed? (not generic)
- solution_summary   : How does the invention solve the problem? Focus on the mechanism.

Rules:
- Avoid vague wording
- Avoid structural descriptions unless structure IS the functional differentiator

---

STEP 2 — SYSTEM CONTEXT

Identify the broader technical system or industry in which the invention operates.

Rules:
- Must describe an industry/application domain, NOT a component
- Ask: "What industry would buy/use this invention?"
- Must be a system or application (e.g., "database search engines"), not a part

Output: system_context

---

STEP 3 — CORE TECHNICAL FUNCTION (CRITICAL)

Identify the PRIMARY function performed by the invention (what it DOES, not what it looks like).

Rules:
- Focus on the action/operation, not the physical form
- Example: "context-driven retrieval of data fragments" (good)
- Example: "a device comprising a housing and processor" (bad - structural)

Geometry note: Exclude geometric descriptions UNLESS the geometry is the functional
differentiator (e.g., a blade profile that creates a specific aerodynamic effect). In that
case include the geometric term with explicit justification of its functional role.

Output: core_function

---

STEP 4 — CLAIM TYPE ANALYSIS (NEW)

For EACH independent claim, classify its type:

- METHOD: The claim describes a process, algorithm, or sequence of steps
- APPARATUS/SYSTEM: The claim describes a device, system, or physical structure
- BOTH: The claim mixes method and apparatus elements

Also extract per claim:
- claim_core_function: The specific function performed by THIS claim
- claim_specific_features: Technical features unique to this claim (not in other claims)

Output format per claim:
{{"claim_number": 1, "type": "METHOD", "core_function": "...", "features": ["..."]}}

---

STEP 5 — ESSENTIAL TECHNICAL TERMS (SECTION-AWARE)

Extract terms with importance (1-10), recording the source section.

Extraction rules:
- From claims (INDEPENDENT only) → base importance 9-10
- From summary of invention → base importance 7-9
- From detailed description → base importance 5-8
- From background/prior art → max base importance 3 (unless reinforced in claims/summary)
- From abstract → base importance 4-7

Apply section weights:
- Multiply base importance by section weight
- Round to nearest integer
- Cap at 10

Each term: {{ "term": "...", "importance": 8, "justification": "...", "source_section": "summary" }}

Source section options: "claims", "summary", "detailed_description", "abstract", "background"

---

STEP 6 — MULTI-INVENTION DETECTION

Some patents contain multiple independent claims covering DISTINCT inventions
(e.g., Claim 1 is a device, Claim 17 is a method with a different technical focus).

Instructions:
1. List all [INDEPENDENT] claim numbers found in the labeled claims
2. Group them by technical focus:
   - If all independent claims cover the same invention -> single_invention: true
   - If claims cover meaningfully distinct inventions -> single_invention: false,
     and describe each invention group

Output:
- independent_claim_numbers: [list of ints]
- single_invention: true/false
- invention_groups: [ {{ "claims": [ints], "focus": "brief description" }} ]
  (one group if single_invention is true)

---

STEP 7 — CLASSIFICATION STRATEGY

Choose ONE of three strategies:

  "system-first"   - The invention is a complete apparatus/machine for a SPECIFIC industry.
                     No other industry would use this exact invention.
                     PRIMARY class = application domain (E21B, B60L, A61B ...)

  "function-first" - The invention is a generic component usable across MULTIPLE industries.
                     PRIMARY class = core function (F16J, F16K, F04, G06N ...)

  "hybrid"         - The invention has a novel functional mechanism AND is tied to a specific
                     application. Both domain AND function classes are co-primary.
                     List them in order of specificity.

Decision rule:
  Ask: "Could this exact invention be deployed in two or more unrelated industries without
  modification?" If NO -> system-first. If YES -> function-first. If the functional
  innovation is as significant as the application specificity -> hybrid.

Self-consistency check (mandatory before proceeding):
  Re-read your technical_object from Step 1.
  Confirm that your chosen strategy is consistent with that description.
  If there is tension, revise system_context or core_function before continuing.

Output:
- classification_strategy: "system-first" | "function-first" | "hybrid"
- strategy_reasoning: explanation referencing technical_object, system_context, core_function
- consistency_check: "consistent" | "revised - [what was revised and why]"

---

STEP 7b — DOMAIN GUIDANCE FOR COMPUTING AND SOFTWARE PATENTS (CRITICAL)

If the invention involves ANY of the following, use these CPC mappings:

SOFTWARE / COMPUTING DOMAINS:
- LLM, chatbot, natural language processing, dialog system, conversational AI
  → PRIMARY: G06F16 (information retrieval), G06F17 (digital computing)
  → SECONDARY: G06N3 (neural networks), G06F40 (natural language processing)
  → AVOID: G06F11 (fault tolerance), G06F3 (input devices), G06F9 (data transfer)

- Query-response system, search engine, information retrieval
  → G06F16 (information retrieval, database structures)
  → G06F16/30 (digital computing for information retrieval)
  → G06F16/3329 (query formulation with natural language)
  → G06F16/90332 (natural language interfaces)

- API, client-server communication, network messaging
  → H04L29/06 (communication control), H04L67/02 (web-based client-server)
  → NOT G06F9/543 (clipboard/data exchange - that's for local OS-level transfer)

- Voice/speech processing in dialog systems
  → G10L15/22 (speech recognition), G10L15/24 (dialog systems)
  → PLUS G06F16 or G06F17 for the dialog management aspect

ANTI-PATTERNS — Do NOT map these words literally:
- "exchange" in dialog context (role exchange) → NOT G06F9/543 (data exchange)
- "response" in chat context → NOT G06F11/277 (fault response)
- "system" in software context → NOT G06F11/182 (redundant systems)
- "user" in chatbot context → NOT G06F3/015 (user input devices)
- "transfer" of messages → NOT G06F9/543 unless it's OS-level clipboard/DDE

---

STEP 8 — INITIAL CPC CLASS HYPOTHESES (SOFT)

Suggest 3-5 CPC classes (4-character codes like F01P, F16K, B60L, E21B).

IMPORTANT:
- These are hypotheses, NOT enforced
- Assign confidence (0.0-1.0) per class
- The LLM can override these later if evidence is strong

Output:
- class_hypotheses: [ {{ "class": "G06F", "confidence": 0.7, "reasoning": "..." }} ]

---

STEP 9 — NEGATIVE SIGNALS (SOFT)

Extract terms and domains that this patent is clearly NOT about.

Rules:
- Be specific - avoid generic terms like "device" or "system"
- Focus on domains that could be confused with the correct classification
- At least 5 negative signals, at least 2 negative domains
- Assign confidence (0.0-1.0) per signal, NOT absolute exclusions

Output:
- negative_signals: [ {{ "term": "image processing", "confidence": 0.8 }} ]
- negative_domains: [ {{ "domain": "computer vision", "confidence": 0.9 }} ]
- negative_reasoning: brief explanation

---

STEP 10 — PER-CLAIM PRELIMINARY CPC

For EACH claim (independent + significant dependents), assign 1-2 CPC subclasses.

Rules:
- Claim 1 (independent) maps to the PRIMARY/broadest classes for the overall invention
- Dependent claims MAY map to the same classes OR additional classes if they add novel technical elements
- Use specific 6-8 digit subclasses where possible (e.g., G06F16/00, not just G06F)
- Focus on the NOVEL contribution of each claim
- If a dependent claim does NOT add a new technical area, map it to the same classes as its parent claim
- Mark all as "provisional": true (subject to Phase 5/6 validation)

For each claim, output:
- claim_number: integer
- claim_type: "independent" or "dependent"
- parent_claim: null for independent, or the claim number it depends on
- cpc_classes: [1-2 specific CPC subclasses]
- reasoning: Brief explanation
- provisional: true

---

OUTPUT FORMAT (STRICT JSON - no markdown, no text outside the JSON object)

{{
  "technical_object": "string",
  "problem_solved": "string",
  "solution_summary": "string",
  "system_context": "string",
  "core_function": "string",
  "claim_analysis": [
    {{
      "claim_number": 1,
      "type": "METHOD",
      "core_function": "string",
      "features": ["string"]
    }}
  ],
  "independent_claim_numbers": [1, 17],
  "single_invention": true,
  "invention_groups": [
    {{
      "claims": [1, 17],
      "focus": "string"
    }}
  ],
  "classification_strategy": "system-first",
  "strategy_reasoning": "string",
  "consistency_check": "consistent",
  "class_hypotheses": [
    {{"class": "G06F", "confidence": 0.7, "reasoning": "string"}}
  ],
  "cpc_classes": ["E21B", "F16J"],
  "cpc_sections": ["E", "F"],
  "cpc_reasoning": "string",
  "terms": [
    {{
      "term": "string",
      "importance": 8,
      "justification": "string",
      "source_section": "summary"
    }}
  ],
  "negative_signals": [
    {{"term": "string", "confidence": 0.6}}
  ],
  "negative_domains": [
    {{"domain": "string", "confidence": 0.6}}
  ],
  "negative_reasoning": "string",
  "claim_classifications": [
    {{
      "claim_number": 1,
      "claim_type": "independent",
      "parent_claim": null,
      "cpc_classes": ["G01N1/00", "B01D35/00"],
      "reasoning": "string",
      "provisional": true
    }}
  ]
}}
"""


def domain_inference_prompt(phase1_data: dict) -> str:
    """
    Phase 1b replacement: Probabilistic domain inference.
    Replaces hardcoded class injection with LLM-based probability estimation.
    """
    terms = phase1_data.get("terms", [])
    term_list = "\n".join(
        f"- {t.get('term', '')} (importance: {t.get('importance', 5)}, source: {t.get('source_section', 'unknown')})"
        for t in terms[:15]
    )

    hypotheses = phase1_data.get("class_hypotheses", [])
    hypo_list = "\n".join(
        f"- {h.get('class', '')} (confidence: {h.get('confidence', 0.5)})"
        for h in hypotheses
    )

    return f"""You are estimating the probability that various CPC domains are relevant to this invention.

Given the extracted terms, system context, and initial hypotheses, estimate the probability
(0.0-1.0) that each CPC domain is relevant.

=== INVENTION PROFILE ===
Technical Object: {phase1_data.get("technical_object", "")}
System Context: {phase1_data.get("system_context", "")}
Core Function: {phase1_data.get("core_function", "")}
Classification Strategy: {phase1_data.get("classification_strategy", "")}

=== EXTRACTED TERMS (top 15) ===
{term_list}

=== INITIAL HYPOTHESES ===
{hypo_list}

=== TASK ===
Estimate probabilities for these CPC domains (and any others you deem relevant):
- G06F (Computing / data processing)
- G06N (AI / neural networks)
- G06K (Pattern recognition / data representation)
- G06V (Image/video recognition)
- G06T (Image data processing)
- H04L (Digital information transmission)
- E21B (Earth drilling / mining)
- F16J (Sealing)
- F16K (Valves)
- F01P (Cooling)
- B60L (Electrical propulsion)
- B01D (Separation)
- G01N (Investigating materials)
- Any other domain you think is relevant

Rules:
- Base probabilities ONLY on technical function and context, NOT on keyword matching alone
- Multiple domains can have high probability
- Do NOT force inclusion of any domain
- Use the classification strategy as guidance: system-first boosts domain codes, function-first boosts function codes

Output format:
{{
  "domain_probabilities": [
    {{"class": "G06F", "probability": 0.75, "reasoning": "Invention involves data processing and retrieval"}}
  ],
  "primary_domain": "G06F",
  "primary_confidence": 0.75,
  "reasoning": "Brief explanation of the top domains"
}}

Output ONLY valid JSON, no markdown, no text outside the object.
"""


def semantic_scoring_prompt(
    core_function: str, system_context: str, candidate: dict
) -> str:
    """
    LLM-assisted semantic scoring for Phase 2.
    Evaluates how well a CPC subgroup matches the invention.
    """
    return f"""You are assisting a CPC scoring engine.

=== INVENTION ===
Core Function: {core_function}
System Context: {system_context}

=== CPC CANDIDATE ===
Code: {candidate.get("symbol", "")}
Title: {candidate.get("title", "")}
Context: {candidate.get("full_context", candidate.get("title", ""))}

=== EVALUATION ===
Rate the following on a scale of 0.0 to 1.0:

1. semantic_similarity: How semantically similar is the CPC description to the invention?
2. function_match: Does the CPC class describe the core function of the invention?
3. context_match: Does the CPC class fit the system/application context?
4. specificity_fit: Is the specificity level appropriate?
   Options: "too_broad", "appropriate", "too_narrow"

Output:
{{
  "semantic_similarity": 0.82,
  "function_match": 0.90,
  "context_match": 0.75,
  "specificity_fit": "appropriate",
  "reasoning": "Brief explanation"
}}

Output ONLY valid JSON, no markdown, no text outside the object.
"""


def validation_prompt_single(
    phase1_data: dict, candidate: dict, score_margin: float, confidence_level: str
) -> str:
    """
    Phase 5: Single-candidate validation prompt (multi-pass).
    Validates one CPC candidate against the invention profile.
    """
    strategy = phase1_data.get("classification_strategy", "unknown")
    claim_analysis = phase1_data.get("claim_analysis", [])

    # Build method vs apparatus context
    method_claims = [c for c in claim_analysis if c.get("type") == "METHOD"]
    apparatus_claims = [c for c in claim_analysis if c.get("type") == "APPARATUS"]
    mixed_claims = [c for c in claim_analysis if c.get("type") == "BOTH"]

    method_apparatus_note = ""
    if method_claims and not apparatus_claims:
        method_apparatus_note = "NOTE: This patent contains ONLY method claims. The CPC class MUST cover methods/processes, not just apparatus."
    elif apparatus_claims and not method_claims:
        method_apparatus_note = "NOTE: This patent contains ONLY apparatus claims. The CPC class MUST cover systems/devices, not just methods."
    elif mixed_claims or (method_claims and apparatus_claims):
        method_apparatus_note = "NOTE: This patent contains both method and apparatus claims. The CPC class should ideally cover both, or at least not contradict either."

    return f"""You are validating a single CPC classification candidate against an invention's true technical nature.

=== INVENTION PROFILE ===
Technical Object: {phase1_data.get("technical_object", "")}
Problem Solved: {phase1_data.get("problem_solved", "")}
Core Function: {phase1_data.get("core_function", "")}
System Context: {phase1_data.get("system_context", "")}
Classification Strategy: {strategy}

=== SCORING CONTEXT ===
Confidence Level: {confidence_level}
Score Margin (top1 - top2): {score_margin:.4f}
{method_apparatus_note}

=== CANDIDATE TO VALIDATE ===
Code: {candidate.get("symbol", "")}
Title: {candidate.get("title", "")}
Description: {candidate.get("full_context", candidate.get("title", ""))}

=== VALIDATION CRITERIA ===

1. FUNCTION ALIGNMENT (0.0-1.0)
   Does this CPC class describe the CORE FUNCTION of the invention?
   - 1.0 = Perfect match: class directly covers the primary operation
   - 0.5 = Partial match: class covers a related function
   - 0.0 = Mismatch: class describes a completely different function

2. CONTEXT ALIGNMENT (0.0-1.0)
   Does this CPC class fit the SYSTEM CONTEXT?
   - 1.0 = Perfect fit: class belongs to the correct industry/domain
   - 0.5 = Related: class is in an adjacent domain
   - 0.0 = Wrong domain: class belongs to a completely different industry

3. VISUAL BIAS CHECK
   If the class mentions image, video, pixel, frame, camera, visual, picture, or photograph:
   - Does the invention ACTUALLY process images/video? (true/false)
   - If false, this is a visual bias mismatch

4. METHOD VS APPARATUS CHECK (CRITICAL)
   - If claim type is METHOD: Does this CPC subgroup primarily describe methods/processes?
   - If APPARATUS: Does it primarily describe systems/devices?
   - If BOTH: Does it cover both, or at least not exclude either?

5. SPECIFICITY FIT
   - too_broad: Class is too generic to capture the novel contribution
   - appropriate: Class specificity matches the invention's detail level
   - too_narrow: Class is more specific than the invention's scope

6. DOMAIN APPROPRIATENESS CHECK (CRITICAL — DYNAMIC)
   Analyze the candidate's CPC domain against the invention's system context and core function.
   
   INSTRUCTION: Based ONLY on the System Context and Core Function provided above, determine:
   
   a) What is the PRIMARY technical domain of this invention?
      (e.g., "natural language processing / chatbot systems", "computer vision", "oil drilling")
   
   b) Does the CPC candidate's domain match this primary domain?
      - Match: The candidate belongs to the same technical area
      - Related: The candidate is in an adjacent or supporting domain
      - Mismatch: The candidate is in a completely unrelated domain
   
   c) Is the CPC candidate's domain contradictory to the invention?
      Example contradictions:
      - Invention is about software/chatbots, but candidate is about hardware input devices
      - Invention is about text processing, but candidate is about image processing
      - Invention is about data retrieval, but candidate is about fault tolerance
      - Invention is about mechanical seals, but candidate is about neural networks
   
   d) Rejection rule:
      - If the candidate is a CLEAR MISMATCH (completely wrong domain) → FAIL immediately
      - If the candidate is RELATED but not the primary focus → PASS with LOW confidence
      - If the candidate MATCHES the primary domain → PASS with confidence based on other checks

7. CONTRADICTION CHECK
   Does this class conflict with the core function or system context in any way?

=== DECISION RULES ===

PASS if:
- function_alignment >= 0.6 AND
- context_alignment >= 0.5 AND
- visual_bias == false AND
- method_apparatus_aligned == true AND
- specificity_fit != "too_broad" AND
- domain_mismatch == false (not a completely wrong domain)

FAIL if any critical check fails.

If the candidate belongs to a COMPLETELY WRONG domain for this invention (e.g., chatbot patent classified in fault tolerance), set decision to FAIL regardless of other scores.

If score_margin < 0.1, mark confidence as LOW regardless of other factors.

=== OUTPUT ===
{{
  "decision": "PASS" or "FAIL",
  "confidence": "high" or "medium" or "low",
  "scores": {{
    "function_alignment": 0.9,
    "context_alignment": 0.8,
    "visual_bias": false,
    "method_apparatus_aligned": true
  }},
  "specificity_fit": "appropriate",
  "contradictions": ["string" or []],
  "reasoning": "One paragraph explaining the validation decision",
  "rejection_reason": "If FAIL, explain specifically why"
}}

Output ONLY valid JSON, no markdown, no text outside the object.
"""


def reconciliation_prompt(
    validated_codes: list, rejected_codes: list, per_claim_classifications: list
) -> str:
    """
    Phase 6: Per-claim reconciliation.
    Removes rejected codes from per-claim classifications and replaces with validated alternatives.
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

    return f"""You are reconciling claim-level CPC classifications with validated CPC results.

=== VALIDATED CPC CODES (PASSED) ===
{validated_list}

=== REJECTED CPC CODES (FAILED) ===
{rejected_list}

=== CURRENT PER-CLAIM CLASSIFICATIONS ===
{per_claim_list}

=== TASK ===
For each claim:
1. Remove any CPC codes that were rejected globally (listed in REJECTED)
2. If removing a code leaves the claim with no CPC classes, replace with the best-matching validated code
3. Ensure alignment with:
   - Claim type (method vs apparatus)
   - Claim-specific function
4. Keep provisional codes only if they match a validated code

Output format:
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

Output ONLY valid JSON, no markdown, no text outside the object.
"""


def consistency_check_prompt(phase1_data: dict, selected_codes: list) -> str:
    """
    Phase 7: Final consistency check.
    Ensures selected CPC codes are coherent and logically fit together.
    """
    codes_list = "\n".join(
        f"- {s.get('symbol', '')}: {s.get('title', '')}" for s in selected_codes
    )

    return f"""You are performing a final coherence check on selected CPC classifications.

=== INVENTION PROFILE ===
Technical Object: {phase1_data.get("technical_object", "")}
Core Function: {phase1_data.get("core_function", "")}
System Context: {phase1_data.get("system_context", "")}
Classification Strategy: {phase1_data.get("classification_strategy", "")}

=== SELECTED CPC CODES ===
{codes_list}

=== TASK ===
1. Do these CPC codes logically fit together? (Are they from compatible domains?)
2. Is there a conflicting domain? (e.g., one code for image processing and another for text retrieval)
3. Is one code clearly dominant/primary?
4. Do the codes cover both method and apparatus aspects if needed?
5. Are any codes redundant or overlapping?

Output:
{{
  "coherent": true or false,
  "issues": ["string" or []],
  "recommended_primary": "G06F16/00",
  "recommended_secondary": ["G06N3/08"],
  "reasoning": "Explanation of coherence assessment"
}}

Output ONLY valid JSON, no markdown, no text outside the object.
"""
