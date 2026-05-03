"""
prompts.py - Corrected patent classification prompts.

Changes vs original:
  Issue 1  - Binary strategy replaced with a 3-way spectrum (system-first / hybrid / function-first)
  Issue 2  - Hardcoded domain rules replaced with weighted guidance + override escape hatch
  Issue 3  - Claim independence is resolved BEFORE the prompt; pre-labeled claims are injected
  Issue 4  - Importance scores anchored to a calibration rubric
  Issue 5  - cpc_hints usage made explicit with verification instruction
  Issue 6  - Multi-invention patents handled: one classification block per independent claim group
  Issue 7  - Negative signals merged into Phase 1 output (no separate LLM call needed)
  Issue 8  - Geometry exclusion scoped: excluded unless geometry IS the functional differentiator
  Issue 9  - Output schema documented inline; caller should validate with Pydantic (see note)
  Issue 10 - Self-consistency re-check added at end of Step 6 before committing to Step 7
"""

import re


def label_claims(raw_claims_text: str) -> str:
    """
    Parse raw claims text and return a version where every claim is prefixed
    with [INDEPENDENT] or [DEPENDENT: ref <n>].

    Handles patterns like:
      "2. The device of claim 1, wherein ..."
      "15. The method according to claim 14 ..."
      "18. The assembly of claims 1 or 2 ..."
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


IMPORTANCE_RUBRIC = """
Importance score calibration (use this scale strictly):
  10 - Term appears in an independent claim AND is the core inventive feature
   9 - Term appears in an independent claim and is a key structural/functional element
   8 - Term is in the description AND directly enables the solution
   7 - Term is in the description and important context for the solution
   5 - Term is supporting/background context
   3 - Term is generic domain vocabulary (keep only if no better term)
   1 - Term is too generic to contribute (should be excluded)
Never assign 10 to description-only terms. Never assign below 7 to independent-claim terms
unless they are purely generic (e.g. "comprising", "device").
"""


def phase1_prompt(cpc_hints: str, labeled_claims: str, description: str) -> str:
    """
    Phase 1 prompt: Examiner-grade CPC classification.

    Parameters
    ----------
    cpc_hints      : CPC hierarchy tree from EPOClassificationClient
    labeled_claims : Output of label_claims() - claims pre-labeled [INDEPENDENT]/[DEPENDENT]
    description    : Patent description text
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
[INDEPENDENT] or [DEPENDENT: ref claim N]. You MUST only extract terms from
[INDEPENDENT] claims in Step 5. Do NOT use [DEPENDENT] claims for term extraction.

{labeled_claims}

=== DESCRIPTION ===
{description}

{IMPORTANCE_RUBRIC}

---

STEP 1 - Technical Understanding

Extract:
- technical_object   : What is the invention? (1-2 sentences, concrete and specific)
- problem_solved     : What specific technical problem is addressed?
- solution_summary   : How does the invention solve the problem? Focus on the mechanism.

---

STEP 2 - System Context

Identify the broader technical system or industry in which the invention operates.

Rules:
- Must describe an industry/application domain, NOT a component
- Ask: "What industry would buy/use this invention?"
- Must be a system or application (e.g., "oil/gas wellhead assembly"), not a part

Output: system_context

---

STEP 3 - Core Technical Function

Identify the PRIMARY function performed by the invention (what it DOES, not what it looks like).

Geometry note: Exclude geometric descriptions UNLESS the geometry is the functional
differentiator (e.g., a blade profile that creates a specific aerodynamic effect). In that
case include the geometric term with explicit justification of its functional role.

Output: core_function

---

STEP 4 - Essential Technical Terms from DESCRIPTION

Extract 5-10 essential terms from the DESCRIPTION that are required to understand the solution.

Rules:
- Must be tied to solving the problem
- Include domain-specific terms (e.g., "wellhead", "tubing hanger", "annulus")
- Exclude generic words: device, system, apparatus, plurality, comprising
- Exclude purely geometric terms UNLESS geometry is the functional differentiator
- Use the importance rubric above to assign scores

Each term: {{ "term": "...", "importance": 8, "justification": "...", "source": "description" }}

---

STEP 5 - Essential Technical Terms from INDEPENDENT CLAIMS ONLY

Extract 3-8 terms from claims labeled [INDEPENDENT] ONLY.
Do NOT extract from [DEPENDENT] claims under any circumstances.

Rules:
- Focus on structural elements, materials, arrangements, functional relationships
- These terms carry higher weight in classification (see importance rubric)
- Use the importance rubric above - claims terms start at 9 unless generic

Each term: {{ "term": "...", "importance": 9, "justification": "From Claim N: ...", "source": "claims" }}

---

STEP 6 - Multi-Invention Check

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

STEP 7 - Classification Strategy

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

STEP 8 - CPC Class Selection

Select 2-4 CPC classes (4-character codes like F01P, F16K, B60L, E21B).

You MUST verify each selected class against the CPC REFERENCE provided at the top.
If a class is not in the reference, replace it with the closest ancestor that IS present.

Apply your classification_strategy:

  system-first  -> PRIMARY = domain class, SECONDARY = function class (if distinct)
  function-first -> PRIMARY = function class, SECONDARY = domain class (if relevant)
  hybrid        -> List both domain and function classes; order by how novel each contribution is

Strong domain guidance (NOT absolute overrides - use judgment for cross-domain inventions):
  - Wellhead/tubing hanger/BOP equipment           -> E21B strongly preferred as primary
  - Drilling equipment (drill bit, drill string)    -> E21B strongly preferred as primary
  - Explosive cutting tools in wells                -> E21B29 strongly preferred
  - Well completion/abandonment methods             -> E21B43 strongly preferred
  - Generic seals / valves / pumps                  -> function class (F16J / F16K / F04) as primary
  - Engine cooling systems                          -> F01P as primary
  - Battery/EV thermal management                   -> F01P or B60L depending on focus
  - Neural networks / ML models                     -> G06N as primary

Cross-domain exception: If the primary technical contribution is in a different domain
than the application (e.g., a wellhead that uses a novel neural network for control),
classify the NOVEL CONTRIBUTION as primary, not the application domain.
Justify this explicitly in cpc_reasoning.

For multi-invention patents (single_invention: false):
  Provide a separate cpc_classes array per invention group.

Output:
- cpc_classes: [list of 4-char codes]  (or list of lists if multi-invention)
- cpc_sections: [list of single letters]
- cpc_reasoning: explanation referencing strategy and CPC reference verification

---

STEP 9 - Negative Signals

Generate terms and domains that this patent is clearly NOT about.
These are used to penalize incorrect classifications in downstream scoring.

Rules:
- Be specific - avoid generic terms like "device" or "system"
- Focus on domains that could be confused with the correct classification
- At least 5 negative signals, at least 2 negative domains

Output:
- negative_signals: [list of strings]
- negative_domains: [list of strings]
- negative_reasoning: brief explanation

---

OUTPUT FORMAT (STRICT JSON - no markdown, no text outside the JSON object)

{{
  "technical_object": "string",
  "problem_solved": "string",
  "solution_summary": "string",
  "system_context": "string",
  "core_function": "string",
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
  "cpc_classes": ["E21B", "F16J"],
  "cpc_sections": ["E", "F"],
  "cpc_reasoning": "string",
  "description_terms": [
    {{
      "term": "string",
      "importance": 8,
      "justification": "string",
      "source": "description"
    }}
  ],
  "claims_terms": [
    {{
      "term": "string",
      "importance": 9,
      "justification": "string - include claim number",
      "source": "claims"
    }}
  ],
  "negative_signals": ["string"],
  "negative_domains": ["string"],
  "negative_reasoning": "string"
}}
"""


def rerank_prompt(phase1_data: dict, top5_codes: list) -> str:
    """
    Post-ranking prompt: re-rank top 5 CPC codes and select the best one.
    """
    cpc_lines = []
    for i, code in enumerate(top5_codes):
        symbol = code.get("symbol", "N/A")
        title = code.get("title", "N/A")
        score = code.get("score", 0)
        cpc_lines.append(f"{i + 1}. {symbol} - {title} (score: {score})")
    cpc_list = "\n".join(cpc_lines)

    strategy = phase1_data.get("classification_strategy", "unknown")
    consistency = phase1_data.get("consistency_check", "consistent")
    invention_groups = phase1_data.get("invention_groups", [])
    multi = len(invention_groups) > 1

    multi_note = ""
    if multi:
        groups_desc = "; ".join(
            f"Group {i + 1} (claims {g['claims']}): {g['focus']}"
            for i, g in enumerate(invention_groups)
        )
        multi_note = f"""
NOTE - Multi-invention patent detected.
Invention groups: {groups_desc}
Re-rank with the PRIMARY invention group in mind (Group 1 unless instructed otherwise).
"""

    return f"""You are a patent examiner with deep expertise in CPC classification.

Given the patent information and top 7 candidate CPC codes below, re-rank them by how
accurately they reflect the invention's primary technical contribution. Then select the
SINGLE BEST code.

Patent Information:
- Technical Object       : {phase1_data.get("technical_object", "")}
- Problem Solved         : {phase1_data.get("problem_solved", "")}
- Solution Summary       : {phase1_data.get("solution_summary", "")}
- System Context         : {phase1_data.get("system_context", "")}
- Core Function          : {phase1_data.get("core_function", "")}
- Classification Strategy: {strategy}
- Consistency Check      : {consistency}
{multi_note}

Top 5 Candidate CPC Codes:
{cpc_list}

Re-ranking rules:
- strategy = "system-first"   -> domain codes rank higher than function codes
- strategy = "function-first" -> function codes rank higher than domain codes
- strategy = "hybrid"         -> rank by specificity of match to the novel contribution
- If the correct best code is NOT in the top 5, set out_of_list: true and name it

Output ONLY valid JSON, no markdown, no text outside the object:

{{
  "re_ranked": [
    {{
      "rank": 1,
      "symbol": "string",
      "title": "string",
      "justification": "string"
    }}
  ],
  "best_code": {{
    "symbol": "string",
    "title": "string",
    "confidence": "high | medium | low",
    "reasoning": "string"
  }},
  "out_of_list": false
}}
"""
