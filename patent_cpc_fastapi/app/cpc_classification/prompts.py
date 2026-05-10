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


def phase1_prompt(labeled_claims: str, description: str) -> str:
    """
    Phase 1 prompt: Semantic extraction only - NO CPC class assignment.

    Responsibility: Understand the invention technically.
    Do NOT predict CPC classes - that is Phase 2's job via Knowledge Graph.
    """

    return f"""You are a patent technical analysis expert.

Your task is to analyze the patent description and claims and produce a structured JSON output
that captures the invention's technical essence. 

IMPORTANT: Do NOT assign CPC classes or predict patent classifications. 
Your job is semantic extraction only - understanding what the invention is and does.
CPC classification will be performed separately by a knowledge graph system.

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

STEP 2 — SYSTEM CONTEXT & INDUSTRY

Identify:
1. The broader technical system or industry in which the invention operates
2. What professionals or industries will USE this invention

Rules:
- Must describe an industry/application domain, NOT a component
- Ask: "What industry would buy/use this invention?"
- Ask: "What professionals (engineers, doctors, farmers, etc.) would use this?"
- Must be a system or application (e.g., "database search engines"), not a part
- Consider: healthcare, agriculture, manufacturing, telecommunications, finance, automotive, etc.

Output format:
- system_context: The technical system (e.g., "natural language processing systems")
- target_industry: The primary industry (e.g., "customer service automation", "medical diagnostics")
- target_professionals: Who uses it (e.g., "software developers", "physicians", "farm equipment operators")

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

STEP 4 — CLAIM TYPE ANALYSIS

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

CRITICAL: Extract COMPOUND terms that include the technical object, not isolated verbs:
- GOOD: "weight clipping" (clipping OF weights in neural networks)
- GOOD: "model quantization" (quantization OF models)
- BAD: "clipping" (ambiguous - could be image clipping)
- BAD: "quantization" (too generic without object)

Rules:
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

STEP 7 — INVENTION CHARACTERIZATION STRATEGY

Characterize the invention's nature to guide downstream classification:

  "system-first"   - The invention is a complete apparatus/machine for a SPECIFIC industry.
                     No other industry would use this exact invention.
                     Example domains: oil drilling, automotive braking, medical surgery

  "function-first" - The invention is a generic component usable across MULTIPLE industries.
                     Example functions: neural networks, sealing, valves, cooling

  "hybrid"         - The invention has a novel functional mechanism AND is tied to a specific
                     application.

Decision rule:
  1. Consider target_industry from Step 2: Is the invention tied to ONE specific industry?
     - If YES (e.g., oil drilling, medical surgery, automotive braking) -> system-first
     - If NO (could be used in many industries) -> function-first
  2. Consider target_professionals: Are the users specialists in one field?
     - Surgeons, miners, farmers -> system-first likely
     - Software engineers, data scientists -> function-first likely
  3. Ask: "Could this exact invention be deployed in two or more unrelated industries without
     modification?" If NO -> system-first. If YES -> function-first.
  4. If the functional innovation is as significant as the application specificity -> hybrid.

Self-consistency check (mandatory before proceeding):
  Re-read your technical_object from Step 1.
  Confirm that your chosen strategy is consistent with that description AND target_industry.
  If there is tension, revise system_context or core_function before continuing.

Output:
- classification_strategy: "system-first" | "function-first" | "hybrid"
- strategy_reasoning: explanation referencing technical_object, system_context, core_function, target_industry
- consistency_check: "consistent" | "revised - [what was revised and why]"

---

STEP 7b — PRIMARY TECHNICAL DOMAIN DETECTION (CRITICAL)

Analyze the invention and determine the PRIMARY technical domain based on:
1. Technical object (what is being improved?)
2. System context (where is it used?)
3. Core function (what does it do?)

Select ONE primary domain from this list. Be decisive - do NOT hedge:

- AI / Machine Learning / Neural Networks → G06N
- General Computing / Data Processing / Software → G06F
- Image / Video / Graphics Processing → G06T
- Computer Vision / Pattern Recognition → G06V
- Speech / Audio / Acoustic Processing → G10L
- Telecommunications / Networking → H04L
- Wireless Communication → H04W
- Semiconductors / Integrated Circuits → H01L
- Medical Technology / Diagnostics → A61B
- Pharmaceutical / Biotech → A61K
- Mechanical Engineering / Structures → F16
- Engines / Turbines / Pumps → F01-F04
- Control Systems / Automation → G05B
- Measurement / Sensors / Testing → G01
- Data Storage / Memory → G11
- Vehicles / Transport / Automotive → B60
- Chemistry / Materials / Chemical Processes → C01-C12
- Manufacturing / Machining → B23
- Oil / Gas / Mining / Drilling → E21

RULE: Domain must reflect the INDUSTRY / SYSTEM / SUBJECT MATTER, not just the algorithm or generic computation.

Examples:
- "Neural network weight quantization" → AI/ML (G06N) — the NN is the SUBJECT
- "Image clipping for graphics" → Image Processing (G06T) — the image is the SUBJECT
- "Speech recognition system" → Audio Processing (G10L) — speech is the SUBJECT
- "Sealing mechanism for engines" → Mechanical (F16) — the seal is the SUBJECT
- "Data transmission protocol" → Telecom (H04L) — communication is the SUBJECT

Output:
- primary_domain: {{ "name": "AI / Machine Learning", "cpc_class": "G06N", "confidence": 0.9 }}
- domain_reasoning: "The invention improves neural network models through weight manipulation."

---

STEP 7c — CONTEXTUAL TERM DISAMBIGUATION (CRITICAL)

For each extracted term, determine its meaning using OBJECT + DOMAIN + CONTEXT.

Many technical terms are ambiguous without context:
- "clipping": weight clipping (G06N) vs image clipping (G06T) vs audio clipping (G10L)
- "filtering": signal filtering (H04L) vs image filtering (G06T) vs data filtering (G06F)
- "encoding": video encoding (G06T) vs text encoding (G06F) vs channel encoding (H04L)
- "compression": model compression (G06N) vs image compression (G06T) vs data compression (G06F)
- "mapping": memory mapping (G06F) vs image mapping (G06T) vs neural mapping (G06N)

RULE: TERM meaning = f(object, system_context, domain)

For each ambiguous term in your extraction, resolve:
- term: "clipping"
- disambiguated_meaning: "weight clipping in neural networks"
- correct_domain: "G06N"
- incorrect_domains: ["G06T", "G10L"]
- justification: "The patent describes clipping neural network weights, not image regions"

Output:
- disambiguated_terms: [ {{ "term": "string", "meaning": "string", "domain": "G06N", "avoid": ["G06T"] }} ]

---

STEP 8 — DOMAIN SIGNALS (CRITICAL FOR DOWNSTREAM CLASSIFICATION)

Identify specific technical domains and technologies present in the invention.
This helps the downstream classification system map to the correct CPC families.

For each domain signal:
- name: The technology/domain name (e.g., "neural networks", "wireless communication")
- confidence: How certain (0.0-1.0) that this domain is central to the invention
- evidence: Which terms or claim elements support this
- cpc_family: The 3-char CPC family this domain maps to (e.g., "G06N", "H04L")

IMPORTANT:
1. The PRIMARY domain from Step 7b MUST be listed first with highest confidence
2. List ALL relevant domains, even if seemingly contradictory
3. Be explicit about tool vs purpose: if AI is used for image processing, list BOTH but mark image as primary
4. Include NEGATIVE mappings: domains that are NOT relevant but could be confused

Example domain signals:
- {{ "name": "neural network parameter optimization", "confidence": 0.95, "evidence": "weight clipping, quantization, model compression", "cpc_family": "G06N", "role": "primary" }}
- {{ "name": "image processing", "confidence": 0.1, "evidence": "none - possible confusion from 'clipping'", "cpc_family": "G06T", "role": "negative" }}
- {{ "name": "natural language processing", "confidence": 0.9, "evidence": "LLM, text generation", "cpc_family": "G06F", "role": "primary" }}
- {{ "name": "embedded systems", "confidence": 0.8, "evidence": "vehicle onboard processor", "cpc_family": "G06F", "role": "secondary" }}

Output:
- domain_signals: [ {{ "name": "string", "confidence": 0.9, "evidence": "string", "cpc_family": "G06N", "role": "primary|secondary|negative" }} ]

---

STEP 9 — NEGATIVE SIGNALS

Extract terms and domains that this patent is clearly NOT about.

Rules:
- Be specific - avoid generic terms like "device" or "system"
- Focus on domains that could be confused with the correct classification
- At least 5 negative signals, at least 2 negative domains
- Assign confidence (0.0-1.0) per signal, NOT absolute exclusions
- Map each negative domain to a CPC family to penalize

CRITICAL for all patents:
- If primary domain is G06N (AI): explicitly list "image processing" (G06T), "acoustics" (G10K) as negatives
- If primary domain is G06T (image): explicitly list "neural networks" (G06N) unless AI is actually used
- If primary domain is H04L (telecom): explicitly list "general computing" (G06F) unless relevant
- If primary domain is F16 (mechanical): explicitly list "software" (G06F) unless relevant

Examples:
- Negative signal: "image processing" (confidence: 0.9) → penalize G06T
- Negative signal: "computer graphics" (confidence: 0.8) → penalize G06T
- Negative signal: "audio processing" (confidence: 0.7) → penalize G10L

Output:
- negative_signals: [ {{ "term": "image processing", "confidence": 0.8, "penalize_family": "G06T" }} ]
- negative_domains: [ {{ "domain": "computer vision", "confidence": 0.9, "penalize_family": "G06V" }} ]
- negative_reasoning: brief explanation

---

OUTPUT FORMAT (STRICT JSON - no markdown, no text outside the JSON object)

{{
  "technical_object": "string",
  "problem_solved": "string",
  "solution_summary": "string",
  "system_context": "string",
  "target_industry": "string",
  "target_professionals": "string",
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
  "primary_domain": {{
    "name": "AI / Machine Learning",
    "cpc_class": "G06N",
    "confidence": 0.95,
    "reasoning": "The invention improves neural network models"
  }},
  "domain_signals": [
    {{
      "name": "neural network parameter optimization",
      "confidence": 0.95,
      "evidence": "weight clipping, quantization",
      "cpc_family": "G06N",
      "role": "primary"
    }},
    {{
      "name": "image processing",
      "confidence": 0.1,
      "evidence": "none - confusion possible",
      "cpc_family": "G06T",
      "role": "negative"
    }}
  ],
  "disambiguated_terms": [
    {{
      "term": "clipping",
      "meaning": "weight clipping in neural networks",
      "domain": "G06N",
      "avoid": ["G06T", "G10L"]
    }}
  ],
  "terms": [
    {{
      "term": "string",
      "importance": 8,
      "justification": "string",
      "source_section": "summary"
    }}
  ],
  "negative_signals": [
    {{"term": "image processing", "confidence": 0.8, "penalize_family": "G06T"}}
  ],
  "negative_domains": [
    {{"domain": "computer vision", "confidence": 0.9, "penalize_family": "G06V"}}
  ],
  "negative_reasoning": "string"
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

3. CROSS-DOMAIN LEAKAGE CHECK (CRITICAL — GENERALIZED)
   
   === DOMAIN DOMINANCE RULE ===
   The invention's PRIMARY DOMAIN (from Phase 1) is: {phase1_data.get("primary_domain", {}).get("name", "unknown")} ({phase1_data.get("primary_domain", {}).get("cpc_class", "unknown")})
   
   Check the candidate's CPC family against the primary domain:
   - If candidate is in the SAME domain as primary → STRONG PASS signal
   - If candidate is in a RELATED domain (supporting/aspect) → CONDITIONAL PASS
   - If candidate is in an UNRELATED domain → STRONG FAIL signal
   
   === CROSS-DOMAIN LEAKAGE PREVENTION ===
   Penalize these UNLESS explicitly supported by the invention:
   - G06T (image) if invention is NOT about image/video/graphics
   - G10L/K (audio/acoustics) if invention is NOT about sound/audio
   - G06V (vision) if invention is NOT about computer vision
   - G06N (AI) if invention is NOT about machine learning/neural networks
   - H04L/W (telecom) if invention is NOT about communication
   - F16 (mechanical) if invention is NOT about mechanical engineering
   - A61 (medical) if invention is NOT about healthcare
   
   Rationale: Prevents keyword-based misclassification across domains.

4. VISUAL/DOMAIN BIAS CHECK
   If the class mentions image, video, pixel, frame, camera, visual, picture, or photograph:
   - Does the invention ACTUALLY process images/video? (true/false)
   - If false, this is a domain bias mismatch
   
   Similarly check for other domain-specific terms:
   - Audio/speech terms → is invention about audio?
   - Neural network terms → is invention about AI?
   - Mechanical terms → is invention about mechanical systems?

5. METHOD VS APPARATUS CHECK (CRITICAL)
   - If claim type is METHOD: Does this CPC subgroup primarily describe methods/processes?
   - If APPARATUS: Does it primarily describe systems/devices?
   - If BOTH: Does it cover both, or at least not exclude either?

6. SPECIFICITY FIT
   - too_broad: Class is too generic to capture the novel contribution
   - appropriate: Class specificity matches the invention's detail level
   - too_narrow: Class is more specific than the invention's scope

7. DOMAIN APPROPRIATENESS CHECK (CRITICAL — DYNAMIC)
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
      - Invention is about telecom protocols, but candidate is about medical imaging
   
   d) Rejection rule:
      - If the candidate is a CLEAR MISMATCH (completely wrong domain) → FAIL immediately
      - If the candidate is RELATED but not the primary focus → PASS with LOW confidence
      - If the candidate MATCHES the primary domain → PASS with confidence based on other checks

8. INTRA-DOMAIN SPECIALIZATION CHECK
   If the candidate is in the correct domain, check if the SUBGROUP matches the specific function:
   
   Examples:
   - G06N (AI): parameter optimization → G06N 3/045; model compression → G06N 3/063
   - G06T (image): filtering → G06T 5/00; rendering → G06T 15/00
   - H04L (telecom): error correction → H04L 1/00; protocols → H04L 29/00
   
   Does the candidate's subgroup reflect the INVENTIVE CONTRIBUTION, not just the field?

9. CONTRADICTION CHECK
   Does this class conflict with the core function or system context in any way?

=== DECISION RULES ===

PASS if:
- function_alignment >= 0.6 AND
- context_alignment >= 0.5 AND
- cross_domain_leakage == false (not in unrelated domain) AND
- visual_bias == false AND
- method_apparatus_aligned == true AND
- specificity_fit != "too_broad" AND
- domain_mismatch == false (not a completely wrong domain)

FAIL if any critical check fails.

If the candidate belongs to a COMPLETELY WRONG domain for this invention (e.g., chatbot patent classified in fault tolerance, or telecom patent in medical devices), set decision to FAIL regardless of other scores.

If score_margin < 0.1, mark confidence as LOW regardless of other factors.

=== MULTI-LABEL GUIDANCE ===
If the invention has multiple distinct technical contributions:
- PRIMARY CPC should cover the MAIN inventive concept
- SECONDARY CPC (if any) should cover a SUPPORTING aspect
- Do NOT assign multiple CPCs for the SAME concept

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
Primary Domain: {phase1_data.get("primary_domain", {}).get("name", "unknown")} ({phase1_data.get("primary_domain", {}).get("cpc_class", "unknown")})

=== SELECTED CPC CODES ===
{codes_list}

=== TASK ===
1. Do these CPC codes logically fit together? (Are they from compatible domains?)
2. Is there a conflicting domain? (e.g., one code for image processing and another for text retrieval)
3. Is one code clearly dominant/primary?
4. Do the codes cover both method and apparatus aspects if needed?
5. Are any codes redundant or overlapping?
6. DOMAIN DOMINANCE CHECK (CRITICAL):
   - Does the PRIMARY selected code match the invention's primary domain?
   - If the invention is about AI (G06N), is G06N the primary code?
   - If the invention is about telecom (H04L), is H04L the primary code?
   - If the invention is about mechanical (F16), is F16 the primary code?
   - Cross-domain leakage: Is there a code from an UNRELATED domain? (e.g., G06T for non-image, G10L for non-audio)
7. INTRA-DOMAIN SPECIALIZATION CHECK:
   - Does the subgroup reflect the SPECIFIC function? (e.g., G06N 3/045 for parameter optimization, not just G06N 3/00)
   - Is the code too broad or too narrow for the invention?
8. MULTI-LABEL CHECK:
   - If multiple codes: Do they represent DISTINCT contributions?
   - PRIMARY: Main inventive concept
   - SECONDARY: Supporting aspect (e.g., hardware acceleration for AI)
   - Do NOT assign multiple codes for the SAME concept

=== FINAL GUARD RULES ===
- If primary_domain is G06N but selected primary is G06T → INCONSISTENT (cross-domain leakage)
- If primary_domain is H04L but selected primary is G06F → INCONSISTENT (unless computing is the focus)
- If primary_domain is F16 but selected primary is G06N → INCONSISTENT (unless AI is the focus)
- If selected codes contain both G06T and G06N → check if invention is actually multi-domain (AI for image processing) or if it's leakage

Output:
{{
  "coherent": true or false,
  "issues": ["string" or []],
  "recommended_primary": "G06F16/00",
  "recommended_secondary": ["G06N3/08"],
  "domain_consistent": true or false,
  "cross_domain_leakage": true or false,
  "leakage_details": "Description of any cross-domain issues",
  "specialization_fit": "appropriate" or "too_broad" or "too_narrow",
  "reasoning": "Explanation of coherence assessment"
}}

Output ONLY valid JSON, no markdown, no text outside the object.
"""
