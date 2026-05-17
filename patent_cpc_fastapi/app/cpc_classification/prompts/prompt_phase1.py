"""
prompt_phase1.py — Multi-pass Phase 1 extraction prompts.

Passes:
  A — Function-first extraction (what does it DO?)
  B — Structure-first extraction (what IS it?)
  C — Problem-Solution extraction (EPO framework)
  D — Drawing description extraction (component vocabulary)

Reconciliation merges all passes into the final Phase 1 output.
Completeness scoring validates the result before Phase 2A.

IMPORTANT ARCHITECTURE RULE:
  Passes A/B/C read ONLY the description section (NOT claims).
  Claims are analyzed for the FIRST time in Phase 1.2 (forensic audit).
  The reconciliation pass receives claims for final VERIFICATION only.
"""

from .shared import UNIFIED_IMPORTANCE_RUBRIC


# ---------------------------------------------------------------------------
# PASS A — Function-first extraction (DESCRIPTION ONLY — no claims)
# ---------------------------------------------------------------------------


def phase1_pass_a_prompt(description: str) -> str:
    return f"""You are a patent technical analyst performing a focused FUNCTION extraction.

Your task is to answer one question about this invention:
WHAT DOES IT DO? Focus exclusively on actions, operations, processes, and verbs.

Do NOT assign CPC classes. Do NOT describe physical structure unless structure IS the function.

IMPORTANT: You are reading ONLY the description section. Claims will be analyzed separately.

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

3. FUNCTIONAL TERMS
   Technical terms that describe operations (not components).
   Each term: {{ "term": "...", "importance": 8, "source_section": "description", "is_verb_phrase": true }}

4. CORE TECHNICAL PROBLEM BEING SOLVED
   What operation was previously impossible, slow, inaccurate, or inefficient?
   Frame as: "The invention enables [action] which was previously [limitation]"

OUTPUT FORMAT (strict JSON):
{{
  "primary_action": "string",
  "secondary_actions": ["string"],
  "functional_terms": [
    {{ "term": "string", "importance": 8, "source_section": "description", "is_verb_phrase": true }}
  ],
  "core_problem": "string",
  "pass": "A"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PASS B — Structure-first extraction (DESCRIPTION ONLY — no claims)
# ---------------------------------------------------------------------------


def phase1_pass_b_prompt(description: str) -> str:
    return f"""You are a patent technical analyst performing a focused STRUCTURE extraction.

Your task is to answer one question about this invention:
WHAT IS IT? Focus exclusively on components, architecture, modules, and physical/logical structure.

Do NOT assign CPC classes. Do NOT describe what the invention does — only what it IS made of.

IMPORTANT: You are reading ONLY the description section. Claims will be analyzed separately.

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
   Each: {{ "name": "...", "role": "what this component contributes" }}

3. SYSTEM ARCHITECTURE
   How components are connected or arranged.
   Describe the data/signal flow between components.

4. STRUCTURAL TERMS
   Technical nouns representing components, modules, data structures.
   Each: {{ "term": "...", "importance": 8, "source_section": "description", "is_noun": true }}

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
    {{ "term": "string", "importance": 8, "source_section": "description", "is_noun": true }}
  ],
  "claim_type_analysis": [
    {{ "claim_number": 1, "type": "METHOD|APPARATUS|BOTH", "structural_elements": ["string"] }}
  ],
  "pass": "B"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PASS C — Problem-Solution extraction (DESCRIPTION ONLY — no claims)
# ---------------------------------------------------------------------------


def phase1_pass_c_prompt(description: str) -> str:
    return f"""You are a patent technical analyst performing an EPO-style PROBLEM-SOLUTION extraction.

Your task is to identify the inventive contribution using the EPO problem-solution framework:
1. What existed before (closest prior art)?
2. What was the unsolved technical problem?
3. What is the solution (the invention)?
4. What technical effect or improvement results?

Do NOT assign CPC classes. Focus entirely on understanding the inventive step.

IMPORTANT: You are reading ONLY the description section. Claims will be analyzed separately.

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
   Essential: Features described as core to the invention.
   Optional: Features described as preferred embodiments.

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
# PASS D — Drawing description extraction (unchanged — already correct)
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
# Claims provided here for VERIFICATION only (not extraction)
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
  Pass A — Function-first (what it DOES) — from DESCRIPTION only
  Pass B — Structure-first (what it IS) — from DESCRIPTION only
  Pass C — Problem-Solution (EPO framework) — from DESCRIPTION only
  Pass D — Drawing descriptions (component vocabulary, data flow, figure inventory)
           [May be absent if no drawings section was provided]

Your job:
1. Find CONVERGENCE — terms and domain signals appearing in ≥ 2 passes are HIGH confidence
2. Find DIVERGENCE — terms in only 1 pass need disambiguation or lower confidence
3. Merge into a single authoritative technical profile
4. Assign final confidence per domain signal based on how many passes support it

IMPORTANT PRINCIPLES:
- Passes A/B/C extracted from DESCRIPTION only — claims were NOT available to them
- Claims below are provided for FINAL VERIFICATION — use them to validate, not replace
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

=== CLAIMS (for final verification only) ===
{labeled_claims}

---

RECONCILIATION TASKS:

TASK 1 — TERM RECONCILIATION
For each unique technical term across all passes:
- Count how many passes mention it (convergence_count)
- Assign final importance based on convergence AND section weight
- Flag ambiguous terms for disambiguation
- Mark terms only in Pass D prior-art figures as low trust

TASK 2 — DOMAIN SIGNAL RECONCILIATION (STRUCTURED FORMAT REQUIRED)
Merge domain signals from all passes:
- Pass A contributes: functional domain signals (what the action belongs to)
- Pass B contributes: structural domain signals (what the components belong to)
- Pass C contributes: problem domain signals (what field the problem is in)
- Pass D contributes: component domain signals (what the named components belong to)
A domain signal confirmed by ≥ 3 passes → confidence ≥ 0.9
A domain signal from only 1 pass → confidence ≤ 0.5 (flag for review)

You MUST produce at least 3 domain_signals with distinct roles:
  - primary:   the SUBJECT MATTER domain (what the invention IS, not what tool it uses)
  - secondary: the METHODOLOGY domain (if ML/NN is a tool, it's secondary, not primary)
  - negative:  a domain that looks similar but is WRONG — helps exclude false positives

DOMAIN SIGNAL FORMAT RULES (CRITICAL):
- EVERY domain signal MUST include a "cpc_family" field with a 4-character CPC code
- NEVER emit a domain signal without a cpc_family code
- If two signals map to the same family, merge them and take the higher confidence
- Maximum 4 signals total
- Methodology domains (G06N, G06F) MUST be included as secondary signals when the
  invention uses neural networks, transformers, or self-supervised learning as its
  core method — even if the subject matter domain (G10L, G06V) is dominant.
  Example: A speech + neural network patent MUST have both G10L (primary) and
  G06N (secondary, confidence ≥ 0.70).

DISAMBIGUATION RULE: The primary domain reflects the INDUSTRY / SYSTEM / SUBJECT.
If invention APPLIES ML to speech → primary is G10L (speech), secondary is G06N (ML).
If invention IS a new ML architecture → primary is G06N (ML), no subject-matter override.

ANTI-HALLUCINATION RULE (CRITICAL):
If a domain signal has ZERO evidence in the patent text — no terms, no components, no
problem description related to it, and zero passes supporting it — it MUST be listed as
role: "negative", NEVER as role: "primary" or "secondary" regardless of confidence.
A domain with no supporting evidence at 0.90 confidence is a hallucination, not a signal.
Example: If no image/video terms appear in the patent, G06T/G06V must be negative signals,
even if the LLM thinks the AI methodology "could be applied to images."

CLASSIFICATION STRATEGY FORMAT (STRUCTURED OBJECT):
Return classification_strategy as a JSON object, NOT a string:
  - strategy: "hybrid" | "single_domain" | "cross_domain"
  - primary_family: 4-char CPC code of the dominant domain
  - secondary_family: 4-char CPC code or null
  - anchor_split: [primary_weight, secondary_weight] — must sum to 1.0
    * single_domain: [1.0, 0.0]
    * hybrid: recommended [0.60, 0.40] unless one family clearly dominates
    * cross_domain: [0.50, 0.50]

TASK 2 — EVIDENCE TABLE EXTRACTION (STRUCTURED ENRICHMENT)
Return evidence_table as a JSON array. Each entry MUST include:
  - term: a technical noun phrase OR method step (verb + object)
  - weight: integer 1-10 (see weighting rules below)
  - justification: concise explanation of why this term is a CPC anchor
  - source: exactly one of "claims", "abstract", "description", "drawing_description"
  - source_quote: VERBATIM short phrase (max 15 words) from the patent justifying the term

SOURCE COVERAGE RULES:
- You MUST extract terms from: Independent claims, Dependent claims, Abstract, and
  key passages of the Detailed Description.
- Drawing descriptions should only be used if they name a functional component
  directly involved in the core method (e.g. "transformer encoder 104").

METHOD STEP PATTERN:
- For functional terms, use Verb + Object pairs rather than generic nouns.
- Examples: "clustering speech frame features" (NOT just "clustering"),
  "computing target features of phonemes" (NOT just "target features"),
  "iteratively training the network model" (NOT just "network model").

TERM COUNT & PRIORITY:
- Produce a MINIMUM of 10 terms and a MAXIMUM of 15.
- Priority 1: Terms from Independent Claims (weight 9-10)
- Priority 2: Terms from Dependent Claims (weight 7-8)
- Priority 3: Terms from Description/Abstract (weight 5-6)
- Priority 4: Drawing Descriptions (weight 3-5, only if functionally relevant)

EXPLICIT WEIGHTING RULES:
- Weight 9-10: Term is recited verbatim in an independent claim AND is part of the core inventive step.
- Weight 7-8:  Term appears in a dependent claim OR is a named component in the system architecture.
- Weight 5-6:  Term appears in the description or abstract and supports the core function.
- Weight 3-4:  Term is from drawing description or is contextual/background.

PROHIBIT STRUCTURAL NOISE:
- NEVER include generic infrastructure terms as standalone rows: "processor",
  "memory", "module", "system", "device", "network", "component", "apparatus".
- ONLY include these if they have a domain-specific modifier (e.g., "bidirectional
  LSTM encoder" is acceptable; "encoder" alone is not).

QUOTE REQUIREMENT:
- Every row MUST have a source_quote containing a verbatim snippet from the patent.
- If no quote is provided, the term is considered low-confidence and weight capped at 5.

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
d) Ask: Is this primary_domain the SUBJECT MATTER, or is it just the methodology?
   If it only reflects the methodology (e.g. G06N because ML is used), reconsider:
   what is the SYSTEM being built? What INDUSTRY/APPLICATION does it serve?
e) If YES to both → consistency_check = "consistent"
f) If NO → revise the conflicting field AND output the REVISED value in the JSON below
   (do not just note the revision — actually output the corrected field value)

TASK 5 — MULTI-INVENTION CHECK
If Pass B's claim_type_analysis shows independent claims with DIFFERENT technical objects:
- Flag as multi-invention
- Assign separate domain signals per invention group
- Note which claims belong to which invention

TASK 6 — DOMAIN HALLUCINATION PREVENTION (MANDATORY BEFORE OUTPUT)
Before outputting ANY domain signal with role "primary" or "secondary", you MUST
find direct evidence for it in the patent text. For each positive domain signal,
ask: Can I quote a specific term from the patent that belongs to this domain?

Domain-specific evidence requirements (if the domain is NOT supported by at least
one of these evidence words, it MUST be role "negative", never "primary"/"secondary"):
  G06T / G06V (Image/Video): image, pixel, camera, visual, picture, photograph,
     video, frame (in the visual sense, not audio frame)
  G10L (Speech/Audio): speech, audio, acoustic, phoneme, voice, utterance, speaker
   G06N (AI/ML): neural network, training, inference, model, weights, layers,
      gradient, backpropagation, LSTM, encoder, decoder, transformer, attention,
      embedding — G06N is the METHODOLOGY. If ANY of these terms exist in the
      patent, G06N MUST be kept as a SECONDARY domain at confidence ≥ 0.70.
      G06N is NEVER zeroed — it is the legitimate methodology domain for any
      ML-enhanced invention. Only G06T/G06V may be forced to 0.00 by the avoid list.
  H04L (Networking): network, packet, transmission, protocol, routing, node
  G06F (Computing): processor, memory, database, file, program, interface
  H04W (Wireless): wireless, radio, antenna, base station, signal, channel
  A61B (Medical): patient, diagnosis, medical, surgical, anatomical, clinical

If ZERO evidence words exist for a domain → it MUST be role "negative" at
confidence ≥ 0.90, never "primary" or "secondary". A domain with no evidence
at 0.90 confidence is a hallucination, not a classification.

TASK 7 — TERM EXTRACTION RULES
source_section MUST be a SINGLE value — the ONE most authoritative section.
Valid values: "claims", "summary", "detailed_description", "drawing_description",
"abstract", "background".
NEVER use combined strings like "claims|summary|detailed_description".
Hierarchy: claims > summary > detailed_description > drawing_description > abstract > background.
If a term appears in both claims and description → source_section = "claims".

Term importance MUST vary — do not assign identical importance to all terms.
  10 = core inventive feature in an independent claim
   9 = key structural/functional element in an independent claim
   8 = detailed description term that directly enables the solution
   7 = supporting/enabling term from description
   5 = generic domain vocabulary

TASK 8 — TECHNICAL GLOSSARY ASSEMBLY (MANDATORY — 10-15 terms minimum)

Build a comprehensive technical glossary by merging terms from FOUR mandatory sources.
The glossary must read like a technical dictionary, not a claim summary.
DISABLE aggressive filtering — do NOT remove terms because they seem similar, share
prefixes, or appear generic. Prioritise coverage over minimalism.

SOURCE 1 — STRUCTURAL COMPONENTS (modules from Technical Object + Pass B/D):
  Pull ALL named components from Pass B's structural_terms and Pass D's
  component_glossary. Include every entry with importance ≥ 8 from Pass B.
  These components are HARD ANCHORS — never filter them out, even if they
  appear general or similar to other terms. Mark category: "Structural".
  Minimum count: ≥ 4 structural terms.

  CRITICAL — MODULE LISTING RULE:
  Every unique module mentioned in the "Technical Object" (e.g., acquisition
  module, segmentation module, target feature generator module, encoder module,
  clustering module, training module) MUST appear as a DISTINCT glossary entry.
  These are the CPC search anchors that connect to specific subclasses.
  Even if modules seem redundant or overlapping, list them individually —
  the search engine needs each one as a separate query term.
  Example: "speech acquisition module" AND "speech feature extraction module"
  are BOTH listed, not merged into one.

SOURCE 2 — FUNCTIONAL OPERATIONS (from Core Technical Function):
  Extract high-impact actions from Pass A's functional_terms, process_steps,
  and secondary_actions. Focus on verbs that describe WHAT the invention DOES:
  extracting, clustering, aligning, segmenting, encoding, pooling, pre-training.
  Mark category: "Functional". Minimum count: ≥ 3 functional terms.

SOURCE 3 — DATA ENTITIES (intermediate results, features, labels, signals):
  Identify the data objects flowing through the system from Pass A+C:
  feature vectors, training labels, pseudo-labels, acoustic features,
  clustering centers, target features, teacher signals, training datasets.
  Mark category: "Data". Minimum count: ≥ 2 data terms.

SOURCE 4 — CLAIM-LEVEL TERMS (legal anchors from Pass C + independent claims):
  Pull inventive features from Pass C's essential_features and the independent
  claims: phoneme-derived initial centers, clustering labels, self-supervised
  pre-training. Mark category: "Legal".

AUTO BACK-FILL (if glossary < 10 terms after merging):
  Pull additional terms from Pass B structural_terms with importance = 10
  or highest convergence_count until reaching 10 terms. Mark as "Structural".

MERGE RULES (coverage-first, not minimalist):
- Merge all four sources into a single glossary of 10-15 terms.
- Deduplicate by term name (case-insensitive, keeping the most specific variant).
- DO NOT remove terms for similarity, shared word prefixes, or perceived generality.
  If in doubt, KEEP the term — the search engine benefits from alternative vocabulary.
- Each term gets exactly ONE category based on its PRIMARY source.
- Structural terms from Pass B/D have EQUAL priority to all other sources —
  they are never second-class citizens.

ANTI-MERGE RULE — NO SEMANTIC SUMMARIZATION:
  Do NOT semantically group or summarise similar terms into umbrella entries.
  Do NOT merge "speech feature extraction module", "speech frame segmentation
  module", and "acquisition module" into one "speech processing module" entry.
  Each module name from the Technical Object is a DISTINCT search anchor with
  its own CPC subclass mapping. List them individually with their raw technical
  names — precision over concision.

DOMAIN DOMINANCE ENFORCEMENT (prevents classification drift):
- Tag every term with its CPC domain: primary domain (e.g., G10L for speech)
  or secondary domain (e.g., G06N for ML methodology).
- At least 70% of all glossary terms MUST map to the primary domain.
- Secondary domain terms (methodology) are allowed ONLY as supporting context —
  never as the dominant vocabulary. Structural/functional terms that describe
  the invention's subject matter must be tagged primary domain.

G06N PROTECTION (methodology domain — never zeroed):
  If ANY of these terms appear in the Technical Object or components —
  neural network, training, LSTM, encoder, decoder, transformer, attention,
  embedding, weights, gradient, backpropagation, self-supervised —
  G06N MUST be kept as a SECONDARY domain at confidence ≥ 0.70.
  G06N is the legitimate methodology domain for any ML-enhanced invention.
  NEVER put G06N in any "avoid" list. NEVER zero out G06N.
  Only G06T and G06V are forced to 0.00 by the avoid list.
- ZERO terms must leak into irrelevant domains (G06T/G06V if no image/video
  processing exists in the patent).

NORMALIZE each entry into technical dictionary format:
  "Term → precise domain-specific definition with CPC hint [avoid: ...]"

FINAL VALIDATION (check before output):
  ✓ ≥ 10 terms in glossary
  ✓ ≥ 4 Structural terms (named components/modules)
  ✓ ≥ 3 Functional terms (operations/actions)
  ✓ ≥ 2 Data terms (signals/features/labels)
  ✓ Strong structural coverage — components are well-represented
  ✓ Glossary aligns with core_function from Pass A
  ✓ ≥ 70% of terms tagged primary domain (e.g., G10L)
  ✓ Zero terms tagged G06T, G06V, or other irrelevant domains
  ✓ Output reads like a technical glossary, not a claim summary

TASK 9 — GENERALIZED FUNCTION & CPC TERMS (REQUIRED NEW OUTPUT LAYERS)

Generate two additional output layers that broaden matchability for CPC search:

LAYER A — core_function_generalized:
Produce exactly 3 broad rewordings of the core_function. Rules:
- Remove overly specific terms (e.g., "phoneme-derived initial centers" → drop the specific mechanism)
- Replace with widely used technical concepts that preserve the invention's technical meaning
- Each entry must be a full verb phrase (not a noun phrase)
- Each entry must be DIFFERENT in its wording emphasis (e.g., one focuses on the ML aspect, another on the speech/acoustic aspect, a third on the training aspect)
- Example: ["train a speech model using clustered acoustic features", "self-supervised learning using clustering of feature representations", "speech representation learning using unsupervised clustering"]

LAYER B — cpc_terms:
Produce 6-10 standard CPC-style technical terms. Rules:
- Use standard patent classification language, not paper-style ML wording
- Prefer broad, established technical terms that map to CPC subclasses (G10L, G06N, G06F, etc.)
- Include terms from ALL domains the invention touches (subject matter + methodology)
- Example: ["speech recognition", "speech processing", "acoustic feature modeling", "feature vector clustering", "pattern classification", "machine learning training", "unsupervised learning", "data clustering"]

CRITICAL RULES:
- Do NOT remove or rewrite core_function from TASK 4 — it must remain precise
- Do NOT merge these layers into core_function
- Each layer is independent
- core_function_generalized entries must be FULL verb phrases (3 entries)
- cpc_terms must be NOUN phrases (6-10 entries)

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
  "core_function_generalized": [
    "string — broader abstraction of core function, removing over-specific terms",
    "string — alternative wording emphasizing different technical aspect",
    "string — third generalization maximizing CPC matchability"
  ],
  "cpc_terms": [
    "string — standard patent classification terminology",
    "string — broad established technical term",
    "string — additional CPC-relevant vocabulary",
    "string — further CPC-relevant vocabulary"
  ],
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

  "classification_strategy": {{
    "strategy": "hybrid|single_domain|cross_domain",
    "primary_family": "G10L",
    "secondary_family": "G06N or null",
    "anchor_split": [0.60, 0.40]
  }},
  "strategy_reasoning": "string",
  "consistency_check": "consistent|revised",
  "consistency_revision_details": "string — if revised, explain what changed and why",
  "technical_object_revised": "string — ONLY if consistency_check is revised",
  "core_function_revised": "string — ONLY if consistency_check is revised",

  "primary_domain": {{
    "name": "G10L — Speech processing (subject matter, not methodology)",
    "cpc_class": "G10L",
    "confidence": 0.9,
    "supporting_passes": ["A", "B", "C", "D"],
    "reasoning": "Invention IS a speech pre-training system; ML is the TOOL, speech is the SUBJECT"
  }},

  "class_hypotheses": [
    {{
      "class": "G10L",
      "confidence": 0.9,
      "source": "multi-pass convergence — speech processing is the subject",
      "passes_supporting": ["A", "B", "C"]
    }}
  ],

  "domain_signals": [
    {{
      "label": "Speech/Audio processing",
      "cpc_family": "G10L",
      "confidence": 0.9,
      "role": "primary",
      "evidence": "speech frame clustering, phoneme-derived initial centers, acoustic features",
      "convergence_count": 3,
      "passes_supporting": ["A", "B", "C"]
    }},
    {{
      "label": "Neural network methods / self-supervised learning",
      "cpc_family": "G06N",
      "confidence": 0.8,
      "role": "secondary",
      "evidence": "ML is the TOOL used for clustering and pre-training",
      "convergence_count": 2,
      "passes_supporting": ["A", "C"]
    }},
    {{
      "label": "Image/video processing",
      "cpc_family": "G06V",
      "confidence": 0.9,
      "role": "negative",
      "evidence": "No image or video features described; invention operates on speech frames",
      "convergence_count": 0,
      "passes_supporting": []
    }}
  ],

  "evidence_table": [
    {{
      "term": "phoneme-derived initial centers",
      "weight": 10,
      "justification": "Core inventive feature from independent claim 1",
      "source": "claims",
      "source_quote": "seeding centroids with phoneme-derived initial centers"
    }},
    {{
      "term": "clustering speech frame features",
      "weight": 8,
      "justification": "Key processing step for feature representation",
      "source": "description",
      "source_quote": "iteratively clustering the speech frame features to generate pseudo-labels"
    }}
  ],

  "disambiguated_terms": [
    {{
      "term": "Speech feature extractor",
      "category": "Structural",
      "meaning": "Acoustic front-end converting raw audio to MFCC/spectral feature vectors",
      "domain": "G10L",
      "avoid": ["G06T", "G06V"],
      "source_passes": ["B", "D"]
    }},
    {{
      "term": "Phoneme aligner",
      "category": "Structural",
      "meaning": "Forced alignment module time-stamping audio with phonetic unit boundaries",
      "domain": "G10L",
      "avoid": ["G06T"],
      "source_passes": ["B"]
    }},
    {{
      "term": "Bidirectional LSTM encoder",
      "category": "Structural",
      "meaning": "Neural network sequentially encoding speech frames in both temporal directions",
      "domain": "G06N",
      "avoid": ["G06F", "G06T"],
      "source_passes": ["B", "D"]
    }},
    {{
      "term": "Target feature generator",
      "category": "Structural",
      "meaning": "Module producing teacher-signal feature vectors for student-teacher ML training",
      "domain": "G06N",
      "avoid": ["G06T", "G06V"],
      "source_passes": ["B", "D"]
    }},
    {{
      "term": "Speech frame segmentation",
      "category": "Functional",
      "meaning": "Dividing continuous audio stream into discrete overlapping frame windows for analysis",
      "domain": "G10L",
      "avoid": ["G06T"],
      "source_passes": ["A", "B"]
    }},
    {{
      "term": "Phoneme-derived initial clustering",
      "category": "Functional",
      "meaning": "Seeding K-means centroids from acoustic-phonetic unit statistics instead of random init",
      "domain": "G10L",
      "avoid": ["G06K"],
      "source_passes": ["A", "C"]
    }},
    {{
      "term": "Attention-based pooling",
      "category": "Functional",
      "meaning": "Weighted aggregation of frame-level encoder outputs into fixed-dimensional utterance embedding",
      "domain": "G06N",
      "avoid": ["G06T"],
      "source_passes": ["A"]
    }},
    {{
      "term": "Acoustic feature vectors",
      "category": "Data",
      "meaning": "MFCC/spectral feature sequences extracted from speech frames — input to the encoder",
      "domain": "G10L",
      "avoid": ["G06T"],
      "source_passes": ["A", "B"]
    }},
    {{
      "term": "Clustering pseudo-labels",
      "category": "Data",
      "meaning": "Iteratively generated cluster-assignment labels used for self-supervised model training",
      "domain": "G10L",
      "avoid": ["G06F", "G06K"],
      "source_passes": ["A", "C"]
    }},
    {{
      "term": "Phoneme-derived initial centers",
      "category": "Legal",
      "meaning": "Seed values for clustering initialised from phonetic unit statistics — core inventive feature",
      "domain": "G10L",
      "avoid": [],
      "source_passes": ["C"]
    }},
    {{
      "term": "Self-supervised pre-training",
      "category": "Legal",
      "meaning": "Training speech models on unlabelled data using clustering-derived pseudo-labels",
      "domain": "G10L",
      "avoid": ["G06T"],
      "source_passes": ["C"]
    }}
  ],

  "terms": [
    {{
      "term": "phoneme-derived initial centers",
      "importance": 10,
      "justification": "Core inventive feature from independent claim 1",
      "source_section": "claims",
      "citation": "verbatim sentence from claim 1",
      "convergence_count": 3,
      "figure": null,
      "trust_level": null
    }},
    {{
      "term": "speech frame clustering",
      "importance": 7,
      "justification": "Key processing step described in detailed description",
      "source_section": "detailed_description",
      "convergence_count": 2,
      "figure": null,
      "trust_level": null
    }},
    {{
      "term": "bidirectional LSTM encoder",
      "importance": 8,
      "justification": "Named component from FIG. 2 (invention figure)",
      "source_section": "drawing_description",
      "convergence_count": 1,
      "figure": "FIG. 2",
      "trust_level": "high"
    }}
  ],

  "negative_signals": [
    {{ "term": "string", "confidence": 0.8, "penalize_family": "G06T" }}
  ],
  "negative_domains": [
    {{ "domain": "string", "confidence": 0.9, "penalize_family": "G06V" }}
  ],
  "negative_reasoning": "string"
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
    # Prompt architecture produces exactly 2 positive signals (primary + secondary) +
    # 1 negative — threshold of 2 is the correct full-credit bar.
    if len(signals) >= 2:
        score += 25
    elif len(signals) >= 1:
        score += 10
        issues.append(f"Only {len(signals)} domain signal(s) with confidence >= 0.5")
    else:
        issues.append("No confident domain signals found")

    pd_conf = extraction.get("primary_domain", {}).get("confidence", 0)
    if pd_conf == 0:
        # primary_domain object may be added by post-processing; fall back to the
        # highest-confidence positive domain signal as a proxy.
        pos_signals = [
            s for s in extraction.get("domain_signals", [])
            if s.get("role") != "negative" and s.get("confidence", 0) > 0
        ]
        if pos_signals:
            pd_conf = max(s["confidence"] for s in pos_signals)
    if pd_conf >= 0.7:
        score += 20
    elif pd_conf >= 0.5:
        score += 10
        issues.append(f"Primary domain confidence only {pd_conf:.2f}")
    else:
        issues.append("Primary domain confidence too low or missing")

    # Check both legacy "terms" (importance) and structured "evidence_table" (weight).
    # evidence_table is the authoritative 10-15 term extraction; terms may have fewer entries.
    high_terms = [t for t in extraction.get("terms", []) if t.get("importance", 0) >= 7]
    high_evidence = [
        e for e in extraction.get("evidence_table", []) if e.get("weight", 0) >= 7
    ]
    effective_high_count = max(len(high_terms), len(high_evidence))
    if effective_high_count >= 5:
        score += 15
    elif effective_high_count >= 2:
        score += 8
        issues.append(f"Only {effective_high_count} high-importance terms")
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
# MAIN ENTRY POINT — orchestrates all passes (DESCRIPTION ONLY)
# ---------------------------------------------------------------------------


def phase1_prompt(
    description: str,
    drawing_descriptions: str = "",
) -> dict:
    """Generate all Phase 1 pass prompts.
    
    IMPORTANT: Passes A/B/C receive DESCRIPTION only (no claims).
    Claims are analyzed for the FIRST time in Phase 1.2 (forensic audit).
    """
    has_drawings = bool(drawing_descriptions and drawing_descriptions.strip())

    return {
        "pass_a": phase1_pass_a_prompt(description),
        "pass_b": phase1_pass_b_prompt(description),
        "pass_c": phase1_pass_c_prompt(description),
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

CRITICAL DISAMBIGUATION RULE — SUBJECT MATTER OVER METHOD:
The primary domain MUST reflect the INDUSTRY / SYSTEM / SUBJECT MATTER, not just the
algorithm or methodology used. The ML algorithm is the TOOL, not the system being built.

Examples:
  Speech recognition / speech pre-training system → G10L — speech is the SUBJECT, ML is the TOOL
  Image classification using neural networks → G06V or G06T — images are the SUBJECT, NN is the TOOL
  Neural network architecture improvement (no application) → G06N — the network IS the subject
  Medical diagnosis via ML → A61B — medical is the SUBJECT, ML is the TOOL
  Wireless signal processing using AI → H04W — wireless communication is the SUBJECT
  Generic data processing / computing → G06F

A domain is the SUBJECT MATTER only if the invention IS that thing (e.g. a new ML
architecture IS G06N). If the invention APPLIES ML to solve a problem in another field,
the OTHER field is the primary domain, and G06N is secondary (methodology).

You MUST provide at least 3 domain signals — one primary, at least one secondary
(methodology), and at least one negative (domains to explicitly avoid):
  - primary:   the INDUSTRY / SUBJECT DOMAIN (e.g. G10L for speech)
  - secondary: the METHODOLOGY domain (e.g. G06N for ML techniques)
  - negative:  a domain that looks similar but is WRONG (e.g. G06T for images, because
               this is speech not images)

Also add supporting/secondary domain signals as needed (e.g. G06F for data processing).

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
- You MUST produce at least 3 domain_signals with distinct roles (primary, secondary, negative)

Output format (strict JSON):
{{
  "domain_probabilities": [
    {{
      "class": "G10L",
      "probability": 0.92,
      "reasoning": "Invention is a speech pre-training system — speech is the SUBJECT MATTER; "
                   "ML techniques (G06N) are the methodology, not the primary domain"
    }}
  ],
  "primary_domain": "G10L",
  "primary_confidence": 0.92,
  "reasoning": "Brief explanation of top domain selection, explicitly addressing subject-matter vs methodology"
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# PHASE 1.1 — Emergency Anchor Recovery
# ---------------------------------------------------------------------------


def phase1_1_emergency_anchor_prompt(
    solution_summary: str,
    problem_solved: str,
    technical_object: str,
    core_function: str,
    independent_claim_1: str,
) -> str:
    return f"""You are a CPC classification recovery agent.

The standard extraction pipeline failed to produce a reliable technical domain.
Your task is to identify the SINGLE most accurate 4-character CPC subclass prefix
(e.g., G06F, G10L, G06N) from the information below.

This is an emergency anchor — you are not performing full classification.
You are identifying the technical field so the search pipeline can be restricted
to the correct CPC section.

=== INVENTION INFORMATION ===

TECHNICAL OBJECT (what the invention IS):
{technical_object or "Not available"}

PROBLEM SOLVED (what technical problem it addresses):
{problem_solved or "Not available"}

SOLUTION SUMMARY (how the invention solves the problem):
{solution_summary or "Not available"}

CORE FUNCTION (what the invention DOES — the primary action):
{core_function or "Not available"}

FIRST INDEPENDENT CLAIM (the broadest legal definition of the invention):
{independent_claim_1 or "Not available"}

=== INSTRUCTIONS ===

Step 1 — Read all five fields above.
         If any field says "Not available", rely on the remaining fields.
         The independent claim is the most authoritative — prioritize it.

Step 2 — Identify the PRIMARY technical field.
         Ask: What is the SUBJECT MATTER of this invention?
         The subject matter is what the invention IS ABOUT, not what METHOD it uses.

         Examples of correct subject-matter reasoning:
         - Invention processes speech audio → G10L (not G06N just because ML is used)
         - Invention improves neural network training → G06N (ML IS the subject)
         - Invention controls a vehicle → B60W (not G06F just because software is used)
         - Invention transmits data over a network → H04L (not G06F)
         - Invention seals a mechanical joint → F16J (not G06N)

Step 3 — Select the single best 4-character CPC subclass prefix.
         Must be exactly 4 characters: one letter, two digits, one letter.
         Examples: G10L, G06N, G06F, H04L, B60W, A61B, F16J, G05B

Step 4 — Assign a confidence score (0.0 to 1.0).
         - 0.9+ : The subject matter is unambiguous from the available information
         - 0.7  : Reasonably confident but one field is vague or missing
         - 0.5  : The available information is limited — educated guess
         - Below 0.5 : Do not output — use G06F as fallback instead

Step 5 — Write one sentence of reasoning that references specific words
         from the invention information above.

=== CRITICAL RULES ===

- Output EXACTLY one 4-character CPC prefix — no full codes like G06F 17/00
- If the LLM output contains a longer code, the system will truncate to 4 characters automatically
- Do NOT hedge with multiple options — pick ONE
- Do NOT explain CPC classification theory — just identify the field
- If all five fields are vague or contradictory, output G06F with confidence 0.4
- Output ONLY valid JSON — no markdown, no text outside the JSON object

=== OUTPUT FORMAT ===
{{
  "emergency_anchor": "G10L",
  "confidence": 0.88,
  "reasoning": "One sentence referencing specific words from the invention information above.",
  "anchor_basis": "independent_claim_1"
}}

The anchor_basis field indicates which input field most influenced the decision.
Use "independent_claim_1", "technical_object", "core_function", "solution_summary",
or "combined" if multiple fields contributed equally.

Output ONLY valid JSON. No markdown, no text outside the JSON object.
"""