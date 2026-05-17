"""
prompt_phase1_2.py — Phase 1.2 Claim-to-Domain Forensic Audit.

Phase 1.2 is the mandatory logic gate between Phase 1 (extraction) and Phase 2 (search).
It validates Phase 1's domain signals against the FULL legal claims text to:
  1. Detect hallucinations (domains with zero claim support)
  2. Identify hybrid inventions (multiple domains functionally linked in claims)
  3. Establish a legally-defensible primary anchor from claim text evidence
"""

import json


def phase1_2_audit_prompt(phase1: dict, full_labeled_claims: str) -> str:
    domain_signals = phase1.get("domain_signals", [])
    domain_signals_str = json.dumps(domain_signals, indent=2)

    terms = phase1.get("terms", phase1.get("essential_terms", []))
    top_terms = sorted(
        [t for t in terms if isinstance(t, dict)],
        key=lambda x: x.get("importance", 0),
        reverse=True,
    )[:10]
    top_terms_str = json.dumps(top_terms, indent=2)

    primary_domain = phase1.get("primary_domain", {})
    primary_str = json.dumps(primary_domain, indent=2)

    # Truncate claims to reasonable limit — Phase 1 uses 2000, Phase 1.2 needs more
    MAX_CLAIMS_CHARS = 8000
    truncated_claims = (
        full_labeled_claims[:MAX_CLAIMS_CHARS]
        if len(full_labeled_claims) > MAX_CLAIMS_CHARS
        else full_labeled_claims
    )
    if len(full_labeled_claims) > MAX_CLAIMS_CHARS:
        truncated_claims += "\n\n[CLAIMS TRUNCATED — full text used for audit, output limited for display]"

    return f"""### MISSION ###
Analyze the provided CLAIMS against the SUGGESTED DOMAINS from Phase 1 extraction.
You act as a PATENT LOGIC AUDITOR — your sole purpose is to validate that every
domain signal has LEGAL SUPPORT in the claims text.

### INPUT DATA ###
PRIMARY DOMAIN (from Phase 1):
{primary_str}

SUGGESTED DOMAINS (from Phase 1 extraction):
{domain_signals_str}

TOP COMPONENTS / TERMS (from Phase 1 extraction):
{top_terms_str}

FULL CLAIMS (labeled — [INDEPENDENT] marks the broadest claim):
{truncated_claims}

### STEP 1: TERM-CLAIM MAPPING ###
For each suggested domain, list the EXACT claim numbers where supporting keywords appear.
Include the specific term from the claim that matches the domain.
- Domain A (e.g., G10L — Speech): [Claim 1: "speech frames", Claim 3: "phoneme models", Claim 5: "acoustic features"]
- Domain B (e.g., G06T — Image): [Claim 8: "image pixels"]  OR  [NONE — zero keyword density]

If a domain has ZERO supporting keywords in the claims, mark it: [NONE — HALLUCINATION].

### STEP 2: ANCHOR VALIDATION ###
- If domain keywords appear in Claim 1 ([INDEPENDENT]), it is a PRIMARY ANCHOR — this is the strongest possible signal because Claim 1 defines the broadest scope of the invention.
- If domain keywords appear ONLY in dependent claims, it is a SECONDARY ANCHOR — the invention may include this aspect but it is not the core.
- If a domain has ZERO keyword density across ALL claims, it is a HALLUCINATION — REJECT it regardless of Phase 1 confidence score.

### STEP 3: HYBRID TYPE DETERMINATION ###
If two or more domains BOTH have claim support, you MUST classify the relationship:
- HORIZONTAL HYBRID: Two domains contribute equally to the technical character.
  Example: "A video conferencing system" — both audio (G10L) and video (G06V) are
  core subjects. Both are PRIMARY. Bridge clause connects them as peers.
- VERTICAL HYBRID: One domain is the SUBJECT (what the invention IS), the other
  is the TOOL/METHOD (how it achieves the result).
  Example: "Speech recognition using neural networks" — Speech (G10L) IS the
  subject, Neural Networks (G06N) is the TOOL. G10L = PRIMARY, G06N = SECONDARY.
  Example: "Image classification via ML" — Images (G06V/T) = PRIMARY, ML (G06N) = SECONDARY.

RULING TEST for Vertical vs Horizontal:
  Ask: "If I remove domain B from the claims, does the invention still exist?"
  - If YES → domain B is a secondary tool/method (VERTICAL, mark as SECONDARY_ANCHOR)
  - If NO → domain B is an integral subject (HORIZONTAL, mark both as PRIMARY_ANCHOR)

For ANY invention that uses AI/ML as a technique on a specific subject matter
(speech, images, medical data, vehicles, signals), the ruling is ALWAYS:
  PRIMARY_ANCHOR = the SUBJECT domain (what is being processed/controlled)
  SECONDARY_ANCHOR = G06N (the ML methodology)
The invention IS the speech recognizer, not a new ML architecture. G06N is never
the primary anchor unless the invention IS a new neural network design itself.

### CRITICAL RULES ###
- Claim 1 ([INDEPENDENT]) carries the most weight. It defines the broadest legal scope.
- A 0.90 confidence score from Phase 1 means NOTHING if zero claims support it.
- For each REJECTED domain, you MUST explain WHY: cite the specific vocabulary it requires that is absent from the claims.
- The legal_justification must reference specific claim numbers and wording.

### G06N vs G06F DISAMBIGUATION (MANDATORY) ###
Neural network / ML terms ALWAYS map to G06N, NEVER to G06F:
  → encoder, decoder, prediction head, hidden states, attention, embedding,
    model parameters, loss function, iterative training, weights, gradients,
    self-supervised, clustering labels, neural network, backpropagation,
    pre-training model, feature extraction model, LSTM, transformer.

G06F requires EXPLICIT computing infrastructure vocabulary absent from ML:
  → processor, memory, database, file system, operating system, program code,
    data structure, register, bus, cache, CPU, GPU as bare hardware terms.

If ALL of a candidate domain's evidence terms are from the G06N list above,
REJECT that domain and attribute its terms to G06N instead.
Example: If G06F is supported ONLY by "encoder + model parameters + iterative
training", REJECT G06F — those terms belong to G06N.

### OUTPUT FORMAT (JSON ONLY) ###
{{
  "audit_status": "SUCCESS",
  "term_claim_mapping": [
    {{
      "domain": "G10L",
      "cpc_family": "G10L",
      "supporting_claims": [
        {{"claim_number": 1, "evidence_terms": ["speech frames", "phoneme models"]}},
        {{"claim_number": 3, "evidence_terms": ["acoustic features"]}}
      ],
      "keyword_density": 3,
      "anchor_type": "PRIMARY_ANCHOR"
    }},
    {{
      "domain": "G06N",
      "cpc_family": "G06N",
      "supporting_claims": [
        {{"claim_number": 1, "evidence_terms": ["neural network", "self-supervised training"]}}
      ],
      "keyword_density": 2,
      "anchor_type": "SECONDARY_ANCHOR"
    }},
    {{
      "domain": "G06T",
      "cpc_family": "G06T",
      "supporting_claims": [],
      "keyword_density": 0,
      "anchor_type": "HALLUCINATION"
    }}
  ],
  "final_primary_anchor": "G10L",
  "secondary_anchors": ["G06N"],
  "rejected_domains": [
    {{
      "domain": "G06T",
      "cpc_family": "G06T",
      "phase_1_confidence": 0.90,
      "rejection_reason": "Zero keyword density in claims. Image/video vocabulary (pixel, camera, visual, frame in visual sense) is absent from all claims."
    }}
  ],
  "hybrid_detected": true,
  "hybrid_type": "vertical",
  "hybrid_ruling": "G10L is the SUBJECT (speech system), G06N is the TOOL (ML methodology). Invention would still address speech processing even with different clustering algorithms.",
  "bridge_clause_found": null,
  "audit_confidence": 0.95,
  "legal_justification": "Claim 1 recites 'a method for speech frame clustering using phoneme-derived initial centers' which anchors the invention in G10L (Speech Processing). Claim 1 also recites 'a self-supervised training model' supporting G06N (ML) as a secondary methodology. No claims mention image, video, pixel, camera, or visual processing — the G06T domain signal is unsupported by the claims."
}}

Output ONLY valid JSON. No markdown, no text outside the JSON object.
CRITICAL: Use lowercase null, true, false — NOT NULL, True, or FALSE.
"""
