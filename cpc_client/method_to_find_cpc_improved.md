# Improved Patent CPC Classification Method

## Overview

This document describes the **improved** multi-phase pipeline for classifying patent text into Cooperative Patent Classification (CPC) codes. This version addresses key weaknesses (W1–W8) found in the original method by introducing probabilistic domain inference, 
# definition på Norsk
"Probabilistisk domeninferens er prosessen med å behandle datagenererende domene som en latent stokastisk variabel og estimere dens posterior sannsynlighet P(domenet | observerte data) ved hjelp av probabilistiske modeller (f.eks. Bayes’ regel, blandingsmodeller eller diskriminative probabilistiske klassifikatorer)."
Enkelt eksempel på norsk :

Situasjon: E-postene våre kan komme fra to domener — jobb og privat.
Observasjon: En e-post inneholder ord som "møte", "prosjekt", "vedlegg".
Tolking: Basert på sannsynlighetene for hvert ord i hvert domene (f.eks. P("møte" | jobb) høyere enn P("møte" | privat)), beregner vi P(domenet = jobb | e-posten) — hvis denne posterioren er høy, antar vi at e-posten er fra jobb-domenet.
"
# End Norsk

section-aware extraction, multi-pass validation, per-claim reconciliation, and final consistency checks.

**Key improvements over the original method:**
- **W1**: Hardcoded class injections replaced with probabilistic domain inference
- **W2**: Domain multipliers calibrated with 1.2x default for unknown domains
- **W3**: Section-aware term extraction (downweights background/prior-art)
- **W4**: Term-density guard prevents inappropriate specificity bonuses
- **W5**: Multi-pass validation (one candidate per prompt)
- **W6**: Score margin awareness triggers low-confidence flags
- **W7**: Per-claim reconciliation removes rejected codes
- **W8**: Method vs apparatus claim detection and routing

---

## Phase 0: Input Pre-processing

**Input:** Raw patent text (description + claims)

**Process:**
1. **Section Detection**
   - Automatically detects sections: Abstract, Background, Summary, Detailed Description, Claims
   - Uses regex patterns to identify section headers

2. **Claims Labeling**
   - Parses raw claims and prefixes each with `[INDEPENDENT]` or `[DEPENDENT: ref claim N]`
   - Identifies dependency patterns like "according to claim 1" or "of claim 1"

3. **Text Separation**
   - Splits description from claims if embedded in single text
   - Claims field is optional; can be provided separately

**Output:** Labeled claims + section-mapped description

---

## Phase 1: LLM Extraction (Section-Aware)

**Input:** Labeled claims + section-mapped description + CPC reference hints

**Process:** 10 structured steps

### Step 0 — Section Weighting (CRITICAL)
Assigns reliability weights to patent sections:

| Section | Weight | Reason |
|---------|--------|--------|
| Claims (independent) | 1.2 | Strongest signal, legally binding |
| Summary of Invention | 1.0 | High signal, describes novelty |
| Detailed Description | 0.9 | Implementation details |
| Abstract | 0.6 | Broad summary, may be vague |
| Background/Prior Art | 0.2 | Describes existing technology, NOT the invention |

> **Rule:** Background terms are capped at importance 3 unless reinforced in claims or summary.

### Step 1 — Technical Understanding
Extracts:
- `technical_object` — What is the invention? (1-2 concrete sentences)
- `problem_solved` — Specific technical problem (not generic)
- `solution_summary` — How it solves the problem, focusing on mechanism

### Step 2 — System Context
Identifies the broader technical system or industry domain.

**Rules:**
- Must describe an industry/application, NOT a component
- Ask: "What industry would buy/use this invention?"

### Step 3 — Core Technical Function (CRITICAL)
Defines what the invention DOES (not what it looks like).

**Avoid:**
- Vague wording
- Structural descriptions ("a device comprising...")

**Good example:** "Context-driven retrieval of data fragments by resonance-based selection"
**Bad example:** "A device comprising a housing and processor"

### Step 4 — Claim Type Analysis (NEW — W8 Fix)
For EACH independent claim, classifies:

| Type | Description | Example CPC Routing |
|------|-------------|---------------------|
| METHOD | Process, algorithm, sequence of steps | G06F17/30 (methods) |
| APPARATUS | Device, system, physical structure | G06F15/00 (computers) |
| BOTH | Mixed method and apparatus elements | Both method and apparatus subclasses |

Also extracts per claim:
- `claim_core_function` — Specific function of this claim
- `claim_specific_features` — Technical features unique to this claim

### Step 5 — Essential Terms (Section-Aware — W3 Fix)
Extracts terms with importance adjusted by source section:

```
Adjusted Importance = Base Importance × Section Weight (capped at 10)
```

| Source Section | Base Range | Weight | Adjusted Range |
|----------------|------------|--------|----------------|
| Claims | 9-10 | 1.2 | 9-10 |
| Summary | 7-9 | 1.0 | 7-9 |
| Detailed Description | 5-8 | 0.9 | 5-7 |
| Abstract | 4-7 | 0.6 | 3-4 |
| Background | 1-3 | 0.2 | 1-3 |

Each term records:
```json
{
  "term": "resonance-based selection",
  "importance": 9,
  "justification": "Core inventive mechanism from Claim 1",
  "source_section": "claims"
}
```

### Step 6 — Multi-Invention Detection
Groups independent claims by distinct technical focus.

**Output:**
- `single_invention`: true/false
- `invention_groups`: List of claim groups with focus descriptions

### Step 7 — Classification Strategy
Chooses one of three strategies:

| Strategy | When to Use | Primary Class |
|----------|-------------|---------------|
| system-first | Specific equipment for one industry | Domain code (E21B, B60L) |
| function-first | Generic component across industries | Function code (F16J, G06N) |
| hybrid | Novel function + specific application | Both, ordered by specificity |

**Decision rule:** "Could this invention be deployed in two unrelated industries without modification?"
- NO → system-first
- YES → function-first
- Both equally novel → hybrid

### Step 8 — Initial CPC Class Hypotheses (SOFT — W1 Fix)
Suggests 3-5 CPC classes as **hypotheses**, not enforced decisions.

```json
{
  "class_hypotheses": [
    {"class": "G06F", "confidence": 0.7, "reasoning": "Involves data processing"},
    {"class": "G06N", "confidence": 0.6, "reasoning": "Uses neural network concepts"}
  ]
}
```

> These are soft predictions. The LLM can override them later if evidence is strong.

### Step 9 — Negative Signals (SOFT)
Extracts terms/domains the patent is clearly NOT about, with confidence scores.

```json
{
  "negative_signals": [
    {"term": "image processing", "confidence": 0.8},
    {"term": "computer vision", "confidence": 0.9}
  ]
}
```

> Confidence allows graded penalization rather than absolute exclusion.

### Step 10 — Per-Claim Preliminary CPC
Maps EACH claim to 1-2 specific CPC subclasses (6-8 digits).

**Rules:**
- Claim 1 (independent) → PRIMARY/broadest classes
- Dependent claims → Same as parent OR additional classes if novel
- All marked as `"provisional": true` (subject to Phase 5/6 validation)

```json
{
  "claim_classifications": [
    {
      "claim_number": 1,
      "claim_type": "independent",
      "parent_claim": null,
      "cpc_classes": ["G06F16/00", "G06N3/08"],
      "reasoning": "Core: context-driven retrieval using embeddings",
      "provisional": true
    }
  ]
}
```

---

## Phase 1b: Probabilistic Domain Inference (W1 Fix)

**Replaces:** Hardcoded class injection (E21B, B60, F16J, etc.)

**Process:**
1. LLM receives extracted terms, system context, core function, and initial hypotheses
2. Estimates probability (0.0-1.0) for each relevant CPC domain
3. Returns domain probabilities with reasoning

```json
{
  "domain_probabilities": [
    {"class": "G06F", "probability": 0.75, "reasoning": "Data processing and retrieval"},
    {"class": "G06N", "probability": 0.60, "reasoning": "Vector/tensor representations"}
  ],
  "primary_domain": "G06F",
  "primary_confidence": 0.75
}
```

**Integration:**
- Domains with probability > 0.5 are added to candidate class list
- Domains with probability > 0.3 are considered for scoring boosts
- No domain is forced — all are probabilistic suggestions

---

## Phase 2: XML Expansion + Improved Scoring

**Input:** CPC class candidates from Phase 1/1b

**Process:**

### 1. XML Expansion
- Parse local CPC scheme XML files
- Expand each class into all allocatable subgroups
- Build `full_context` by concatenating parent titles

### 2. Term Importance Processing
- Normalize terms (lowercase, stemming)
- Apply section-aware importance weights
- Build `term_importance` dictionary

### 3. Scoring Engine (per subgroup)

**A. Negative Signal Penalty (Soft)**
```
Penalty = -5.0 × signal_confidence (for terms)
Penalty = -3.0 × domain_confidence (for domains)
```

**B. Term Matching**
1. **Word overlap** (Jaccard-like, IDF-weighted):
   `idf × (importance/5.0) × 3`

2. **Substring match** (exact phrase in context):
   `idf × (importance/5.0) × 5`

3. **Synonym match** (via CPC_SYNONYMS dict):
   `idf × (importance/5.0) × 4`

**C. Context Boosts**
- System context overlap: `idf × 2`
- Core function overlap: `idf × 4`

**D. Domain-Specific Boosting (W2 Fix)**
```python
base_multiplier = 1.2  # Default for unknown domains
if domain in domain_probabilities:
    base_multiplier = 1.0 + (domain_probability × 1.0)
score *= base_multiplier
```

> No more hardcoded 2.0x/3.0x multipliers. All domains use calibrated probabilities.

**E. Specificity Bonus with Term-Density Guard (W4 Fix)**
```python
symbol_depth = symbol.count("/") + sum(symbol.count(d) for d in "0123456789")
if matching_terms >= 2:  # GUARD: only if ≥2 terms match
    depth_bonus = min(symbol_depth × 0.5, 3.0)
    score += depth_bonus
```

> Prevents deep but irrelevant codes from getting artificial boosts.

**F. Score Margin Calculation (W6 Fix)**
```python
score_margin = (top1_score - top2_score) / normalization_denom
if margin > 0.3:   confidence_level = "high"
elif margin < 0.1: confidence_level = "low"
else:              confidence_level = "medium"
```

> Margin is passed to Phase 5 to trigger low-confidence flags when top candidates are close.

**Output:** Top 7 scored candidates with normalized scores, margins, and confidence levels

---

## Phase 3: Ranking

**Process:**
- Sort candidates by normalized score descending
- Keep top 7
- Pass to Phase 5 along with score margin and confidence level

---

## Phase 5: Multi-Pass Validation (W5 Fix)

**Replaces:** Single-prompt validation of all candidates

**Process:**
Each candidate is validated in a **separate LLM call**.

**Input per call:**
- Invention profile (technical_object, core_function, system_context, strategy)
- One candidate CPC code + title + description
- Score margin + confidence level
- Claim type context (method vs apparatus)

### Validation Criteria

| Check | Scale | Pass Threshold |
|-------|-------|----------------|
| Function Alignment | 0.0-1.0 | ≥ 0.6 |
| Context Alignment | 0.0-1.0 | ≥ 0.5 |
| Visual Bias | true/false | Must be false |
| Method/Apparatus | true/false | Must match claim type |
| Specificity Fit | too_broad/appropriate/too_narrow | Not "too_broad" |
| Contradictions | List | Must be empty |

### Decision Rules

**PASS if:**
- function_alignment ≥ 0.6
- context_alignment ≥ 0.5
- visual_bias == false
- method_apparatus_aligned == true
- specificity_fit != "too_broad"

**FAIL if any critical check fails.**

**Low confidence override:**
- If score_margin < 0.1, confidence is forced to LOW regardless of other factors

### Output per candidate
```json
{
  "decision": "PASS",
  "confidence": "high",
  "scores": {
    "function_alignment": 0.9,
    "context_alignment": 0.8,
    "visual_bias": false,
    "method_apparatus_aligned": true
  },
  "specificity_fit": "appropriate",
  "contradictions": [],
  "reasoning": "This class directly covers information retrieval...",
  "rejection_reason": ""
}
```

**Aggregate Phase 5 Output:**
```json
{
  "validated_candidates": [...],
  "filtered_out": [
    {
      "symbol": "G06V10/75",
      "validation": "FAIL",
      "rejection_reason": "Visual bias - class focuses on image/video feature matching, not text/data retrieval"
    }
  ],
  "best_code": {
    "symbol": "G06F16/00",
    "confidence": "high",
    "reasoning": "Validated as best match for context-driven data retrieval"
  }
}
```

---

## Phase 6: Per-Claim Reconciliation (W7 Fix)

**Purpose:** Remove rejected codes from per-claim classifications and replace with validated alternatives.

**Input:**
- Validated CPC codes (passed Phase 5)
- Rejected CPC codes (failed Phase 5) with reasons
- Per-claim provisional classifications from Phase 1

**Process:**
1. For each claim, remove any CPC codes that appear in the rejected list
2. If removing a code leaves the claim empty, replace with best-matching validated code
3. Ensure alignment with:
   - Claim type (method vs apparatus)
   - Claim-specific function
4. Keep provisional codes only if they match a validated code

**Output:**
```json
{
  "reconciled_claims": [
    {
      "claim_number": 1,
      "claim_type": "independent",
      "final_cpc": ["G06F16/00", "G06N3/08"],
      "reasoning": "Aligned with validated codes and claim function"
    },
    {
      "claim_number": 2,
      "claim_type": "dependent",
      "final_cpc": ["G06F16/00"],
      "reasoning": "Does not add new technical area; inherits from Claim 1"
    }
  ],
  "changes_made": [
    "Claim 3: Rejected G06V10/75, replaced with G06F16/00"
  ]
}
```

---

## Phase 7: Final Consistency Check

**Purpose:** Ensure selected CPC codes are coherent and logically fit together.

**Input:**
- Invention profile
- Top 3 validated CPC codes

**Checks:**
1. Do the codes logically fit together? (compatible domains?)
2. Is there a conflicting domain? (e.g., image processing + text retrieval)
3. Is one code clearly dominant/primary?
4. Do codes cover both method and apparatus if needed?
5. Are any codes redundant or overlapping?

**Output:**
```json
{
  "coherent": true,
  "issues": [],
  "recommended_primary": "G06F16/00",
  "recommended_secondary": ["G06N3/08"],
  "reasoning": "Both codes belong to computing domain and cover complementary aspects of the invention"
}
```

---

## Final Output Structure

```json
{
  "premier": {
    "symbol": "G06F16/00",
    "title": "Information retrieval",
    "confidence": "high",
    "reasoning": "Validated as best match for context-driven data retrieval system"
  },
  "per_claim": [
    {
      "claim_number": 1,
      "claim_type": "independent",
      "final_cpc": ["G06F16/00", "G06N3/08"],
      "reasoning": "Core invention: resonance-based data retrieval using embeddings"
    },
    {
      "claim_number": 2,
      "claim_type": "dependent",
      "final_cpc": ["G06F17/30"],
      "reasoning": "Adds indexing structure for wave-encoded data"
    }
  ],
  "phase1": {
    "technical_object": "...",
    "core_function": "...",
    "claim_analysis": [...],
    "class_hypotheses": [...],
    "terms": [...],
    "negative_signals": [...],
    "claim_classifications": [...]
  },
  "phase2": {
    "codes": [...],
    "reasoning": "...",
    "score_margin": 0.35,
    "confidence_level": "high"
  },
  "phase3": [...],
  "phase5": {
    "validated_candidates": [...],
    "filtered_out": [
      {
        "symbol": "G06V10/75",
        "rejection_reason": "Visual bias - focuses on image features"
      }
    ],
    "best_code": {...}
  },
  "phase7": {
    "coherent": true,
    "issues": [],
    "recommended_primary": "G06F16/00"
  },
  "cpc": [
    {"code": "G06F16/00", "score": 0.95},
    {"code": "G06N3/08", "score": 0.87}
  ]
}
```

---

## Example: Wave-Encoder Patent

### Invention Profile
- **Technical Object:** Multi-dimensional probability-wave data processing system
- **Core Function:** Context-driven retrieval by resonance-based selection
- **System Context:** Database search engines, knowledge-base retrieval
- **Claim Types:** Claim 1 = METHOD, Claim 5 = APPARATUS (hardware accelerator)

### Phase 1b: Domain Inference
```
G06F: 0.82 (data processing/retrieval)
G06N: 0.70 (vector/tensor models)
G06K: 0.45 (pattern recognition - weaker match)
G06V: 0.25 (image/video - low, but not zero due to "feature extraction" terms)
```

### Phase 2: Top Candidates (before validation)
1. G06V10/75 — Organisation of matching processes (score: 0.91)
2. G06F16/00 — Information retrieval (score: 0.88)
3. G06N3/08 — Vector or tensor-based models (score: 0.85)
4. G11C29/00 — Content-addressable memory (score: 0.72)

> Note: G06V10/75 scores highest due to term overlap ("matching processes", "context analysis"), but these are image/video-specific terms in G06V.

### Phase 5: Multi-Pass Validation

**Candidate 1: G06V10/75**
- Function Alignment: 0.3 (matching processes mentioned, but for images)
- Context Alignment: 0.2 (image/video domain, not IT/data)
- Visual Bias: TRUE (mentions "image or video features")
- **Decision: FAIL**
- Rejection Reason: "Visual bias — class focuses on image/video feature matching, not text/data retrieval. Context mismatch: IT/data-management invention classified in computer vision domain."

**Candidate 2: G06F16/00**
- Function Alignment: 0.95 (directly covers information retrieval)
- Context Alignment: 0.90 (IT/data-management domain)
- Visual Bias: FALSE
- Method/Apparatus: Matches METHOD claim
- **Decision: PASS**
- Confidence: HIGH

**Candidate 3: G06N3/08**
- Function Alignment: 0.85 (covers vector/tensor embeddings)
- Context Alignment: 0.80 (AI/ML domain, applicable to data retrieval)
- Visual Bias: FALSE
- **Decision: PASS**
- Confidence: HIGH

**Candidate 4: G11C29/00**
- Function Alignment: 0.60 (content-addressable storage)
- Context Alignment: 0.70 (hardware context)
- Visual Bias: FALSE
- Method/Apparatus: Better for APPARATUS claim (Claim 5)
- **Decision: PASS**
- Confidence: MEDIUM

### Phase 6: Per-Claim Reconciliation

**Claim 1 (METHOD):**
- Original: [G06V10/75, G06F16/00]
- After reconciliation: [G06F16/00, G06N3/08]
- Change: Removed G06V10/75 (rejected), kept G06F16/00, added G06N3/08

**Claim 5 (APPARATUS — hardware accelerator):**
- Original: [G06F16/00]
- After reconciliation: [G06F16/00, G11C29/00]
- Change: Added G11C29/00 (content-addressable memory) for hardware storage aspect

### Phase 7: Consistency Check
- Coherent: TRUE
- Issues: []
- Recommended Primary: G06F16/00
- Recommended Secondary: [G06N3/08, G11C29/00]

### Final Output
```json
{
  "premier": {
    "symbol": "G06F16/00",
    "confidence": "high"
  },
  "per_claim": [
    {
      "claim_number": 1,
      "final_cpc": ["G06F16/00", "G06N3/08"],
      "reasoning": "Method claim: retrieval + embeddings"
    },
    {
      "claim_number": 5,
      "final_cpc": ["G06F16/00", "G11C29/00"],
      "reasoning": "Apparatus claim: retrieval + content-addressable hardware"
    }
  ],
  "phase5": {
    "filtered_out": [
      {
        "symbol": "G06V10/75",
        "rejection_reason": "Visual bias - focuses on image/video features"
      }
    ]
  }
}
```

---

## Weaknesses Addressed Summary

| Weakness | Original Problem | Fix | Status |
|----------|-----------------|-----|--------|
| W1 | Hardcoded domain injections (E21B, B60, etc.) | Probabilistic domain inference | ✅ Fixed |
| W2 | Asymmetric multipliers (2x-3x for some domains, 1x for others) | Calibrated 1.2x default, scaled by probability | ✅ Fixed |
| W3 | Background/prior-art terms overweighted | Section-aware extraction with 0.2x background weight | ✅ Fixed |
| W4 | Specificity bonus applied blindly | Term-density guard (≥2 matching terms required) | ✅ Fixed |
| W5 | Single prompt validates all candidates | Multi-pass: one candidate per prompt | ✅ Fixed |
| W6 | No score margin awareness | Margin calculation + confidence level passed to Phase 5 | ✅ Fixed |
| W7 | Rejected codes remain in per-claim classifications | Phase 6 reconciliation removes and replaces | ✅ Fixed |
| W8 | No method vs apparatus detection | Claim type analysis + method/apparatus alignment checks | ✅ Fixed |

---

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | All LLM prompts (Phase 1, 1b, 2, 5, 6, 7) |
| `search_cpc.py` | Pipeline orchestration (Phases 1-7) |
| `extracting_cpc.py` | Phase 1 LLM extraction wrapper |
| `cpc_xml_parser.py` | XML scheme parsing and subgroup expansion |
| `main.py` | FastAPI endpoint |
| `method_to_find_cpc.md` | Original method documentation |
| `method_to_find_cpc_improved.md` | This document |
