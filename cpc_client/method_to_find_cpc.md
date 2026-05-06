# Patent CPC Classification Method

## Overview

This document describes the multi-phase pipeline used to classify patent text into Cooperative Patent Classification (CPC) codes. The method combines LLM-based extraction, XML-based scoring, and a validation gate to ensure classifications align with the invention's true technical nature.

---

## Phase 1: LLM Extraction (Understanding the Invention)

**Input:** Patent description text + labeled claims

**Process:**
1. **Claims Pre-processing**
   - Raw claims are labeled as `[INDEPENDENT]` or `[DEPENDENT: ref claim N]`
   - Only independent claims are used for term extraction (they carry higher weight)

2. **LLM Analysis (9 Steps)**
   The LLM is prompted to extract:

   - **Step 1: Technical Understanding**
     - `technical_object` — What is the invention? (1-2 concrete sentences)
     - `problem_solved` — What specific technical problem is addressed?
     - `solution_summary` — How does the invention solve the problem?

   - **Step 2: System Context**
     - The broader technical system or industry (e.g., "database search engines", "oil/gas wellhead assembly")

   - **Step 3: Core Technical Function**
     - The PRIMARY function the invention performs (what it DOES, not what it looks like)
     - Example: "context-driven retrieval of data fragments by resonance-based selection"

   - **Step 4-5: Essential Technical Terms**
     - From DESCRIPTION: 5-10 terms, importance 1-10
     - From INDEPENDENT CLAIMS: 3-8 terms, importance starts at 9 (2x weight)
     - Generic words excluded: device, system, apparatus, plurality, comprising

   - **Step 6: Multi-Invention Check**
     - Detects if independent claims cover distinct inventions
     - Groups claims by technical focus

   - **Step 7: Classification Strategy**
     - `system-first` — Complete apparatus for a specific industry (primary = domain code)
     - `function-first` — Generic component usable across industries (primary = function code)
     - `hybrid` — Both domain and function are co-primary

   - **Step 8: CPC Class Selection**
     - Selects 2-4 CPC classes (4-character codes like G06F, G06N)
     - Uses domain guidance (e.g., neural networks → G06N, wellhead → E21B)
     - Verifies against CPC reference

   - **Step 9: Negative Signals**
     - Generates terms/domains the patent is clearly NOT about
     - Used to penalize incorrect classifications in Phase 2

   - **Step 10: Per-Claim Classification**
     - Maps EACH claim (independent + significant dependents) to 1-2 specific CPC subclasses
     - Example: Claim 1 → G06F16/00, G06N3/08

**Output:** Structured JSON with all extracted fields including `claim_classifications`

---

## Phase 1b: Post-Processing (Smart Class Injection)

**Purpose:** Inject known domain classes based on hardcoded keyword rules when the LLM might miss them.

**Process:**
1. Determine if `system-first` based on strategy or heuristic keyword matching
2. **Primary Injections** (prepended to class list):
   - Wellhead/drilling → E21B
   - Vehicle → B60
   - Electrical/battery → B60L
3. **Secondary Injections** (appended):
   - Cooling → F01P
   - Sealing → F16J
   - Valve → F16K
4. Extract negative signals and domains from Phase 1

**Example:**
```
Input classes from LLM: ["G06N", "G06F"]
After injection (wellhead context): ["E21B", "G06N", "G06F", "F16J"]
```

---

## Phase 2: XML Expansion + TF-IDF Scoring

**Input:** CPC class candidates from Phase 1/1b

**Process:**
1. **XML Expansion**
   - Parse local CPC scheme XML files (`cpc_scheme_2026/`)
   - Expand each class into all allocatable subgroups
   - Build `full_context` by concatenating parent titles

2. **Term Importance Processing**
   - Normalize terms (lowercase, stemming)
   - Claims terms get 2x importance multiplier
   - Build `term_importance` dictionary

3. **Scoring Engine** (per subgroup)
   
   **A. Negative Signal Penalty**
   - Negative signal match: `-5.0`
   - Negative domain match: `-3.0`
   
   **B. Domain-Specific Filtering**
   - Skip B60L if not electrical/vehicle patent
   - Filter F16J in wellhead context for seal-type relevance
   - Skip irrelevant E21B subgroups based on core function
   
   **C. Term Matching**
   - **Word overlap** (Jaccard-like, IDF-weighted): `idf × (importance/5) × 3`
   - **Substring match** (exact phrase in context): `idf × (importance/5) × 5`
   - **Synonym match** (via CPC_SYNONYMS dict): `idf × (importance/5) × 4`
   
   **D. Context Boosts**
   - System context overlap: `idf × 2`
   - Core function overlap: `idf × 4`
   
   **E. Class-Specific Multipliers**
   - E21B + wellhead terms: `×2.0`
   - E21B29 + explosive + cutter: `×3.0`
   - F01P + vent/deaeration: `×1.6`
   - F16K + vent/air/bleed: `×1.6`
   - B60L + battery/thermal: `×1.4`
   - F16J + metal seals: `×1.5`
   - **G06F/G06N: No multipliers** (relying on term overlap)
   
   **F. Specificity Bonus**
   - Deeper codes get `+0.5` per depth level, capped at `+3.0`
   
   **G. Normalization**
   - `normalized_score = score / (max_score + 0.5 × median_score)`
   - Capped at `1.0`

**Output:** Top 7 scored candidates with normalized scores

---

## Phase 3: Ranking

**Process:**
- Sort all scored candidates by normalized score descending
- Keep top 7 candidates
- Format for Phase 5 validation

**Output:**
```json
[
  {"symbol": "G06F16/00", "title": "Information retrieval", "score": 0.95},
  {"symbol": "G06N3/08", "title": "Vector or tensor-based models", "score": 0.87}
]
```

---

## Phase 5: Validation Gate (Replaces Old Phase 4)

**Purpose:** Validate each candidate against the invention's true technical nature. Filter out mismatches.

**Input:** 
- Invention profile (technical_object, problem_solved, core_function, system_context, strategy)
- Top 7 candidates from Phase 3

**Process:**
1. LLM receives validation prompt with invention profile + candidates
2. For EACH candidate, checks 5 criteria:
   
   **1. Function Alignment (Critical)**
   - Does the CPC class describe the CORE FUNCTION?
   - Example: If core function is "context-driven retrieval", does the class describe retrieval/search?
   
   **2. Context Alignment (Critical)**
   - Does the class fit the SYSTEM CONTEXT?
   - Example: If context is "database search engines", is the class in IT/data management?
   
   **3. Visual Bias Check (Critical)**
   - If class mentions image/video/pixel/camera/visual:
     - PASS only if invention actually processes images
     - FAIL if invention processes text/data/signals
     - G06V classes scrutinized carefully
   
   **4. Strategy Alignment**
   - `system-first` → prefer domain codes
   - `function-first` → prefer function codes
   - `hybrid` → both represented
   
   **5. Specificity**
   - Does class capture specific novel contribution?
   - Prefer "resonance-based selection" over generic "information retrieval"

3. **Separate Results:**
   - `PASS` candidates → validated_candidates
   - `FAIL` candidates → filtered_out (with rejection_reason)

4. **Select Best Code:**
   - Only from `PASS` candidates
   - If all fail → fallback to highest-scoring original with low confidence

**Output:**
```json
{
  "validated_candidates": [...],
  "filtered_out": [...],
  "best_code": {"symbol": "...", "confidence": "...", "reasoning": "..."},
  "validation_summary": "3 of 7 candidates passed validation. Best code: G06F16/00."
}
```

---

## Final Output Structure

The API returns a single JSON object:

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
      "parent_claim": null,
      "cpc_classes": ["G06F16/00", "G06N3/08"],
      "reasoning": "Core invention: context-driven data retrieval using embeddings"
    },
    {
      "claim_number": 2,
      "claim_type": "dependent",
      "parent_claim": 1,
      "cpc_classes": ["G06F17/30"],
      "reasoning": "Adds indexing structure for wave-encoded data"
    }
  ],
  "phase1": { ... },
  "phase2": { ... },
  "phase3": [ ... ],
  "phase5": {
    "validated_candidates": [...],
    "filtered_out": [...],
    "best_code": {...},
    "validation_summary": "..."
  },
  "cpc": [
    {"code": "G06F16/00", "score": 0.95},
    {"code": "G06N3/08", "score": 0.87}
  ]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `premier` | The single best validated CPC class for the overall invention |
| `per_claim` | Claim-by-claim classification showing which CPC subclasses cover each claim |
| `phase1` | Raw LLM extraction output (technical_object, terms, strategy, etc.) |
| `phase2` | Scoring methodology explanation |
| `phase3` | Top 7 candidates from TF-IDF scoring before validation |
| `phase5` | Validation results: passed candidates, filtered candidates, best code, summary |
| `cpc` | Final ranked list of CPC codes (from validated candidates or fallback) |

---

## Key Design Principles

1. **Function-First Classification**
   - The model classifies by WHAT the invention DOES, not what it LOOKS LIKE
   - Core technical function drives the primary classification

2. **Claims-Aware**
   - Independent claims get 2x weight in term extraction
   - Per-claim classifications map specific claims to their CPC areas

3. **Domain-Specific Boosting**
   - Mechanical domains (oil/gas, automotive) have aggressive class injection and multipliers
   - Computing domains rely on term overlap + validation gate

4. **Negative Signal Filtering**
   - Explicitly excludes domains the patent is NOT about
   - Prevents false positives in closely related areas

5. **Validation Gate**
   - Final quality check against the invention profile
   - Catches LLM hallucinations and XML scoring artifacts
   - Explicit rejection reasons for transparency

---

## Example: Wave-Encoder Patent

**Invention:** Multi-dimensional probability-wave data processing system

**Phase 1 Extraction:**
- Core Function: "Context-driven retrieval of data fragments by resonance-based selection"
- System Context: "Database search engines, knowledge-base retrieval platforms"
- Strategy: function-first

**Phase 2 Scoring (Top Candidates):**
1. G06V10/75 — Organisation of matching processes (score: 0.91)
2. G06F16/00 — Information retrieval (score: 0.88)
3. G06N3/08 — Vector or tensor-based models (score: 0.85)

**Phase 5 Validation:**
- G06V10/75 → **FAIL** (Visual bias — focuses on image/video features, not text retrieval)
- G06F16/00 → **PASS** (Matches core function: information retrieval)
- G06N3/08 → **PASS** (Matches embeddings/vector representations)

**Final Output:**
```json
{
  "premier": {
    "symbol": "G06F16/00",
    "title": "Information retrieval",
    "confidence": "high"
  },
  "per_claim": [
    {
      "claim_number": 1,
      "cpc_classes": ["G06F16/00", "G06N3/08"],
      "reasoning": "Core invention: resonance-based data retrieval using wave-encoded embeddings"
    }
  ],
  "phase5": {
    "validated_candidates": [
      {"symbol": "G06F16/00", "validation": "PASS", "confidence": "high"},
      {"symbol": "G06N3/08", "validation": "PASS", "confidence": "high"}
    ],
    "filtered_out": [
      {
        "symbol": "G06V10/75",
        "validation": "FAIL",
        "rejection_reason": "Visual bias — class focuses on image/video feature matching, not text/data retrieval"
      }
    ]
  }
}
```
