# Strategy for Finding CPC Classification

## Overview

This document describes the step-by-step method used by our patent CPC (Cooperative Patent Classification) classification pipeline. The system classifies patent text into specific CPC subgroup codes using a three-phase approach that adapts to whether the invention is **application-specific equipment** or a **generic functional component**.

---

## Files Used

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI endpoint — receives text, combines with optional claims, calls classifier |
| `app/cpc_classification/search_cpc.py` | **Main pipeline** — orchestrates all 3 phases + post-processing |
| `app/cpc_classification/prompts.py` | Phase 1 LLM system prompt with classification strategy logic |
| `app/cpc_classification/extracting_cpc.py` | Phase 1 parser — sends text to LLM, parses JSON response |
| `app/cpc_classification/cpc_xml_parser.py` | XML parser — reads local CPC scheme files, builds hierarchy with parent context |
| `app/search_core/ollama_client.py` | LLM client — sends requests to Ollama/local LLM |
| `app/cpc_classification/resources/cpc_scheme_2026/` | Local CPC XML files (691 files, ~134MB) |
| `streamlit_app.py` | Streamlit UI — displays Phase 1 + Phase 2/3 results |

---

## Step 1: Input Reception

**File:** `app/main.py` (lines 48-66)

The FastAPI endpoint `/classify` receives patent text with an optional `claims` field.

```python
class ClassifyRequest(BaseModel):
    text: str
    claims: str | None = None
```

If claims are provided, they are combined with the description:
```python
full_text = f"DESCRIPTION:\n{req.text}\n\nCLAIMS:\n{req.claims}"
```

The combined text is passed to the classifier.

---

## Step 2: Phase 1 — LLM Extraction

**File:** `app/cpc_classification/search_cpc.py` (lines 184-187)
**File:** `app/cpc_classification/extracting_cpc.py` (lines 70-78)
**File:** `app/cpc_classification/prompts.py` (full file)

### 2.1 Prompt Design

The system prompt (`prompts.py`) instructs the LLM to extract structured information with a **critical decision**: whether the invention is **system-first** (application-specific equipment) or **function-first** (generic component).

**Key extracted fields:**
- `technical_object` — What is the invention? (1-2 sentences)
- `problem_solved` — What technical problem is addressed?
- `system_context` — The broader technical system or industry (e.g., "oil/gas wellhead assembly")
- `core_function` — The primary function (e.g., "sealing / preventing leakage")
- `classification_strategy` — `"system-first"` or `"function-first"`
- `cpc_classes` — 2-4 broad CPC classes (e.g., `["E21B", "F16J"]`)
- `essential_terms` — 5-15 terms with `importance` (1-10) and `justification`

### 2.2 Classification Strategy Logic

**System-first** = the invention is a complete apparatus/machine for a specific industry:
- PRIMARY class = system/application domain (e.g., E21B for wellheads)
- SECONDARY class = core function (e.g., F16J for seals)

**Function-first** = the invention is a generic component usable in many industries:
- PRIMARY class = core function (e.g., F16J for seals)
- SECONDARY class = application domain (e.g., E21B for oil equipment)

### 2.3 LLM Call

```python
response = self.llm.chat(
    system_prompt=prompt,
    user_message=text,
    temperature=0.1,
    max_tokens=2000,
)
```

Temperature is set to `0.1` for deterministic, repeatable outputs.

### 2.4 JSON Parsing

**File:** `app/cpc_classification/extracting_cpc.py` (lines 83-117)

The response is parsed in three attempts:
1. Direct `json.loads(response)`
2. Strip markdown fences (` ```json ... ``` `) and retry
3. Greedy regex fallback (`re.search(r"\{.*\}", response, re.DOTALL)`)

Terms are normalized and sorted by importance descending:
```python
data["essential_terms"] = _normalize_terms(data)
terms.sort(key=lambda x: (-x["importance"], x["term"]))
```

---

## Step 3: Post-Processing — Smart Class Injection

**File:** `app/cpc_classification/search_cpc.py` (lines 201-340)

### 3.1 Determine Strategy

The system checks both the LLM's `classification_strategy` field and applies heuristics:

```python
def _is_system_first(system_context: str, core_function: str) -> bool:
    strong_system_signals = [
        "wellhead", "tubing hanger", "blowout preventer",
        "drilling rig", "engine", "battery pack",
        "reactor", "transmission", "gearbox"
    ]
    # If system context contains strong domain signals → system-first
```

### 3.2 Primary Class Injection (System-First)

If the invention is system-first and the domain is detected, the domain class is injected as **PRIMARY**:

| Domain Signal | Injected Class | Position |
|--------------|----------------|----------|
| `wellhead`, `tubing hanger`, `drilling`, `oil well`, `hydrocarbon` | `E21B` | Primary |
| `vehicle`, `automotive`, `car`, `truck` | `B60` | Primary |

### 3.3 Function Class Injection (Context-Aware)

Function classes are injected as **primary** or **secondary** depending on strategy:

| Function Signal | Class | System-First Position | Function-First Position |
|----------------|-------|----------------------|------------------------|
| `cool`, `cooling`, `thermal management` | `F01P` | Secondary | Primary |
| `seal`, `sealing` | `F16J` | Secondary (if E21B present) | Primary |
| `valve`, `vent`, `bleed` | `F16K` | Secondary (if E21B present) | Primary |
| `battery`, `electric vehicle` | `B60L` | Primary | Secondary |

### 3.4 Class Ordering

Final class list ordering:
```
[PRIMARY injections] + [LLM classes (excluding duplicates)] + [SECONDARY injections]
```

This ensures domain classes (e.g., E21B) are expanded **first** in Phase 2, giving them priority.

### 3.5 Conflict Resolution

- If both `F25` (refrigeration) and `F01P` (machine cooling) are present, remove `F25`:
```python
if "F25" in cpc_classes and "F01P" in cpc_classes:
    cpc_classes = [c for c in cpc_classes if c != "F25"]
```

---

## Step 4: Phase 2 — XML Expansion

**File:** `app/cpc_classification/search_cpc.py` (lines 355-365)
**File:** `app/cpc_classification/cpc_xml_parser.py` (full file)

### 4.1 Parse Local CPC XML Files

For each broad class in the list (e.g., `E21B`, `F16J`), the XML parser loads the corresponding file from:
```
app/cpc_classification/resources/cpc_scheme_2026/cpc-scheme-{CLASS}.xml
```

### 4.2 Parent Title Inheritance

**File:** `app/cpc_classification/cpc_xml_parser.py` (lines 80-119)

Each CPC subgroup inherits its parent titles to build a `full_context` field:

```python
def _parse_item(self, item_elem, parent_chain, parent_titles):
    full_titles = parent_titles + [title]
    full_context = " ".join(filter(None, full_titles))
    return {
        "symbol": symbol,
        "title": title,
        "full_context": full_context,  # Includes all parent titles
        "level": level,
        "is_allocatable": is_allocatable,
    }
```

**Example:** `E21B33/04` (Tubing hangers) inherits context from `E21B33/00` (Wellheads), so matching against "wellhead" or "tubing" succeeds even on the subgroup title.

### 4.3 Cache Management

Parsed results are cached to JSON files:
```
app/cpc_classification/resources/cpc_scheme_2026/cpc-cache-{CLASS}.json
```

Subsequent requests skip XML parsing and load from cache.

---

## Step 5: Phase 2 — TF-IDF Scoring Algorithm

**File:** `app/cpc_classification/search_cpc.py` (lines 367-605)

### 5.1 Tokenization & Normalization

```python
def _normalize_word(word: str) -> str:
    word = word.lower().strip(".,;:!?()[]{}")
    # Basic stemming: remove -ing, -ed, -s, -es
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    return word

def _tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {_normalize_word(w) for w in words if len(w) > 2}
```

### 5.2 Document Frequency (IDF) Calculation

For all subgroups across all expanded classes:
```python
doc_freq = Counter()
for sg in all_subgroups:
    context = sg.get("full_context", "").lower()
    tokens = _tokenize(context)
    for token in tokens:
        doc_freq[token] += 1
```

IDF for a token = `log(total_subgroups / doc_freq[token])`

Rarer terms (appearing in fewer CPC titles) get higher IDF weights.

### 5.3 Term Matching (3 Methods)

For each CPC subgroup and each extracted term:

#### Method 1: Word Overlap (Jaccard-like)
```python
overlap = term_tokens & context_tokens
if overlap:
    overlap_idf = sum(log(title_count / doc_freq[t]) for t in overlap)
    term_score += overlap_idf * (importance / 5.0) * 3
```

#### Method 2: Substring Matching
```python
if term in context:
    avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(term_tokens)
    idf = log(title_count / avg_df)
    term_score += idf * (importance / 5.0) * 5
```

#### Method 3: Synonym Matching
```python
synonyms = _get_expanded_terms(term)
for syn in synonyms:
    if syn in context:
        syn_tokens = _tokenize(syn)
        avg_df = sum(doc_freq.get(t, 1) for t in syn_tokens) / len(syn_tokens)
        idf = log(title_count / avg_df)
        term_score += idf * (importance / 5.0) * 4
```

### 5.4 Synonym Dictionary

**File:** `app/cpc_classification/search_cpc.py` (lines 42-94)

Covers multiple technical domains:

| Domain | Term | Synonyms |
|--------|------|----------|
| Venting | `venting` | deaeration, degassing, air removal, bleeding |
| Cooling | `cooling` | temperature control, heat removal, thermal management |
| Sealing | `sealing` | gasketing, packing, jointing |
| Valves | `valve` | tap, cock, vent, shut-off |
| Oil/Gas | `wellhead` | well head, blowout preventer, christmas tree |
| Oil/Gas | `tubing hanger` | tubing support, casing hanger, production hanger |
| Oil/Gas | `drilling` | boring, earth drilling, well drilling |
| Oil/Gas | `annulus` | annular space, annular void, borehole annulus |

### 5.5 Context Boosts

| Boost Type | Source | Weight |
|-----------|--------|--------|
| System context overlap | `system_context` tokens matching context | `idf * 2` |
| Core function overlap | `core_function` tokens matching context | `idf * 4` |

### 5.6 Class-Specific Boosts

| Class | Trigger Words | Multiplier |
|-------|--------------|------------|
| `E21B` | wellhead, tubing hanger, casing, annulus, drilling | **2.0x** |
| `E21B` | well, borehole, downhole | 1.5x |
| `E21B` | valve, seal, packing | 1.3x |
| `F01P` | vent, deaerat, degas, air, filling | 1.6x |
| `F01P` | cool, heat, thermal, coolant | 1.3x |
| `F16K` | vent, air, bleed, deaerat | 1.6x |
| `F16K` | valve, tap, cock | 1.2x |
| `B60L` | battery, thermal, cool, heat | 1.4x |
| `F16J` | metal, dynamic, mtm | 1.5x |
| `F16J` | seal, packing, gasket | 1.2x |

### 5.7 Domain-Specific Filtering

**B60L filter** (lines 390-405): Skip B60L codes that don't relate to cooling/thermal/battery:
```python
if symbol.startswith("B60L"):
    cooling_related = any(w in context for w in ["cool", "heat", "thermal", "battery"])
    if not cooling_related:
        continue  # Skip this code entirely
```

### 5.8 Specificity Bonus

More specific (deeper) subgroup codes get a small bonus:
```python
symbol_depth = symbol.count("/") + sum(symbol.count(d) for d in "0123456789")
depth_bonus = min(symbol_depth * 0.5, 3.0)
score += depth_bonus
```

`F01P11/028` (deep) gets more bonus than `F01P` (shallow).

### 5.9 Score Normalization

**File:** `app/cpc_classification/search_cpc.py` (lines 580-601)

```python
scores = [s[0] for s in scored]
max_score = max(scores)
median_score = sorted(scores)[len(scores) // 2]
denom = max_score + median_score * 0.5

for score, sg in scored[:5]:
    normalized_score = min(score / denom, 1.0)
```

Normalization uses `(max + 0.5*median)` instead of just `max` for stability against outliers.

---

## Step 6: Phase 3 — Ranking & Output

**File:** `app/cpc_classification/search_cpc.py` (lines 607-629)

### 6.1 Top-5 Selection

Only the top 5 highest-scoring subgroups are returned:

```python
ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:5]
```

### 6.2 Output Format

```json
{
  "phase1": {
    "technical_object": "...",
    "system_context": "...",
    "core_function": "...",
    "classification_strategy": "system-first",
    "cpc_classes": ["E21B", "F16J"],
    "essential_terms": [...]
  },
  "phase2": {
    "codes": ["E21B33/04", "E21B33/068", "F16J15/34"],
    "reasoning": "Ranked by improved TF-IDF scoring..."
  },
  "phase3": [
    {"symbol": "E21B33/04", "title": "Tubing hangers", "score": 1.0},
    {"symbol": "E21B33/068", "title": "...", "score": 0.95}
  ],
  "cpc": [
    {"code": "E21B33/04", "score": 1.0},
    {"code": "E21B33/068", "score": 0.95}
  ]
}
```

---

## Step 7: UI Display

**File:** `streamlit_app.py`

The Streamlit UI displays:
- **Phase 1**: System context, core function, CPC classes, essential terms table with importance bar chart
- **Phase 2/3**: Top 5 ranked CPC codes with titles, scores, and similarity percentages

---

## Example: Wellhead Patent (Tubing Hanger)

**Input:** Patent about tubing hanger with metal seal in wellhead assembly

**Phase 1 LLM Output:**
- `system_context`: "oil/gas wellhead assembly"
- `core_function`: "sealing / preventing leakage"
- `classification_strategy`: "system-first"
- `cpc_classes`: ["E21B", "F16J"]

**Post-Processing:**
- Detects "wellhead" → injects `E21B` as PRIMARY
- Detects "sealing" with system-first → injects `F16J` as SECONDARY
- Final: `["E21B", "F16J"]`

**Phase 2 Scoring:**
- Expands `E21B` → finds `E21B33/04` (Tubing hangers), gets 2.0x boost for "wellhead"
- Expands `F16J` → finds `F16J15/34` (Metallic packing), gets 1.5x boost for "metal"
- Top result: `E21B33/04` with score ~1.0

**Final Output (Top 5):**
1. `E21B33/04` — Tubing hangers (score: 1.0)
2. `E21B33/068` — Wellheads with valves (score: 0.95)
3. `E21B34/02` — Valves for wells (score: 0.88)
4. `F16J15/34` — Metallic packing (score: 0.72)
5. `F16J15/16` — Seal rings (score: 0.65)

---

## Key Design Decisions

1. **No Hardcoded Blacklists** — Domain filters (like B60L filtering) are conditional context checks, not fixed code lists
2. **System-First vs Function-First** — The pipeline adapts classification strategy based on whether the invention is domain-specific equipment or a generic component
3. **Local XML Files** — No external API dependency for CPC scheme lookup; all 691 CPC XML files are parsed locally
4. **Parent Context Inheritance** — Subgroups inherit parent titles for richer matching without manual rule expansion
5. **Synonym Expansion** — Technical CPC terminology (deaeration, degassing) is mapped to patent text terms (venting, bleeding)
6. **Top-5 Only** — The pipeline returns only the 5 most relevant subgroup codes to avoid information overload

---

## Architecture Diagram

```
Patent Text (+ optional Claims)
    ↓
[FastAPI] app/main.py
    ↓
[CPCClassifier] app/cpc_classification/search_cpc.py
    ├── Phase 1: LLM Extraction
    │     ├── prompts.py (system-first vs function-first strategy)
    │     ├── extracting_cpc.py (JSON parsing + term normalization)
    │     └── OllamaClient (local LLM, temp=0.1)
    │
    ├── Post-Processing: Smart Injection
    │     ├── Detect domain (wellhead → E21B, vehicle → B60)
    │     ├── Detect function (cooling → F01P, sealing → F16J)
    │     └── Order: PRIMARY + existing + SECONDARY
    │
    ├── Phase 2: XML Expansion
    │     ├── cpc_xml_parser.py (parse XML + parent title inheritance)
    │     └── Load from cache if available
    │
    ├── Phase 2: Scoring
    │     ├── Tokenize + stem
    │     ├── Calculate IDF across all subgroups
    │     ├── Term matching (overlap + substring + synonym)
    │     ├── System context boost (idf * 2)
    │     ├── Core function boost (idf * 4)
    │     ├── Class-specific boosts (E21B: 2.0x, F01P: 1.6x)
    │     ├── Domain filtering (B60L non-cooling skip)
    │     ├── Specificity bonus (depth * 0.5)
    │     └── Normalize (score / (max + 0.5*median))
    │
    └── Phase 3: Ranking
          └── Return top 5 codes

    ↓
JSON Response → Streamlit UI / API Client
```
