# Patent CPC FastAPI - Updated File Descriptions

## Overview

This directory contains the FastAPI backend for patent CPC (Cooperative Patent Classification) classification. The system uses a multi-phase pipeline separating **semantic understanding** (Phase 1) from **CPC classification** (Phases 2-8). The architecture is designed for production-grade CPC classification with deterministic scoring and constrained LLM usage.

**Last Updated:** 2026-05-12

---

## Directory Structure

```
app/
├── main.py                          # FastAPI application entry point
├── files_description.md             # Original file descriptions (legacy)
├── files_description_updated.md     # This file - updated documentation
├── precompute_cpc_embeddings.py     # Script to precompute CPC embeddings
├── search_core/                     # LLM client module
│   ├── __init__.py
│   └── ollama_client.py            # Ollama API client wrapper
└── cpc_classification/              # Core classification pipeline
    ├── __init__.py
    ├── analysing_cpc.py            # CPC analysis utilities
    ├── cpc_role_classifier.py      # Phase 1.5: Invention role classification (NEW)
    ├── cpc_chroma_store.py         # ChromaDB vector store integration
    ├── cpc_cross_domain_validator.py  # Phase 3.6: Cross-domain validation (NEW)
    ├── cpc_cross_encoder.py        # Cross-encoder reranker (NEW)
    ├── cpc_decision_tree.py        # Phase 3.5: Decision tree constraints (NEW)
    ├── cpc_family_router.py        # Phase 2A (LEGACY): CPC family routing
    ├── cpc_layer_decomposer.py    # Phase 2A (REPLACEMENT): Layer decomposition (NEW)
    ├── cpc_fast_index.py           # Fast indexing for CPC lookup
    ├── cpc_hierarchy_engine.py     # Phase 3.6: Universal hierarchy (DEPRECATED)
    ├── cpc_hypothesis_consolidation.py  # Phase 4: Cluster hypotheses
    ├── cpc_hypothesis_resolver.py       # Phase 5: Deterministic resolver + Tri-Pillar
    ├── cpc_phase2d_anchor.py          # Phase 2D: Subclass structural anchor filter (NEW)
    ├── cpc_role_labeling.py           # Phase 8: 3-layer role labeling + Executive Report
    ├── cpc_xml_parser.py           # XML parser for CPC scheme files
    ├── epo_client.py              # EPO API client (optional/experimental)
    ├── extracting_cpc.py          # Phase 1: Semantic extraction only
    ├── knowledge_graph.py         # Knowledge graph with semantic embeddings
    ├── prompts.py                 # LLM prompts for all phases (UPDATED)
    ├── search_cpc.py              # Main pipeline orchestrator (Phases 1-8)
    ├── utils.py                   # Utility functions
    └── tests/                     # Test suite
        ├── run_full_pipeline.py
        ├── run_phase2.py
        ├── run_phase2_simulation.py
        ├── test_bm25_quick.py
        ├── test_cross_domain_fix.py       # Domain routing tests
        ├── test_cross_domain_validation.py # Phase 3.6 tests (NEW)
        ├── test_generalized_domains.py     # Multi-domain tests (NEW)
        ├── test_hierarchy_priority.py      # Phase 3.5 tests (NEW)
        ├── test_hybrid_retrieval.py
        ├── test_modality_routing.py
        ├── test_phase2_refactor.py
        ├── test_phase35.py                 # Phase 3.5 tests (NEW)
        ├── test_phase36.py                 # Phase 3.6 tests (NEW)
        ├── test_phase4.py
        ├── test_phase5.py
        ├── test_phase8_role_labeling.py    # Phase 8 tests (NEW)
        └── resources/
    └── resources/                 # Data files and caches
        ├── chroma_db/             # ChromaDB vector database files
        ├── cpc_scheme_2026/       # CPC scheme XML files (source data)
        ├── cpc_graph.pkl          # Knowledge graph cache (generated)
        ├── cpc_embeddings.npz     # Embedding vectors cache (generated)
        ├── cpc_graph_meta.json    # Graph metadata (generated)
        └── ipc_cpc_hints.txt      # IPC/CPC classification hints
```

---

## Core Pipeline Files

### `search_cpc.py`
**Purpose:** Main pipeline orchestrator - coordinates all classification phases (Phases 1-8)

**Pipeline Phases (Current):**
1. **Phase 1** - Semantic extraction (terms, context, function) - **NO CPC codes**
2. **Phase 1.5** - Invention role classification (CORE_TECH/SYSTEM/APPLICATION/SUPPORT)
3. **TCR** - Technical Weight Analysis (computational vs physical dominance)
4. **Phase 2A** - Layer Decomposition: multi-layer CPC mapping (pure_software, data_reasoning, interaction, control, application)
5. **Phase 2B** - Restricted XML expansion (only within technical-layer 4-char prefixes, pre-filtered to existing XML files)
6. **Phase 2C** - Hybrid Scoring: TF-IDF (bigrams) + Semantic Embeddings (0.4×TF-IDF + 0.6×Semantic) + Find-Until-Full expansion (500→1000→all)
7. **Phase 2D** - Subclass Structural Anchor: filters candidates against Phase 2A technical-layer anchors, excludes non-technical families (G06Q, etc.)
8. **Phase 3** - Rank top 20 candidates
9. **Phase 3.5** - Decision Tree: domain dominance, disambiguation, canonical sorting (standard before indexing), quota guardrail
10. **Phase 3.6** - Cross-Domain Validation: entity table, layer validation, family lock
11. **Phase 4** - Hypothesis consolidation: clusters into max 2 hypotheses + interpretation engine
12. **Phase 5** - Deterministic resolution + Tri-Pillar FACETS (Goal/Method/Context) + back-scanning
13. **Phase 8** - Executive Classification Report: At-a-Glance card, Tech Stack table, Professional Justification, Suggested Indexing Codes, Download button

**Key Class:** `CPCClassifier`
- Accepts optional `knowledge_graph` parameter
- Falls back to domain-signal routing if KG unavailable
- Returns structured JSON with all phase results

---

## Phase Modules (In Pipeline Order)

### `extracting_cpc.py` - Phase 1
**Purpose:** Extract semantic understanding from patent text (NO CPC classes)

**Key Functions:**
- `extract()` - Main extraction method
- Term normalization and importance weighting
- Section-aware extraction (claims get 2x weight)

**Output:**
- `technical_object` - What the invention is
- `core_function` - What it does (action/operation)
- `system_context` - Industry/domain
- `primary_domain` - {name, cpc_class, confidence} (NEW)
- `domain_signals[]` - Technical domains with confidence and CPC family
- `disambiguated_terms[]` - Resolved ambiguous terms (NEW)
- `terms[]` with importance scores
- `negative_signals[]` with penalize_family (NEW)
- `classification_strategy` - system-first vs function-first vs hybrid

**Important:** Phase 1 does NOT output CPC classes. All CPC mapping is handled by Phase 2A.

---

### `cpc_role_classifier.py` - Phase 1.5
**Purpose:** Classify invention role (CORE_TECH / SYSTEM / APPLICATION / SUPPORT)

**Key Class:** `CPCRoleClassifier`

**Role Definitions:**
- **CORE_TECH**: Inventions modifying underlying technology (algorithms, model architecture, training methods)
- **SYSTEM**: Inventions orchestrating/coordination multiple components (pipelines, multi-component systems)
- **APPLICATION**: Inventions applying known technology to specific domain (medical, automotive, finance)
- **SUPPORT**: Auxiliary functionality (logging, storage, UI, monitoring)

**Critical Rules:**
1. Presence of AI/ML does NOT imply CORE_TECH
2. If AI is used WITHOUT modifying internal structure → SYSTEM
3. If multiple components interact → bias toward SYSTEM
4. If no clear algorithmic innovation → NOT CORE_TECH

**Output:**
```python
{
    "role": "SYSTEM",
    "confidence": 0.85,
    "reasoning": ["multi-component interaction detected", "AI used as tool"],
    "evidence": ["pipeline orchestration", "multi-model coordination"]
}
```

---

### `cpc_layer_decomposer.py` - Phase 2A (REPLACED)
**Purpose:** Multi-layer CPC decomposition (replaces family routing)

**Key Class:** `CPCLayerDecomposer`

**4-Layer Model:**
- **Application Layer**: What system is FOR → B60W, B60R, A61B
- **Data/Reasoning Layer**: How knowledge is represented → G06F16, G06F40
- **Interaction Layer**: User/system interface → G10L, G06F40, G06N
- **Control Layer**: System orchestration → G05B, G05D

**NEW: Mandatory Domain Anchor (Prompt 1 Fix):**
- If Phase 1 identifies high-confidence primary domain (≥0.7), boost associated CPC families
- Domain anchor map: NLP→G10L, Vehicle Control→B60W, AI→G06N
- Prevents semantic drift into wrong CPC sections

**NEW: AI + Automotive Co-occurrence Rules (Prompt 1 Fix):**
- When invention involves BOTH AI and Automotive keywords:
  - Boost software layers (control, interaction) ×1.3
  - Soft demotion of hardware layers (H02, B60Q) UNLESS hardware keywords present
- Prevents drift into motor switches (H02P) or signaling (B60Q)

**Output:**
```python
{
    "layers": {
        "application": [{"symbol": "B60W", ...}],
        "interaction": [{"symbol": "G10L", ...}],
        "control": [{"symbol": "G05B", ...}],
    },
    "primary_layer": "interaction",
    "layer_scores": {"interaction": 28.47, "application": 6.60, ...},
    "ai_role": "as_tool",
    "source": "layer_decomposition"
}
```

---

### `cpc_xml_parser.py` - Phase 2B
**Purpose:** Parse CPC scheme XML files into usable data structures

**Key Functions:**
- `expand_classes()` - Expand CPC class codes into subgroups
- Supports `allowed_roots` parameter for Phase 2B filtering (98% search space reduction)
- Parse XML hierarchy (parent-child relationships)
- Extract titles, symbols, and allocatable flags

**Input:** `cpc_scheme_2026/cpc-scheme-*.xml`
**Output:** List of subgroup dictionaries with symbol, title, level, parent_chain

---

### `cpc_decision_tree.py` - Phase 3.5
**Purpose:** Apply deterministic, rule-based constraints to CPC candidates

**Key Class:** `CPCDecisionTreeConstraint`

**NEW: Functional Verb Filter (Prompt 2 Fix):**
- Uses "What the invention DOES" (verbs) to override "What it mentions" (nouns)
- Verb-to-CPC mapping:
  - "interpreting"/"generating" → G06F40, G10L, G06N (×2.0)
  - "controlling"/"executing" → B60W, G05B (×2.0)
  - "training"/"optimizing" → G06N3/ (×2.0)

**NEW: Keyword Gravity (Prompt 2 Fix):**
- Reduces TF-IDF weight of ambiguous nouns NOT in Technical Object
- Examples: "state" (0.3x), "switch" (0.4x), "voltage" (0.3x)
- Prevents drift from keyword traps

**Steps:**
1. Domain Dominance - boost matching domain ×2.0
2. Layer-aware constraints (no cross-layer penalties)
3. Object-Aware Disambiguation - "weight clipping" → G06N
4. Functional boosting
5. **Verb Filter** (NEW) - verb-based scoring override
6. **Keyword Gravity** (NEW) - ambiguous noun weight reduction
7. Hierarchy Priority (Level 1-4 within domain)
8. Contribution Filter

**Output:** Adjusted candidates with rules log showing before/after scores

---

### `cpc_cross_domain_validator.py` - Phase 3.6
**Purpose:** Cross-Domain Validation Layer - prevents domain collapse

**Key Class:** `CrossDomainValidator`

**NEW: Contextual Entity Consistency Table (Prompt 3 Fix):**
- Multi-layer pattern validation
- Key patterns:
  - ("vehicle state", "database query") → anchor G06F/B60W, block B60Q/H02
  - ("llm", "speech") → anchor G10L/G06N
  - ("window", "controlling") → anchor B60W, block B60Q/H02
  - ("vehicle", "database") → anchor G06F16/B60W, block B60Q

**NEW: Layer-based Validation (Prompt 3 Fix):**
- Validates candidates against layer-specific requirements
- Each layer requires specific CPC families and signal counts
- Marks candidates as UNVERIFIED if <2 domain signals

**Steps:**
1. Domain Anchor Check - verify family has supporting signals
2. Anti-Domain Collapse - block families without required context
3. Contextual Entity Consistency
4. **Contextual Entity Table** (NEW) - multi-layer pattern rules
5. **Layer-based Validation** (NEW) - layer-specific requirements
6. Final Family Lock - requires ≥2 independent signals

---

### `cpc_hypothesis_consolidation.py` - Phase 4
**Purpose:** Consolidates Phase 3 candidates into structured hypotheses

**Key Class:** `CPCHypothesisConsolidator`

**Steps:**
1. Normalize codes to family level (G06N3/063 → G06N)
2. Cluster by family + title similarity
3. Compute cluster strength: `mean(candidate_scores) × coherence × log(size)`
4. Rank clusters by normalized score
5. Build max 2 hypotheses (PRIMARY + optional SECONDARY)

**Constraints:**
- No tertiary role (only PRIMARY / SECONDARY)
- Discard threshold: score < 0.3
- Secondary only if gap < 50%

---

### `cpc_hypothesis_resolver.py` - Phase 5
**Purpose:** Deterministic resolver that selects best hypothesis

**Key Class:** `CPCHypothesisResolver`

**Scoring Formula:**
```
final_score = 0.5 × phase4_score + 0.3 × functional_alignment + 0.2 × technical_coverage
```

**Hard Constraints:**
- Must choose ONLY from Phase 4 hypotheses
- Must NOT generate new CPC codes
- Exactly 1 primary
- Max 1 secondary (only if gap < 0.25)
- LLM used ONLY for tie-breaking (not for classification)

---

### `cpc_role_labeling.py` - Phase 8 + 8.5
**Purpose:** Transform flat CPC classification into a structured reasoning graph with role assignments

**Key Class:** `CPCRoleLabeling`

**How Phase 8 Works (The Logic):**
1. **Contribution Mapping**: Looks at "Importance Score" and "Technical Object" from Phase 1. If a code matches the "Core Technical Function," it is labeled CORE.
2. **Parent-Child Linking**: Uses the CPC hierarchy to identify broader classes for the LEGAL_COVERAGE layer.
3. **Narrative Generation**: Uses the titles of CPC codes to write a human-readable summary.

**Phase 8.5 - LLM Technical Reasoning (Optional):**
- Uses LLM to generate examiner-style reasoning for each CPC code
- System prompt: "Act as a Patent Examiner"
- Generates 3-4 sentence executive summary explaining how CORE interacts with SUPPORT within CONTEXT
- Adds one-sentence reasoning for each code linking it to patent description

**3-Layer Model:**
- **Layer 1 (CORE)**: 1-2 CPCs - highest technical contribution, strongest novelty
- **Layer 2 (SUPPORT)**: 1-2 CPCs - enabling technology, infrastructure
- **Layer 2 (CONTEXT)**: 1-2 CPCs - system environment, background domain
- **Layer 3 (LEGAL_COVERAGE)**: 1-2 CPCs - broader parent classes, fallback safety nets

**Why This is Valuable:**
- **For Patent Attorneys**: See the "Role" and immediately understand the AI's logic - no guessing
- **For Quality Control**: Hardware codes (H02, B60Q) in CORE layer = immediate red flag
- **For Prior Art Search**: Separate Context from Core for more targeted searches

**Output:**
```python
{
    "layer1_core": [{"symbol": "G06N3/045", "role": "CORE", ...}],
    "layer2_support": [{"symbol": "G06N3/063", "role": "SUPPORT", ...}],
    "layer2_context": [...],
    "layer3_coverage": [{"symbol": "G06N3/", "role": "LEGAL_COVERAGE", ...}],
    "role_summary": "CORE: G06N3/045 | SUPPORT: G06N3/063 | COVERAGE: G06N3/",
    "reasoning_graph": {...},
    "phase85_executive_summary": "LLM-generated 3-4 sentence summary...",
    "phase85_code_reasoning": [{"symbol": "G10L15/22", "reasoning": "..."}],
    "detailed_report": "Formatted report in text format",
}
```

**Formatted Report (in API response):**
```markdown
# CPC Classification & Technical Reasoning Report

## 1. Primary Classification (The Core Invention)
| CPC Symbol | Role | Technical Definition |
|------------|------|---------------------|

## 2. Supporting & Enabling Technology
...

## 3. Application Context
...

## 4. Technical Reasoning Graph (Executive Summary)
...

## 5. Legal Coverage (Search Safety Net)
...
```

---

## Support Modules

### `cpc_bm25_index.py`
**Purpose:** BM25 indexing for fast candidate retrieval

**Key Functions:**
- `build_or_load_bm25_index()` - Build or load BM25 index from CPC XML
- Fast sparse retrieval using term frequency
- Complements dense semantic search

**Integration:** Used in `knowledge_graph.py` for hybrid retrieval (BM25 + cross-encoder)

---

### `cpc_cross_encoder.py`
**Purpose:** Cross-encoder reranker for precise candidate ranking

**Key Functions:**
- Rerank top-K candidates from BM25/bi-encoder
- Token-level interaction between patent text and CPC descriptions
- More accurate than bi-encoder for final ranking

**Usage:** Applied to top-20 candidates from fast retriever due to computational cost

---

### `knowledge_graph.py`
**Purpose:** Knowledge graph with semantic embeddings for Phase 2A routing

**Key Class:** `CPCKnowledgeGraph`

**Features:**
- Builds NetworkX graph from CPC XML/JSON files
- Generates sentence-BERT embeddings (`all-mpnet-base-v2`, 110MB)
- Semantic search: cosine similarity between patent text and CPC titles
- Graph query: find classes covering extracted concepts
- Hybrid scoring: semantic (60%) + graph (40%)
- **NEW:** BM25 + cross-encoder integration for hybrid retrieval

**Cache Files:**
- `cpc_graph.pkl` - NetworkX graph structure
- `cpc_embeddings.npz` - Embedding vectors
- `cpc_graph_meta.json` - Build metadata and source hash

**Build Time:**
- First run: 15-25 minutes (GPU), 18+ hours (CPU)
- Subsequent runs: 5 seconds (load from cache)

---

### `prompts.py`
**Purpose:** LLM prompts for all pipeline phases

**Updated Prompts:**
- `phase1_prompt()` - Phase 1 semantic extraction with:
  - Primary domain detection (Step 7b)
  - Contextual term disambiguation (Step 7c)
  - Enhanced domain signals with CPC family and role
  - Negative signals with penalize_family
- `validation_prompt_single()` - Phase 6 single-candidate validation
- `reconciliation_prompt()` - Phase 6 per-claim reconciliation
- `consistency_check_prompt()` - Phase 7 final check with domain consistency

**Key Update:** Phase 1 now explicitly detects primary_domain and disambiguates terms to prevent cross-domain leakage.

---

### `cpc_chroma_store.py`
**Purpose:** ChromaDB vector database integration (optional)

**Features:**
- Store CPC subgroup embeddings in ChromaDB
- Vector similarity search alternative
- Persistent storage in `resources/chroma_db/`

**Status:** Optional enhancement, not used in main pipeline.

---

### `cpc_fast_index.py`
**Purpose:** Fast indexing for CPC lookup

**Features:**
- Inverted index for quick term-to-CPC mapping
- Optimized for large-scale searches
- Complements the knowledge graph approach

---

### `analysing_cpc.py`
**Purpose:** CPC analysis utilities

**Functions:**
- Score calculation and normalization
- Statistical analysis of classification results
- Helper functions for term matching

---

### `utils.py`
**Purpose:** General utility functions

**Functions:**
- Text normalization and tokenization
- JSON parsing with error handling
- File I/O helpers

---

## Test Suite

### Core Tests

| Test File | Purpose | Tests |
|-----------|---------|-------|
| `test_cross_domain_fix.py` | Cross-domain misclassification fix | 4 tests |
| `test_generalized_domains.py` | Multi-domain routing (mechanical, telecom) | 2 tests |
| `test_phase35.py` | Phase 3.5 decision tree constraints | 4 tests |
| `test_hierarchy_priority.py` | Intra-domain hierarchy (G06N, G06T, H04L) | 4 tests |
| `test_cross_domain_validation.py` | Phase 3.6 cross-domain validation | 4 tests |
| `test_phase8_role_labeling.py` | Phase 8 role assignment | 2 tests |

### Additional Tests
- `test_phase4.py` - Phase 4 hypothesis consolidation
- `test_phase5.py` - Phase 5 deterministic resolution
- `test_modality_routing.py` - Modality detection (vision vs AI)
- `test_hybrid_retrieval.py` - BM25 + cross-encoder integration
- `test_bm25_quick.py` - BM25 index building
- `run_phase2_simulation.py` - Phase 2 simulation runner
- `run_full_pipeline.py` - End-to-end pipeline test
- `run_pipeline_fast.py` - Fast pipeline runner

**Total:** 20 test files, 25+ test cases

---

## Data Files

### `resources/cpc_scheme_2026/`
**Purpose:** Source data - CPC classification scheme files

**Contents:**
- `cpc-scheme-*.xml` - Original EPO CPC scheme XML files (~690 files, 239MB)
- `cpc-cache-*.json` - Preprocessed JSON cache files (parsed from XML)

### `resources/cpc_graph.pkl`
**Purpose:** Serialized NetworkX knowledge graph (~50-100 MB)

### `resources/cpc_embeddings.npz`
**Purpose:** Compressed embedding vectors (~300-500 MB, 250K × 768 dimensions)

### `resources/cpc_graph_meta.json`
**Purpose:** Graph metadata (source hash, build date, node/edge counts)

### `resources/ipc_cpc_hints.txt`
**Purpose:** Hints for IPC to CPC classification mapping

---

## Configuration

### `.env` File (not in repo, create locally)
```bash
# LLM Model
LLM_MODEL=phi4-reasoning:plus

# Knowledge Graph
SKIP_KG=0              # Set to 1 to disable KG
SKIP_KG_REBUILD=1      # Skip hash check on startup
KG_SECTIONS=G,H        # Build only specific sections

# EPO API (optional)
EPO_API_KEY=your_key_here
```

---

## Usage Flow (Updated)

```
1. Patent text submitted to /classify endpoint
   |
2. Phase 1: LLM extracts semantic understanding (NO CPC codes)
   - technical_object, core_function, domain_signals, terms
   - primary_domain, disambiguated_terms, negative_signals
   |
3. Phase 1.5: Invention role classification (CORE_TECH/SYSTEM/APPLICATION/SUPPORT)
   |
4. TCR: Technical Weight Analysis (computational vs physical)
   |
5. Phase 2A: Layer Decomposition — multi-layer CPC mapping
   - Only technical layers used for anchor extraction (excludes application)
   |
6. Phase 2B: XML expansion restricted to technical-layer 4‑char prefixes
   - Pre-filtered to only existing XML files on disk
   - Per-class expansion counts logged
   |
7. Phase 2C: Hybrid Scoring (TF-IDF bigrams + Semantic Embeddings)
   - Scores ALL expanded candidates, no truncation
   |
8. Phase 2D: Subclass Anchor Filter
   - Keeps only candidates whose 4-char prefix matches technical layer anchors
   - Find-Until-Full: progressive expansion [500→1000→all] until ≥20 survivors
   |
9. Phase 3: Rank top 20 survivors
   |
10. Phase 3.5: Decision Tree — domain dominance, disambiguation, canonical sorting
    - Standard codes (PRIMARY_STANDARD) before indexing codes (SECONDARY_INDEXING)
    - Quota guardrail ensures ≥2 standard codes in top 5
    |
11. Phase 3.6: Cross-Domain Validation
    - Domain anchor check, anti-collapse, entity consistency
    |
12. Phase 4: Hypothesis Consolidation + Interpretation
    - Clusters into max 2 hypotheses (PRIMARY + optional SECONDARY)
    - Human-readable health analysis (support weight + coherence)
    |
13. Phase 5: Deterministic Resolution + Tri-Pillar FACETS
    - Selects primary hypothesis
    - Back-scans Phase 2C raw candidates for G06F, G06N, G05B champions
    - Output: Premier code + Facets (Goal / Method / Context)
    |
14. Phase 8: Executive Classification Report
    - At-a-Glance card with Premier code + confidence badge
    - Tech Stack table from Phase 5 facets
    - Professional Justification (LLM narrative)
    - Suggested Indexing Codes (copy-paste ready)
    - Download button for Markdown export
```

---

## Pipeline Architecture Principles

1. **Separation of Concerns**
   - Phase 1: Semantic understanding only
   - Phase 2-3.6: CPC classification and validation
   - Phase 8: Structured reasoning graph

2. **Deterministic Scoring**
   - Phase 3.5: Rule-based constraints (domain > keyword)
   - Phase 3.6: Cross-domain validation (context > abstraction)
   - Phase 4: Normalized cluster scores
   - Phase 5: Weighted formula (0.5×phase4 + 0.3×func_align + 0.2×tech_cov)

3. **Constrained LLM Usage**
   - LLM only in Phase 1 (extraction) and Phase 6-7 (optional reconciliation)
   - No LLM for CPC classification or code generation

4. **Domain-First, Not Abstraction-First**
   - Technical domain (G06N/G06F/G10L/H04L) determines classification
   - Abstract contribution types (A-F hierarchy) removed - caused domain collapse
   - Contextual entity consistency prevents keyword traps

5. **Multi-Layer Output**
   - Layer 1 (CORE): 1-2 CPCs representing invention essence
   - Layer 2 (SUPPORT/CONTEXT): 1-3 CPCs explaining system environment
   - Layer 3 (LEGAL_COVERAGE): Optional broader classes for legal safety

6. **Explanation Graph + Quality Check**
   - Phase 8 produces human-readable summary first
   - Hardware codes (H02, B60Q) in CORE layer = immediate red flag
   - Users understand classification logic without reading all phases

---

## File Size Summary

| File/Directory | Size | Type |
|----------------|------|------|
| `cpc_scheme_2026/` | ~239 MB | Source XML + JSON |
| `cpc_graph.pkl` | ~50-100 MB | Generated cache |
| `cpc_embeddings.npz` | ~300-500 MB | Generated cache |
| `chroma_db/` | ~10-50 MB | Vector DB |
| **Total generated** | **~400-650 MB** | Cache files |

---

## Key Dependencies

```
fastapi          - Web framework
uvicorn          - ASGI server
ollama           - LLM client (local)
sentence-transformers  - Embedding model (110MB)
networkx         - Graph structure
numpy            - Numerical operations
scikit-learn     - Cosine similarity
lxml             - XML parsing (for CPC scheme)
rank-bm25        - BM25 indexing (NEW)
transformers     - Cross-encoder models (NEW)
```

---

## Notes

- **Cache files** (`*.pkl`, `*.npz`) are in `.gitignore` - do not commit
- **Tests** are kept in repo - they document system behavior
- **First startup** takes 15-25 min if building knowledge graph
- **Subsequent startups** load cache in ~5 seconds
- **SKIP_KG=1** for instant startup (domain-signal fallback mode)
- **Phase 3.6** replaces deprecated A-F hierarchy - prevents domain collapse
- **Phase 8** produces structured CPC explanation graph instead of flat output

---

## Recent Changes (2026-05-10)

### Phase 1.5 + Layer Decomposition + Phase 8 Enhancement
- `cpc_role_classifier.py` - Phase 1.5: Invention role classification (CORE_TECH/SYSTEM/APPLICATION/SUPPORT)
- `cpc_layer_decomposer.py` - Phase 2A (REPLACEMENT): Multi-layer CPC decomposition
- `cpc_role_labeling.py` - Phase 8/8.5 enhanced with Explanation Graph + LLM Reasoning + Formatted Report

### Semantic Drift Fixes (Prompt 1-4)
- **Prompt 1** (`cpc_layer_decomposer.py`): Mandatory Domain Anchor + AI+Auto co-occurrence rules
- **Prompt 2** (`cpc_decision_tree.py`): Functional Verb Filter + Keyword Gravity
- **Prompt 3** (`cpc_cross_domain_validator.py`): Contextual Entity Table + Layer-based Validation
- **Prompt 4** (`search_cpc.py`): Phase 7.5 Feedback Loop (re-filter using Phase 7 recommendations)

### Examiner-Ready Report (Phase 8.5)
- `cpc_role_labeling.py`: Added LLM-based technical reasoning generation
- `search_cpc.py`: Added `formatted_report` field in API response
- Phase 3: Increased from Top 10 to Top 20 candidates
- Phase 8.5: LLM generates 3-4 sentence executive summary explaining component orchestration
- Formatted report includes tables for CORE/SUPPORT/CONTEXT and reasoning for each code

### Added
- `cpc_decision_tree.py` - Phase 3.5 decision tree constraints + Verb Filter + Keyword Gravity
- `cpc_cross_domain_validator.py` - Phase 3.6 entity table + layer validation
- `cpc_role_labeling.py` - Phase 8/8.5 role labeling, explanation graph, LLM reasoning
- `cpc_bm25_index.py` - BM25 indexing for hybrid retrieval
- `cpc_cross_encoder.py` - Cross-encoder reranker
- `test_phase15_role_classifier.py` - Phase 1.5 tests

### Modified
- `cpc_family_router.py` - Legacy mode (layer decomposer is primary now)
- `prompts.py` - Added primary domain detection, term disambiguation
- `search_cpc.py` - Integrated all phases (1, 1.5, 2A, 3.5, 3.6, 7, 7.5, 8, 8.5), Phase 3 now Top 20, added formatted_report
- `cpc_decision_tree.py` - Added hierarchy priority, contribution filter, layer-aware mode, verb filter
- `cpc_cross_domain_validator.py` - Added contextual entity table, layer-based validation
- `cpc_role_labeling.py` - Added LLM reasoning (Phase 8.5), formatted report generation

### Deprecated
- `cpc_family_router.py` (replaced by layer decomposer)
- `cpc_hierarchy_engine.py` - A-F taxonomy caused domain collapse, replaced by cross-domain validator

---

## Recent Changes (2026-05-11)

### Phase 2C: Hybrid Scoring (TF-IDF + Semantic Embeddings)

- **`search_cpc.py`**: Added `_compute_semantic_scores()` method to `CPCClassifier`
  - Uses KG's `all-mpnet-base-v2` sentence-transformer model to encode patent text (technical_object + core_function)
  - Computes cosine similarity between patent embedding and each CPC candidate's precomputed embedding
  - Normalizes semantic scores to [0, 1] scale
  - Hybrid formula: `Final_Score = 0.4 × TF-IDF_norm + 0.6 × Semantic_Similarity`
  - Gracefully falls back to TF-IDF-only if KG unavailable
  - Applied to both `classify()` inline scoring and `classify_from_phase1()` path

### Phase 2C: Bigram (n-gram) Upgrade

- **`search_cpc.py`**: Added `_tokenize_with_bigrams()` and `_make_term_bigrams()` functions
  - CPC contexts now produce both unigrams AND bigrams (e.g., `"system_prompt"`, `"user_prompt"`, `"program_code"`)
  - Bigrams anchored in document frequency for IDF computation
  - Term matching loop generates bigrams from multi-word patent terms and checks against CPC bigrams
  - Matched technical bigrams get ×6 IDF-weighted bonus — anchors ambiguous words to domain-specific phrases
  - Prevents "prompt" from being confused with generic UI prompts (G06F) and anchors it to AI/NLP domains (G06N, G06F40)

### Phase 2C: Score Precision Fix

- **`search_cpc.py`**: Fixed bug where hybrid-scored tuples still held raw TF-IDF score
  - Previously, only re-sorted by hybrid key but normalization/display used TF-IDF — causing 0.000 margins
  - Now replaces tuple score with actual computed hybrid value (`round(hybrid, 6)`)
  - Score rounding increased: `round(normalized_score, 4)` → `round(normalized_score, 6)`
  - Score margin rounding increased: `round(score_margin, 4)` → `round(score_margin, 6)`
- **`streamlit_app.py`**: Phase 2C display updated — `{margin:.3f}` → `{margin:.6f}`

### Phase 2C: Candidate Count Expansion (Top 10 → Top 20)

- **`search_cpc.py`**: Both `classify()` and `classify_from_phase1()` changed `scored[:10]` → `scored[:20]`
  - Phase 2C now outputs top 20 candidates (was 10)
  - Gives Phase 3.5 (Decision Tree) and Phase 3.6 (Cross-Domain Validator) more diverse candidates
  - Support/Context layer codes with lower scores can survive initial TF-IDF cut
- **`cpc_hypothesis_consolidation.py`**: `max_candidates` default changed from 10 → 20
  - Phase 4 consolidation now uses top 20 candidates for hypothesis clustering

### Streamlit: Phase-by-Phase Display

- **`streamlit_app.py`**: Complete restructure for phase-controlled output
  - Added `PHASES` list (14 phases: 1, 1.5, TCR, 2A, 2B, 2C, 3, 3.5, 3.6, 4, 5, 6, 7, 8)
  - Phase selector dropdown + Prev/Next navigation buttons at top
  - Result cached in `st.session_state` — switching phases does NOT re-run pipeline
  - Each phase section wrapped in `if/elif` conditional — only selected phase renders
  - Default view: Phase 1
  - Phase 2C caption updated to reflect hybrid scoring (TF-IDF + n-grams + embeddings)

### Modified
- `search_cpc.py` — Added `import numpy`, `_tokenize_with_bigrams()`, `_make_term_bigrams()`, `_compute_semantic_scores()`, hybrid scoring (both paths), bigram matching (both paths), precision fix, K=20 expansion
- `cpc_hypothesis_consolidation.py` — `max_candidates: 10 → 20`
- `streamlit_app.py` — Phase-by-phase display, Phase 2C caption update, score margin precision

---

## Recent Changes (2026-05-11 — Continued)

### Phase 2D: Subclass Structural Anchor (NEW)

- **`cpc_phase2d_anchor.py`**: New module — `Phase2DSubclassAnchor` class
  - Extracts 4-character CPC subclass prefixes (e.g., G06F, G10L, G06N) from Phase 2A technical layers (pure_software, data_reasoning, interaction, control)
  - Strictly excludes application-layer codes and 3-digit broad classes
  - Filters Phase 2C candidates: keeps only those whose prefix matches the anchor set
  - Excludes non-technical families: G06Q, G06C, G07F, G07G, G09F, G09B, A63F
  - Falls back to family router output if layer decomposition unavailable (uses `_FAMILY_TO_SUBCLASS` mapping)
  - Maintains hybrid scores (0.4×TF-IDF + 0.6×Semantic) from Phase 2C
  - Outputs purified top 20 candidates for Phase 3
- **`search_cpc.py`**: Phase 2D integrated between Phase 2C and Phase 3 in both `classify()` and `classify_from_phase1()` paths
- **`streamlit_app.py`**: Added Phase 2D display section — anchor subclasses count, kept/discarded metrics, discard log expander

### Phase 2B/2C: Combined Classes Pollution Fix

- **`search_cpc.py`**: Fixed critical bug — `merge_layers_to_family_list()` returned ALL layer symbols including application-layer families (G06Q, B60W, B60R, A61, etc.)
  - When `graph_classes` was empty (KG not loaded), `combined_classes` fell back to this polluted list
  - `parse_file("G06Q")` found the G06Q XML file and flooded the candidate pool with business-method codes
  - Fix: Extract `allowed_roots` from technical layers only (pure_software, data_reasoning, interaction, control) using `[A-Z]\d{2}[A-Z]` regex
  - `combined_classes` fallback now uses `allowed_roots` (clean prefixes) instead of raw `top_cpc_families`
  - XML expansion restricted to technical subclasses only (G06F, G10L, G06N, G05B)

### Phase 2B: Pre-filtering + Per-Class Expansion Logging

- **`search_cpc.py`**: Added pre-filtering of `combined_classes` to only include codes with actual XML files on disk
  - Checks `cpc-scheme-{code}.xml` existence before attempting `parse_file()`
  - Codes without XML files (e.g., G05D, G06K) are skipped with a warning
  - `allowed_roots` filtered to match valid combined classes
  - Per-class expansion counts collected: `phase2b_expansion_counts` dict {prefix: subgroup_count}
  - Log example: `Phase 2B: Expanded 847 subgroups across 4 families: {G05B: 45, G06F: 320, G06N: 332, G10L: 150}`
- **`streamlit_app.py`**: Phase 2B display updated — shows per-family expansion breakdown as metrics, skipped classes warning

### Phase 2C: Deep Retrieval (K=20 → 100) + Find-Until-Full

- **`search_cpc.py`**: Both `classify()` and `classify_from_phase1()` changed `scored[:20]` → `scored[:100]`
  - Phase 2C now scores top 100 candidates (was 20)
  - Phase 2D `max_result` changed from 50 → 100
  - Find-Until-Full progressive expansion loop: [500, 1000, all]
    - Normalize ALL scored candidates into `all_candidates`
    - Pass top 500 → Phase 2D → ≥20 survivors? STOP
    - Pass top 1000 → Phase 2D → ≥20 survivors? STOP
    - Pass ALL → Phase 2D → whatever survives
  - Log: `"Deep Search required. Scanned 500 to find X valid technical anchors."` or `"Find-Until-Full: Scanned 500 to find 20 valid technical anchors. ✓ Quota met."`
  - `find_until_full_log` stored in result dict for Streamlit display
- **`streamlit_app.py`**: Phase 2C display updated — shows "Total Scored" metric + Find-Until-Full expansion log

### Phase 3.5: Canonical Sorting — Standard Codes Before Indexing Codes

- **`cpc_decision_tree.py`**: Added `_is_indexing_code(symbol)` helper function
  - Regex pattern: `^[A-Z]\d{2}[A-Z](2\d{3})` — detects 2xxx-series indexing codes (e.g., G05B2219/..., G06F2221/...)
  - Standard codes: G06F 8/xx, G05B 19/xx → `PRIMARY_STANDARD`
  - Indexing codes: G05B 2219/..., G06F 2110/... → `SECONDARY_INDEXING`
  - Each candidate tagged with `code_type` field
- **Step 9: Canonical "Noun-First" Sorting**:
  - Level 1: type — PRIMARY_STANDARD before SECONDARY_INDEXING
  - Level 2: score — descending within each type group
  - Reserves top 20 for quota enforcement
- **Step 10: Quota Guardrail**: If top 5 are ALL SECONDARY_INDEXING, reaches into positions 6-20 to promote at least 2 PRIMARY_STANDARD codes into top 5
- **`streamlit_app.py`**: Phase 3.5 candidates split into "Core Invention (Standard Codes)" and "Technical Details (Indexing Codes)" sections

### Phase 4: Human-Readable Interpretation Engine

- **`cpc_hypothesis_consolidation.py`**: Added `_generate_interpretation()` method to `CPCHypothesisConsolidator`
  - Analyzes Support Weight and Coherence to produce human-readable insights
  - **Support Weight** thresholds: >50% → "Clean" patent, <15% → "Messy/Hybrid"
  - **Coherence** thresholds: >0.8 → "Family Neighbors" (stable niche), <0.6 → "Hallucinating connection"
  - **Actionable Advice**: Both high → "Proceed to Phase 5", Both low → "Increase Phase 2C retrieval depth"
  - Result stored in `phase4_interpretation` field
- **`streamlit_app.py`**: Phase 4 now shows "[INSIGHT] Classification Health Analysis" block with icon-badged insights and actionable advice

### Phase 5: Tri-Pillar Classification + FACETS Display

- **`cpc_hypothesis_resolver.py`**: Major upgrade — Tri-Pillar Resolution
  - Added `PILLAR_DEFINITIONS` with three functional roles:
    - `pillar1_goal`: Primary Function (G06F, G06Q) — core technical result
    - `pillar2_method`: Methodology (G06N) — AI/ML implementation
    - `pillar3_context`: Application Domain (G05B, B60W, A61B, H02J) — industrial environment
  - `resolve()` now accepts `all_raw_candidates` parameter for back-scanning
  - `_resolve_pillars()`: Finds highest-scoring champion per pillar from Phase 2C raw candidates
  - `_find_champion_in_pool()`: Static method — scans candidates by family prefix
  - Each pillar result includes `source` field: `"phase2c_back_scan"` or `"not_found"`
- **`search_cpc.py`**: Passes `all_candidates` to resolver: `resolver.resolve(phase4_result, phase1, all_candidates)`
- **Phase 5 UI Refinement** (`streamlit_app.py`):
  - Premier CPC Classification shown first (renamed caption: "Phase 7 Logic Reconciliation")
  - Renamed `[PILLARS]` → `[FACETS] Cross-Domain Classifications`
  - Primary Facet shown as top header; Methodological and Application facets grouped under collapsible "Supporting Technical Facets"
  - Tooltips added: "Core technical result", "AI/ML implementation strategy", "Target hardware/industrial environment"
  - `[INFO] No secondary family` → `"Classification Health: Primary focus confirmed. High signal separation detected"`

### Pipeline Optimization: Phase 6 & 7 Decommissioned

- **`search_cpc.py`**: Removed Phase 6 (Per-Claim Classification) processing entirely
  - Removed `reconciled_claims` variable and LLM reconciliation call
  - Removed Phase 7/7.5 labels — consistency logic still runs internally for Premier/Phase 8
  - Removed `result["phase7"]` and `result["per_claim"]` from output
  - Premier assignment moved before result dict construction for proper data flow
- **`streamlit_app.py`**: Phases 6 and 7 removed from PHASES list and display blocks
  - Phase 7 raw JSON debug expander removed
  - Pipeline now: Phase 1 → 1.5 → TCR → 2A → 2B → 2C → 2D → 3 → 3.5 → 3.6 → 4 → 5 → 8 (13 phases)

### Phase 8: Executive Classification Report

- **`search_cpc.py`**: `_build_formatted_report()` completely rewritten
  - Now accepts `pillars` (Phase 5 facets) and `premier` (main recommendation) parameters
  - Generates structured Markdown report with 5 sections:
    1. **Main Recommendation** — Premier code + confidence badge
    2. **🛠 Technical Breakdown** — Facets table (Primary Goal / AI Methodology / Domain Context)
    3. **💡 Professional Justification** — LLM summary or fallback from facets
    4. **📋 Suggested Indexing Codes** — Non-pillar codes from Core/Support/Context/Coverage layers
    5. **📊 Supporting Classification Details** — Condensed layer breakdown
  - Added `_shorten()` static method for text truncation
- **`streamlit_app.py`**: Phase 8 redesigned as Executive Card layout
  - **At-a-Glance Card**: Premier code in large bold font with confidence badge (✅ High / 🔶 Medium / ⚠️ Low)
  - **🛠 Technical Breakdown**: DataFrame table showing facets by role
  - **💡 Professional Justification**: LLM-generated reasoning or fallback
  - **📋 Suggested Indexing Codes** (collapsible): Copy-paste ready CPC references
  - **📊 Full Classification Report** (collapsible): Raw Markdown report
  - **📊 Supporting Classification Details** (collapsible): Core/Support/Context/Coverage breakdown
  - **📥 Download Executive Report (Markdown)**: Download button using `st.download_button()`

### Added
- `cpc_phase2d_anchor.py` — Phase 2D: Subclass Structural Anchor filter
- `_tokenize_with_bigrams()` and `_make_term_bigrams()` functions in `search_cpc.py`
- `_compute_semantic_scores()` method in `CPCClassifier`
- `_is_indexing_code()` helper in `cpc_decision_tree.py`
- `_generate_interpretation()` method in `CPCHypothesisConsolidator`
- `_resolve_pillars()` and `_find_champion_in_pool()` methods in `CPCHypothesisResolver`
- `PILLAR_DEFINITIONS` constant in `cpc_hypothesis_resolver.py`
- `_shorten()` static method in `CPCClassifier`

### Modified
- `search_cpc.py` — Hybrid scoring (0.4 TF-IDF + 0.6 semantic), bigrams, precision fix, K=20→100, combined_classes pollution fix, pre-filtering, per-class expansion counts, Find-Until-Full loop, Phase 2D integration, Phase 6/7 removal, `_build_formatted_report()` rewrite, premier data flow
- `cpc_hypothesis_consolidation.py` — `max_candidates: 10 → 20`, added `_generate_interpretation()`
- `cpc_hypothesis_resolver.py` — Tri-Pillar resolution, pillar definitions, `all_raw_candidates` parameter, facet labels
- `cpc_decision_tree.py` — `_is_indexing_code()`, canonical sorting, quota guardrail, code_type tagging
- `streamlit_app.py` — Phase-by-phase display, Phase 2B expansion breakdown, Phase 2C Find-Until-Full log, Phase 2D anchor filter, Phase 3.5 standard/indexing split, Phase 4 interpretation engine, Phase 5 FACETS display + Premier first, Phase 6/7 removal, Phase 8 Executive Card with download button
