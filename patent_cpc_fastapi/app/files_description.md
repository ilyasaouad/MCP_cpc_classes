# Patent CPC FastAPI - File Descriptions



# Design principle (critical)

# CPC classification should NOT minimize error
# It should structure technical interpretation

# The goal is:

# Not “correct vs incorrect”
# But “what role does each CPC code play?”

Example for minimizing error:

# CPC is multi-labeled
# If a patent touches 3 domains, you must return 3 codes.
# If you only return 1 code, that is a “hit” for that code,
# but technically you failed to classify the other two aspects.

# So “correct” = completeness.
# Not = “highest probability”.
 
Is not like classic classification task, like true/false for image of cat or dog, where we try to minimize error, but it is an interpretation task.

The interpreter should structure technical interpretation in a consisten way.  

# The interpreter should use probabilistic reasoning to estimate the probability of each CPC code, and not deterministic reasoning.  

 

## Overview

This directory contains the FastAPI backend for patent CPC (Cooperative Patent Classification) classification. The system uses a multi-phase pipeline separating **semantic understanding** (Phase 1) from **CPC classification** (Phases 2-5). The architecture is designed for production-grade CPC classification with deterministic scoring and constrained LLM usage.

---

## Directory Structure

```
app/
├── main.py                          # FastAPI application entry point
├── precompute_cpc_embeddings.py     # Script to precompute CPC embeddings
├── search_core/                     # LLM client module
│   ├── __init__.py
│   └── ollama_client.py            # Ollama API client wrapper
└── cpc_classification/              # Core classification pipeline
    ├── __init__.py
    ├── analysing_cpc.py            # CPC analysis utilities
    ├── cpc_chroma_store.py         # ChromaDB vector store integration
    ├── cpc_family_router.py        # Phase 2A: CPC family routing (NEW)
    ├── cpc_fast_index.py           # Fast indexing for CPC lookup
    ├── cpc_hypothesis_consolidation.py  # Phase 4: Cluster hypotheses (NEW)
    ├── cpc_hypothesis_resolver.py       # Phase 5: Deterministic resolver (NEW)
    ├── cpc_xml_parser.py           # XML parser for CPC scheme files
    ├── epo_client.py              # EPO API client (optional/experimental)
    ├── extracting_cpc.py          # Phase 1: Semantic extraction only
    ├── knowledge_graph.py         # Knowledge graph with semantic embeddings
    ├── prompts.py                 # LLM prompts for all phases
    ├── search_cpc.py              # Main pipeline orchestrator (Phases 1-5)
    ├── utils.py                   # Utility functions
    └── resources/                 # Data files and caches
        ├── chroma_db/             # ChromaDB vector database files
        ├── cpc_scheme_2026/       # CPC scheme XML files (source data)
        │   ├── cpc-scheme-*.xml   # Original CPC scheme XML files
        │   └── cpc-cache-*.json   # Parsed JSON cache files (preprocessed)
        ├── cpc_graph.pkl          # Knowledge graph cache (generated)
        ├── cpc_embeddings.npz     # Embedding vectors cache (generated)
        ├── cpc_graph_meta.json    # Graph metadata (generated)
        └── ipc_cpc_hints.txt      # IPC/CPC classification hints
```

---

## Core Files

### `main.py`
**Purpose:** FastAPI application entry point and HTTP server

**Key Functions:**
- Initializes the FastAPI application
- Loads knowledge graph at startup (with caching and auto-rebuild)
- Provides `/classify` endpoint for patent classification
- Supports environment variables: `SKIP_KG`, `KG_SECTIONS`, `LLM_MODEL`

**Environment Variables:**
- `SKIP_KG=1` - Disable knowledge graph for faster startup
- `KG_SECTIONS=G,H` - Build only specific CPC sections
- `LLM_MODEL` - LLM model name (default: `gpt-oss:120b-cloud`)

---

### `search_core/ollama_client.py`
**Purpose:** Wrapper for Ollama LLM API

**Key Functions:**
- `chat()` - Send chat messages to local Ollama server
- Handles model loading, temperature, max_tokens
- Connects to `http://localhost:11434` (Ollama default)

**Used by:**
- `extracting_cpc.py` (Phase 1 extraction)
- Phase 6-7 reconciliation (tie-breaking only)

---

## CPC Classification Pipeline

### `cpc_classification/search_cpc.py`
**Purpose:** Main pipeline orchestrator - coordinates all classification phases

**Pipeline Phases:**
1. **Phase 1** - Semantic extraction (terms, context, function) - **NO CPC codes**
2. **Phase 2A** - CPC Family Router: selects top 3 CPC families
3. **Phase 2B** - Restricted XML expansion (only within Phase 2A families)
4. **Phase 2C** - TF-IDF scoring with term matching
5. **Phase 3** - Rank top 10 candidates
6. **Phase 4** - Hypothesis consolidation: clusters into max 2 hypotheses
7. **Phase 5** - Deterministic resolution: selects exactly 1 primary

**Key Class:** `CPCClassifier`
- Accepts optional `knowledge_graph` parameter
- Falls back to domain-signal routing if KG unavailable
- Returns structured JSON with all phase results

**Architecture Principle:**
```
Phase 1 = Understand the invention (semantic only)
Phase 2 = Route to CPC families
Phase 3 = Score candidates
Phase 4 = Cluster into hypotheses
Phase 5 = Resolve to final selection
```

---

### `cpc_classification/extracting_cpc.py`
**Purpose:** Phase 1 - Extract semantic understanding from patent text

**Key Functions:**
- `extract()` - Main extraction method
- Term normalization and importance weighting
- Section-aware extraction (claims get 2x weight)

**Output:**
- `technical_object` - What the invention is
- `core_function` - What it does (action/operation)
- `system_context` - Industry/domain
- `domain_signals[]` - Technical domains with confidence
- `terms[]` with importance scores
- `negative_signals[]` for downstream filtering
- `classification_strategy` - system-first vs function-first vs hybrid

**Important:** Phase 1 does NOT output CPC classes, class_hypotheses, or cpc_classes. All CPC mapping is handled by Phase 2A.

---

### `cpc_classification/cpc_family_router.py`
**Purpose:** Phase 2A - Routes patent to top CPC families (3-character codes)

**Key Class:** `CPCFamilyRouter`

**Design:**
- **Purpose domains** (weight=1.2): computer vision, manufacturing, medical, etc.
- **Tool domains** (weight=0.35): LLM, neural network, machine learning
- **Function domains** (weight=0.6): rule-based, expert system, decision support

**Co-occurrence Rules:**
- Vision + AI → boost vision (G06V), penalize AI (G06N)
- NLP + AI → boost NLP (G06F), penalize AI
- Medical + AI → boost medical (A61), penalize AI

**Modality Detection:**
- Detects primary modality (vision, nlp, audio, mechanical)
- Applies boost/penalty based on term counts

**Output:**
```python
{
    "families": ["G06N", "G06F", "G06Q"],
    "primary": "G06N",
    "secondary": ["G06F", "G06Q"],
    "modality": "nlp",
    "source": "embedding" | "domain_signals"
}
```

---

### `cpc_classification/cpc_xml_parser.py`
**Purpose:** Parse CPC scheme XML files into usable data structures

**Key Functions:**
- `expand_classes()` - Expand CPC class codes into subgroups
- Supports `allowed_roots` parameter for Phase 2B filtering
- Parse XML hierarchy (parent-child relationships)
- Extract titles, symbols, and allocatable flags

**Input:** `cpc_scheme_2026/cpc-scheme-*.xml`
**Output:** List of subgroup dictionaries with symbol, title, level, parent_chain

---

### `cpc_classification/cpc_hypothesis_consolidation.py`
**Purpose:** Phase 4 - Consolidates Phase 3 candidates into structured hypotheses

**Key Class:** `CPCHypothesisConsolidator`

**Steps:**
1. Normalize codes to family level (G06N3/063 → G06N)
2. Cluster by family + title similarity
3. Compute cluster strength:
   ```
   cluster_score = mean(candidate_scores) × coherence × log(size)
   ```
4. Rank clusters by normalized score
5. Build max 2 hypotheses (PRIMARY + optional SECONDARY)

**Constraints:**
- No tertiary role (only PRIMARY / SECONDARY)
- Discard threshold: score < 0.3
- Secondary only if gap < 50%

**Output:**
```python
{
    "phase4_hypotheses": [
        {
            "family": "G06N",
            "role": "primary",
            "score": 6.03,           # raw sum
            "normalized_score": 1.0,  # mean × coherence × log(size)
            "mean_score": 0.86,
            "candidate_count": 7,
            "coherence": 0.83,
            "supporting_codes": [...],
            "reasoning": "Core invention domain: 7 candidates..."
        }
    ],
    "phase4_primary_family": "G06N",
    "phase4_support_weight": 0.64,
    "phase4_confidence": "high"
}
```

---

### `cpc_classification/cpc_hypothesis_resolver.py`
**Purpose:** Phase 5 - Deterministic resolver that selects best hypothesis

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

**Output:**
```python
{
    "primary": {
        "family": "G06N",
        "final_score": 0.91,
        "functional_alignment": 0.85,
        "technical_coverage": 0.78,
        "confidence": "high",
        "reasoning": "Most aligned with LLM weight quantization..."
    },
    "secondary": {  # optional
        "family": "G06F",
        "final_score": 0.72,
        "confidence": "medium"
    },
    "decision_logic": {
        "score_gap": 0.19,
        "secondary_accepted": true,
        "selection_method": "deterministic_scoring"
    }
}
```

---

### `cpc_classification/knowledge_graph.py`
**Purpose:** Knowledge graph with semantic embeddings for Phase 2A routing

**Key Class:** `CPCKnowledgeGraph`

**Features:**  # Disse kan forbedres betragteligt 
- Builds NetworkX graph from CPC XML/JSON files # kan brukert Nodj4 database 
- Generates sentence-BERT embeddings (`all-mpnet-base-v2`, 110MB) # Bedre embedding modeller finnes, f.eks. via OpenAI  og kan integreres i Nodj4 / LLM modelene 
- Semantic search: cosine similarity between patent text and CPC titles # kan brukert andre methode Cross-encoder reranker, BM25 to get candidates, then cosine to rerank. 
- Graph query: find classes covering extracted concepts
- Hybrid scoring: semantic (60%) + graph (40%)

# best alterantive er fine tunning av en spesiell embedding LLM til CPC klasser, 
# siden de er relativt få kan det gi best resultat og enklest for å opprettholde god nøyaktighet.
# krever dog en god del data for å få det skikkelig bra. ca 10 000 klasser med flere 
# eksempler fra patente.  (fra google) de hadde en oppgave som var lik denne, men da med 
# legemidler istedenfor CPC klasser og de fikk da en nøyaktighet på ca 95%. 

# Vi bruker isdent: vi bruker all-mpnet-base-v2, men for forbedre med 
BM25 (high recall)[TF-IDF] + all-mpnet-base-v2 embed så retrieve top-N (cosine) så cross-encoder rerank top-K.

# Eksempel — poenggiving, Best Matching 25 (BM25), som brukt f.ex i Elasticsearch:

Spørsmål: "wireless charging phone"

Dokumenter:
A: "Wireless charging for smartphones using induction coils."
B: "Phone battery life improvements and power management."
C: "Inductive charging pads for electric vehicles."

BM25-intuisjon:

A matcher alle søkeordene → høyest poengsum.
B matcher bare "phone" → lavere poengsum.
C matcher "charging" men annet domene → middels poengsum.

Enkel illustrativ pseudo-poeng (ikke nøyaktig formel):

A: 3 matchende termer × høy TF/IDF → poeng ≈ 5.2
C: 1 matchende term ("charging") men lavere IDF → poeng ≈ 2.1
B: 1 matchende term ("phone") vanlig term → poeng ≈ 1.4

# Bruk BM25 for å hente A og C (høy recall), og rerank deretter med semantiske embeddings om nødvendig.



# Example — Cross-encoder reranker for CPC classes on a knowledge graph

Query (Phase‑1 summary): 
# "autonomous drone navigation using visual SLAM for obstacle avoidance in indoor warehouses"
Candidates retrieved by BM25 / bi-encoder (titles & KG nodes):
A: "G05D1/02 — Navigation control systems"
B: "G05D1/00 — Control systems not otherwise provided for"
C: "G01C21/00 — Navigation; Navigational instruments"
D: "B64C39/00 — Aircraft control systems"

Cross-encoder input pairs (concatenate query + CPC title/node text) and example scores:

score(Query, A) = 0.88
score(Query, B) = 0.55
score(Query, C) = 0.63
score(Query, D) = 0.47
Final ranking (by score):
A: "G05D1/02 — Navigation control systems" (0.88)
C: "G01C21/00 — Navigation; Navigational instruments" (0.63)
B: "G05D1/00 — Control systems not otherwise provided for" (0.55)
D: "B64C39/00 — Aircraft control systems" (0.47)


Cross-encoder models token-level interaction between the Phase‑1 text (summery og problem og solution osv.) og CPC title/node description, so it better distinguishes domain-relevant CPC nodes (A) from broader or less relevant ones (B, D).
Use only on top-K candidates from the fast retriever (e.g., K=20) due to cost, time og kompleksitet.



**Cache Files:**
- `cpc_graph.pkl` - NetworkX graph structure # 
- `cpc_embeddings.npz` - Embedding vectors
- `cpc_graph_meta.json` - Build metadata and source hash

**Build Time:**
- First run: 15-25 minutes (depends on GPU), med CPU brukte mer 18 timer
- Subsequent runs: 5 seconds (load from cache)

---

### `cpc_classification/prompts.py`
**Purpose:** LLM prompts for all pipeline phases

**Key Functions:**
- `phase1_prompt()` - Phase 1 semantic extraction (NO CPC classes) # Fase 1 semantisk utrrekking (INGEN CPC-klasser)
- `validation_prompt_single()` - Phase 6 single-candidate validation (optional) # Fase 6 validering av enkeltkandidat
- `reconciliation_prompt()` - Phase 6 per-claim reconciliation #forsoning par krav
- `consistency_check_prompt()` - Phase 7 final check # slutter check

 


**Phase 1 Prompt Features:**
- Section-aware term extraction (background downweighted)
- Claim type analysis (method vs apparatus)
- Domain signal extraction (NOT CPC classes)
- Dynamic domain appropriateness checking
- Negative signal detection

**Important:** Phase 1 prompt explicitly instructs: "Do NOT assign CPC classes."

---

### `cpc_classification/epo_client.py`
**Purpose:** Client for EPO (European Patent Office) APIs (optional/experimental)

**Status:** Not used in main pipeline. Local CPC XML files + knowledge graph are preferred for accuracy and privacy.

**Key Functions:**
- Fetch CPC hierarchy from EPO Linked Open Data
- Retrieve patent classifications from Espacenet
- OPS (Open Patent Services) API integration

---

### `cpc_classification/cpc_chroma_store.py`
**Purpose:** ChromaDB vector database integration (optional)

**Features:**
- Store CPC subgroup embeddings in ChromaDB
- Vector similarity search alternative
- Persistent storage in `resources/chroma_db/`

**Status:** Optional enhancement, not used in main pipeline.

---

### `cpc_classification/cpc_fast_index.py`
**Purpose:** Fast indexing for CPC lookup

**Features:**
- Inverted index for quick term-to-CPC mapping
- Optimized for large-scale searches
- Complements the knowledge graph approach

---

### `cpc_classification/analysing_cpc.py`
**Purpose:** CPC analysis utilities

**Functions:**
- Score calculation and normalization
- Statistical analysis of classification results
- Helper functions for term matching

---

### `cpc_classification/utils.py`
**Purpose:** General utility functions

**Functions:**
- Text normalization and tokenization
- JSON parsing with error handling
- File I/O helpers

---

## Data Files

### `resources/cpc_scheme_2026/`
**Purpose:** Source data - CPC classification scheme files

**Contents:**
- `cpc-scheme-*.xml` - Original EPO CPC scheme XML files (~690 files, 239MB)
- `cpc-cache-*.json` - Preprocessed JSON cache files (parsed from XML)

**Format:**
XML files contain CPC hierarchy with:
- `classification-symbol` - CPC code (e.g., G06F16/00)
- `class-title` - Description/title
- `classification-item` - Individual classification entries
- `parent-child` relationships

---

### `resources/cpc_graph.pkl`
**Purpose:** Serialized NetworkX knowledge graph

**Contents:**
- Nodes: CPC classes (3-char) and subgroups (full symbols)
- Edges: parent_of, contains, related_to relationships
- Node attributes: title, level, is_allocatable

**Size:** ~50-100 MB
**Generated:** By `knowledge_graph.py` during build

---

### `resources/cpc_embeddings.npz`
**Purpose:** Compressed embedding vectors for all CPC subgroups

**Contents:**
- `symbols[]` - List of CPC symbols
- `embeddings[]` - Numpy array (250K × 768 dimensions)

**Size:** ~300-500 MB
**Model:** `all-mpnet-base-v2` (sentence-transformers)
**Generated:** By `knowledge_graph.py` during build

---

### `resources/cpc_graph_meta.json`
**Purpose:** Metadata about the knowledge graph build

**Contents:**
- `source_hash` - MD5 hash of source XML files
- `num_nodes` - Number of graph nodes
- `num_edges` - Number of graph edges
- `num_embeddings` - Number of embedding vectors
- `model_name` - Embedding model used
- `build_date` - Build timestamp

**Used for:** Change detection (rebuild if source files change)

---

### `resources/ipc_cpc_hints.txt`
**Purpose:** Hints for IPC to CPC classification mapping

**Contents:**
- Keyword patterns for classification guidance
- Domain-specific hints

**Status:** Available but NOT used in Phase 1 (semantic extraction only). Phase 2A uses knowledge graph embeddings instead.

---

## Configuration

### `.env` File (not in repo, create locally)
```bash
# LLM Model
LLM_MODEL=phi4-reasoning:plus

# Knowledge Graph
SKIP_KG=0              # Set to 1 to disable KG
KG_SECTIONS=G,H        # Build only specific sections

# EPO API (optional)
EPO_API_KEY=your_key_here
```

---

## Usage Flow

```
1. Patent text submitted to /classify endpoint
   |
2. Phase 1: LLM extracts semantic understanding (NO CPC codes)
   - technical_object, core_function, domain_signals, terms
   |
3. Phase 2A: Family Router selects top 3 CPC families
   - Uses KG embeddings OR domain signal heuristics
   - Distinguishes purpose vs tool domains
   |
4. Phase 2B: Restricted XML expansion (only within selected families)
   - Reduces search space ~98%
   |
5. Phase 2C: TF-IDF scoring with term matching
   |
6. Phase 3: Rank top 10 candidates
   |
7. Phase 4: Consolidate into max 2 hypotheses
   - Cluster by family, compute normalized scores
   - PRIMARY + optional SECONDARY (no tertiary)
   |
8. Phase 5: Deterministic resolution
   - Select exactly 1 primary (+ optional secondary if gap < 0.25)
   - Score-driven, no LLM classification
   |
9. Phase 6-7: Optional reconciliation and consistency checks
   |
10. Return JSON with primary/secondary families and decision logic
```

---

## Pipeline Architecture Principles

1. **Separation of Concerns**
   - Phase 1: Semantic understanding only
   - Phase 2+: CPC classification only

2. **Deterministic Scoring**
   - Phase 4: Normalized cluster scores
   - Phase 5: Weighted formula (0.5×phase4 + 0.3×func_align + 0.2×tech_cov)

3. **Constrained LLM Usage**
   - LLM only in Phase 1 (extraction) and Phase 6-7 (optional reconciliation)
   - No LLM for CPC classification or code generation

4. **Purpose vs Tool Distinction**
   - Computer vision (purpose) > AI/ML (tool)
   - Manufacturing (purpose) > AI/ML (tool)
   - Medical (purpose) > AI/ML (tool)

---

## File Size Summary

| File/Directory | Size | Type |
|----------------|------|------|
| `cpc_scheme_2026/` | ~239 MB | Source XML + JSON |
| `cpc_graph.pkl` | ~50-100 MB | Generated cache |
| `cpc_embeddings.npz` | ~300-500 MB | Generated cache |
| `chroma_db/` | ~10-50 MB | Vector DB |
| Total generated | ~400-650 MB | Cache files |

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
```

---

## Notes

- **Cache files** (`*.pkl`, `*.npz`) are in `.gitignore` - do not commit
- **First startup** takes 15-25 min if building knowledge graph
- **Subsequent startups** load cache in ~5 seconds
- **Colab build** recommended for knowledge graph (free T4 GPU)
- **SKIP_KG=1** for instant startup (domain-signal fallback mode)
- **Phase 4-5** are deterministic - no LLM used for classification decisions


# CPC classification MVP (Minimum Viable Product), Minste brukbare produkt.
Dette har vi:
Fase 1: Semantisk ekstraksjon (uten CPC-klasser)
Fase 2A: Familieruter (domene-bevisst)
Fase 2B: Begrenset ekspansjonFase 
2C: TF-IDF-skåring
Fase 3: Rangering
Fase 4: Klynging (maks 2 hypoteser)
Fase 5: Deterministisk løser (resolver)Hybrid-gjenfinning (BM25 + cross-encoder)
Dette er en MVP – det er en fungerende ende-til-ende-rørledning (pipeline) med alle kjernekomponenter.

Dette fungerer:
Ende-til-ende-rørledning: Fase 1 → Fase 5
Semantisk ekstraksjon uten CPC-lekkasjeFamilieruting med skille mellom formål og verktøy
Begrenset ekspansjon (98 % reduksjon av søkerommet)
TF-IDF-skåringKlynging i maks 2 hypoteser
Deterministisk løsningHybrid-gjenfinning (BM25 + cross-encoder)

Hva som gjør den «minimum» (minste løsning):
Bruker reserve-heuristikk (fallbacks) når kunnskapsgrafen (KG) er utilgjengelig
Ingen behov for finjustering (fine-tuning)
Kjører kun på CPU
Deterministisk (bruker ikke LLM for klassifiserings beslutninger)

Hva som gjør den «viable» (levedyktig):
Produserer strukturerte CPC-hypoteserSkiller forståelse fra klassifiseringHåndterer patenter som dekker flere domener korrekt
Gir konfidensskårer og beslutningslogikk

Vi kan nå klassifisere patenter i CPC.

# CV of the developer working in this project 



