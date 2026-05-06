# Patent CPC FastAPI - File Descriptions

## Overview

This directory contains the FastAPI backend for patent CPC (Cooperative Patent Classification) classification. The system uses a multi-phase pipeline combining LLM-based extraction, XML-based scoring, knowledge graph semantic search, and validation gates.

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
    ├── cpc_fast_index.py           # Fast indexing for CPC lookup
    ├── cpc_xml_parser.py           # XML parser for CPC scheme files
    ├── epo_client.py              # EPO (European Patent Office) API client
    ├── extracting_cpc.py          # Phase 1: LLM extraction of patent terms
    ├── knowledge_graph.py         # Phase 2a: Knowledge graph with semantic embeddings
    ├── prompts.py                 # LLM prompts for all phases
    ├── search_cpc.py              # Main pipeline orchestrator (Phases 1-7)
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
- `search_cpc.py` (Phase 5 validation)

---

## CPC Classification Pipeline

### `cpc_classification/search_cpc.py`
**Purpose:** Main pipeline orchestrator - coordinates all classification phases

**Pipeline Phases:**
1. **Phase 1** - LLM extraction (terms, context, function)
2. **Phase 1b** - Domain inference (probabilistic, replaces hardcoded rules)
3. **Phase 2a** - Knowledge graph query (semantic search) *[NEW]*
4. **Phase 2b** - XML expansion + TF-IDF scoring # defintion: TF-IDF measure how important a word is to a document in a collection of documents.
5. **Phase 3** - Ranking by composite score # ranking by composite score means ranking by importance of each class in the collection of documents.
6. **Phase 5** - Multi-pass validation (one candidate per prompt)
7. **Phase 6** - Per-claim reconciliation
8. **Phase 7** - Final consistency check

**Key Class:** `CPCClassifier`  # Classififer is a class , in file cpc_classification/search_cpc.py 
- Accepts optional `knowledge_graph` parameter # knowledge_graph is a class, it is in file cpc_classification/knowledge_graph.py that have all data about cpc classes stored in directed acyclic graph. search better then beep deep tree method, because it used semantic search and graph search algrithm, not just keywords matching, so it more accurate and efficient to find the most relevant cpc classes.

- Falls back to LLM-only if graph not available, to have backward compatibility with the previous version which used only LLM for classification. LLM can may not be updated about new CPC classes and LLM may also use only old traning data about CPC classes. That is why we need to use knowledge_graph to get the most up to date information about CPC classes.

- Returns structured JSON with all phase results

---
 
### `cpc_classification/extracting_cpc.py`
**Purpose:** Phase 1 - Extract technical terms and concepts from patent text

**Key Functions:**
- `extract()` - Main extraction method
- `label_claims()` - Label claims as independent/dependent
- Term normalization and importance weighting, wicth means removing duplicates and organizing terms by importance, for example terms from independent claims are more important than terms from background and summary
- Section-aware extraction (claims get 2x weight)

**Output:**
- `technical_object`, `problem_solved`, `core_function`
- `system_context`, `classification_strategy`
- `terms[]` with importance scores
- `negative_signals[]` for downstream filtering

---

### `cpc_classification/knowledge_graph.py`
**Purpose:** Phase 2a - Semantic search using knowledge graph + embeddings, knowlegde graph pre-computes during the build time. and its stored as cache file , so it can be reused for multiple runs. Also embeddings are pre-computed and stored as cache file. the knowledge graph + embeddings is much better than LLM alone for CPC classification because it is more accurate and efficient. It also can be updated separately from the LLM, so we can update the knowledge graph and embeddings without retraining the LLM. The knowledge graph + embeddings is much better than LLM alone for CPC classification because it is more accurate and efficient. The semantic search is done using sentence-BERT embeddings, which are pre-computed and stored as cache file. The graph search is done using NetworkX graph, which is pre-computed and stored as cache file.

**Key Class:** `CPCKnowledgeGraph`

**Features:**
- Builds NetworkX graph from CPC XML/JSON files
- Generates sentence-BERT embeddings (`all-mpnet-base-v2`, 110MB), better embeddings like openai or gemini are also available , but openai is commercial and gemini is not open source. so sentence-BERT is a good choice for open source.
- Semantic search: cosine similarity between patent text and CPC titles, may also updated using langchain's Embeddings models, which is much better for CPC classification.
- Graph query: find classes covering extracted concepts
- Hybrid scoring: semantic (60%) + graph (40%)

**Cache Files:**
- `cpc_graph.pkl` - NetworkX graph structure, can also use neo4j database for storing knowledge graph, which is more powerful and scalable than NetworkX graph. NetworkX is just a python library for graph manipulation, while neo4j is a graph database.
- `cpc_embeddings.npz` - Embedding vectors
- `cpc_graph_meta.json` - Build metadata and source hash

**Build Time:**
- First run: 15-25 minutes (depends on GPU)
- Subsequent runs: 5 seconds (load from cache)

---

### `cpc_classification/prompts.py`
**Purpose:** LLM prompts for all pipeline phases

**Key Functions:**
- `phase1_prompt()` - Phase 1 extraction prompt
- `domain_inference_prompt()` - Phase 1b domain probability estimation
- `validation_prompt_single()` - Phase 5 single-candidate validation
- `reconciliation_prompt()` - Phase 6 per-claim reconciliation
- `consistency_check_prompt()` - Phase 7 final check

**Prompt Features:**
- Section-aware term extraction (background downweighted)
- Claim type analysis (method vs apparatus)
- Soft class hypotheses with confidence scores
- Dynamic domain appropriateness checking

---

### `cpc_classification/cpc_xml_parser.py`
**Purpose:** Parse CPC scheme XML files into usable data structures

**Key Functions:**
- `expand_classes()` - Expand CPC class codes into subgroups
- Parse XML hierarchy (parent-child relationships)
- Extract titles, symbols, and allocatable flags

**Input:** `cpc_scheme_2026/cpc-scheme-*.xml`
**Output:** List of subgroup dictionaries with symbol, title, level, parent_chain

---

### `cpc_classification/epo_client.py`
**Purpose:** Client for EPO (European Patent Office) APIs , but not used in the main pipeline. instead of that we use local CPC XML files for building knowledge graph. we use local files because they are more reliable and faster to access than EPO APIs, and also EPO APIs is like a balckbox we dont realy know witch method or algrithm is used to generate the CPC codes. instead we can build our own knowledge graph with our own logic and methods which is more transparent and controllable.  

Also due we can not give our private data like no publicated patents to EPO APIs for cpc classification

We have tested EPO API in public test data set and it was not very accurate,  espeacially for new publicased application in 2026,what many not be used from GPT or any other LLM for training yet. for this cases knowledge graph is much better because it is updated regually . in our test it was around 60-70% accurate in classification. for this case we will fallback to GPT for classifiaction but it will be slower and less accurate. we have left the EPO API as an option but not used in the main pipeline. it is more for the experimantal purposes.

**Key Functions:**
- Fetch CPC hierarchy from EPO Linked Open Data
- Retrieve patent classifications from Espacenet
- OPS (Open Patent Services) API integration

**Note:** Requires free EPO API key for some endpoints

---

### `cpc_classification/cpc_chroma_store.py`
**Purpose:** ChromaDB vector database integration

**Features:**
- Store CPC subgroup embeddings in ChromaDB
- Vector similarity search alternative
- Persistent storage in `resources/chroma_db/`

**Status:** Optional enhancement, not used in main pipeline, can be used in phase 2a instead of using local cache files as fallback mechanism, and also for big data processing or production environment.  
In development we used local cache files because they are faster to access than ChromaDB and also for simplicity of the project.  
In production environment with lots of data , ChromaDB will be a better choice for storing embeddings because it is more scalable and efficient for large-scale searches.  

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
**Generated:** By `knowledge_graph.py` and  during build, but we have used Colab for that suppose because we can use gree GPU for embedding generation and also we can use more powerful hardware for building the graph.  see file [build_kg_colab.ipynb](../cpc_client/build_kg_colab.ipynb), we have used Google colab instead of building it on our local machine because google colab provides free GPU and also we can use more powerful hardware for building the graph. we than copy the output files graph and embeddings to our local machine.

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
- Domain-specific hints (e.g., "neural network -> G06N")
- Used by Phase 1 LLM extraction

---

## Configuration

### `.env` File (not in repo, create locally)
```bash
# LLM Model
LLM_MODEL=phi4-reasoning:plus # free local model for better performance of scientific texts like patents, we have tested gpt-oss model too but it was not good for this purpose, also we have tried other models like phi3,phi2 but they were not as good as phi4-reasoning:plus. 

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
2. Phase 1: LLM extracts technical terms and concepts
   |
3. Phase 1b: LLM infers domain probabilities
   |
4. Phase 2a: Knowledge graph suggests CPC classes (semantic search)
   |
5. Phase 2b: Combine LLM + graph suggestions, expand XML subgroups
   |
6. Phase 2c: TF-IDF scoring with term matching
   |
7. Phase 3: Rank top 7 candidates
   |
8. Phase 5: Validate each candidate individually
   |
9. Phase 6: Reconcile per-claim classifications
   |
10. Phase 7: Final consistency check
    |
11. Return JSON with premier class, per-claim mappings, validation results
```

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
- **Colab build** recommended for knowledge graph (free T4 GPU)  see file [build_kg_colab.ipynb](../cpc_client/build_kg_colab.ipynb), used about 10 hours for building the full graph.
- **SKIP_KG=1** for instant startup (LLM-only mode)
