The Ilyas-Classifier is designed as a "Self-Correcting Pipeline." It doesn't just process data; it audits itself at every step to ensure that by the time you reach the final classification, the logic is legally and technically sound.

Here is the step-by-step lifecycle of a patent through the system:

# Phase 1: Semantic Discovery (The "Raw" Pass)
The system reads the Abstract, Summary, and Background to build a preliminary "Technical Profile."

Action: It extracts three main lists:

Technical Objects (The "What").

Components/Glossary (The "How").

Domain Signals (Initial guesses like "Speech" or "Image").

Result: A messy but detailed set of technical signals.

# Phase 1.1: Recovery (The "Clarity" Pass)
Trigger: This only fires if Phase 1 fails to produce a clear summary or if the technical objects are too vague to understand.

Action: It looks back at the Glossary (the nouns) to force a "re-summarization" based on the tools used in the patent.

Result: A stabilized summary and component list.

# Phase 1.2: Mandatory Forensic Audit (The "Legal Gate")
Trigger: MANDATORY. Every patent goes through this before entering Phase 2.

Action: The system reads the Full Claims for the first time. It compares the "Domain Signals" from Phase 1 to the actual words used in the legal claims.

The Filter:

If Phase 1 said "Image" but the Claims never mention "Pixels," Phase 1.2 deletes the Image signal.

If the Claims mention "Speech" and "Video," Phase 1.2 validates a Hybrid.

Result: A Clean Anchor (e.g., G10L) that is legally proven by the claims.

# Phase 1.5: Invention Role Classification
Action: Classifies the invention's role (CORE_TECH / SYSTEM / APPLICATION / SUPPORT) and performs Technical Weight Analysis (TCR) to determine computational vs physical dominance.

Result: Role label + TCR score that guides downstream layer decomposition.

Phase 2: Multi-Source CPC Expansion & Scoring
Now that the system has a "Clean Anchor," it stops guessing and starts searching.

# Phase 2A v2: Fusion-Based Family Router
Replaces heuristic keyword routing with multi-source fusion:
  final_score = 0.45 × embedding + 0.35 × KG + 0.20 × anchor

All signals aligned to CPC family level (4-char, e.g. G10L).
Anchor-only penalty (0.5×) applied when no embedding/KG support.
Confidence filtering: MIN_SCORE=0.05, TOP_K=3.

# Phase 2B: Weighted Hierarchical Expansion
Expands Phase 2A families into scored CPC subgroups using:
  - KG hierarchy (class_to_subgroups) — primary
  - KG graph traversal (depth 2-3) — fallback
  - XML parser — last resort

Each subclass scored: inheritance (50%) + KG similarity (30%) + embedding (20%).
Expansion balance enforced: each family ≥ 10% of largest family's count.
Pruning: subclasses with score < 0.30 removed.
Output: flat_candidates (for Phase 2C) + family_expansions (structured).

# Phase 2C: Hybrid Scoring
TF-IDF (with bigrams) + Semantic Embeddings (0.4×TF-IDF + 0.6×Semantic).
Cross-domain guardrails: NN subject penalizes G06T/G10K, boosts G06N.
False-friend penalties for generic terms (exchange, response, system).

# Phase 2D: Subclass Structural Anchor Filter
Filters candidates to technical-layer anchors only.
Find-Until-Full expansion (500 → 1000 → all) until ≥ 20 survivors.
Excludes G06Q/B60 application-layer noise.

# Phase 3: The Multi-Dimensional Matcher
This is where the actual "Classifying" happens. The system compares the patent's specific features against the candidate codes found in Phase 2.

Action: It runs three internal sub-passes:

# Phase 3.1 (Object Match): Does this code cover "Speech Recognition"?

# Phase 3.2 (Method Match): Does this code cover "Neural Networks"?

# Phase 3.3 (Problem Match): Does this code solve "Accuracy issues in noisy environments"?

Result: A ranked list of CPC codes with individual scores.

# Phase 3.5: Decision Tree
Domain dominance check, canonical sorting (standard before indexing), quota guardrail.

# Phase 3.6: Cross-Domain Validation (The "Final Boss")
Action: A final LLM pass that looks at the top 3-5 results and asks: "Do these codes make sense together?"

The Conflict Check: If the system is suggesting a code for "Underwater Drilling" and "Speech Processing," Phase 3.6 will perform a final logic check to ensure no "Ghost Domains" survived.

Result: The Final Classification Report.

# Phase 4: Hypothesis Consolidation
Max 2 clusters + human-readable interpretation engine.

# Phase 5: The "Evidence" Mapping (Reasoning)
In this phase, the system builds the "Why." Patent offices and attorneys don't trust a code unless there is proof.

Action: For every CPC code selected in Phase 3, the system goes back into the Full Claims and the Description to find the exact sentences that justify that code.

Outcome: A structured "Evidence Table." For example:

Code G10L 15/02 (Feature Extraction) is justified by Claim 1, Line 14: "extracting a plurality of acoustic features..."

Result: The classification becomes "defensible" (ready for a legal audit).

Tri-Pillar FACETS: Goal (G06F) / Method (G06N) / Context (G05B) with back-scanning.

# Phase 6: Hierarchical Ranking & Formatting
A patent usually has one Primary code and several Secondary or Additional codes.

Action: The system applies the "Invention Information" vs. "Additional Information" rules. It ranks the codes based on which one represents the core inventive step (the "heart" of the patent).

Formatting: The data is transformed into the final JSON, XML, or PDF report requested by the user.

Result: A clean, professional classification sheet.

# Phase 7: The "Feedback Loop" (System Learning)
This is the "AI IQ" phase. The system reviews its own journey from Phase 1 to Phase 6.

Action: It compares its final result with the initial "hallucinations" (like the Image/Video error).

The Lesson: It asks, "Why did I think this was Video in Phase 1?"

Outcome: It updates its Local Knowledge Base (a "Memory Bank"). It adds a rule: "When keywords like 'Clustering' appear near 'Phonemes,' ignore the 'Image Processing' domain signal."

Result: The system becomes "seasoned." It won't make the same mistake twice.

# Phase 8: Quality Assurance (QA) & Human-in-the-Loop
This is the "Final Exit" where the system prepares for human review.

Action: The system calculates a Global Confidence Score.

If the score is >0.95, it marks it as "Auto-Certified."

If the score is lower, it highlights the "Problem Areas" (e.g., "I'm 70% sure about G10L, but check the hybrid link to H04L").

UI/UX: The data is pushed to a dashboard where a human examiner can click "Approve" or "Edit."

Result: A 100% verified classification that is ready for filing with the Patent Office.

# Summary of the Whole System
Phase Range | Focus | Role
Phases 1 - 1.2 | Extraction | Semantic understanding and Legal Audit (Claims).
Phase 1.5 | Role | Invention role classification + TCR.
Phases 2A-2D | Matching | Fusion-based routing → hierarchical expansion → hybrid scoring → anchor filtering.
Phases 3 - 3.6 | Ranking | Score candidates, decision tree, cross-domain validation.
Phases 4 - 5 | Reporting | Hypothesis consolidation, evidence mapping, Tri-Pillar FACETS.
Phases 6 - 8 | Optimization | Hierarchical ranking, self-learning, final QA.

By making Phase 1.2 mandatory, you have ensured that the Search (Phase 2) and Scoring (Phase 3) never waste time on hallucinations like that "Image Processing" error.

-------------------------- FILES DESCRIPTION -------------------------

# patent_cpc_fastapi — Project File Description (v2)

## Overview

**patent_cpc_fastapi** is the core CPC patent classification engine. It accepts raw patent text (description + claims), runs a 13-phase classification pipeline, and returns structured CPC codes with roles, facets, justification, and an executive report.

The pipeline uses a **multi-pass LLM architecture** for Phase 1 semantic extraction, **multi-source fusion** for Phase 2A family routing, **weighted hierarchical expansion** for Phase 2B, deterministic rule-based filtering for Phases 2C–2D, and LLM-generated narratives for Phase 8.

**Technology:** Python 3.13+, FastAPI, Ollama (LLM), Sentence Transformers (all-mpnet-base-v2), scikit-learn, NetworkX

**Last Updated:** 2026-05-15

---

## Directory Structure

```
patent_cpc_fastapi/
├── .env                              # LLM model, KG settings
├── .env.example                      # Template
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container instructions
├── app/
│   ├── main.py                       # FastAPI entry point (POST /classify, GET /health)
│   ├── files_description_updated.md  # Previous changelog
│   ├── files_description_updated_v0.md  # v0 changelog
│   ├── files_description_updated_v1.md  # v1 changelog
│   ├── files_description_updated_v2.md  # THIS FILE
│   └── cpc_classification/
│       ├── __init__.py               # Package init
│       ├── search_cpc.py             # THIN ORCHESTRATOR — calls phase runners
│       ├── extracting_cpc.py         # Phase 1: multi-pass LLM extraction
│       ├── cpc_xml_parser.py         # Parses CPC scheme XML files
│       ├── knowledge_graph.py        # CPC Knowledge Graph (embeddings + BM25 + hybrid retrieval)
│       ├── cpc_family_router.py      # Legacy family routing (pre-layer decomposer)
│       ├── cpc_layer_decomposer.py   # Phase 2A: 5-layer CPC decomposition
│       ├── cpc_hypothesis_consolidation.py  # Phase 4: clustering + interpretation engine
│       ├── cpc_hypothesis_resolver.py       # Phase 5: deterministic resolver + Tri-Pillar
│       ├── cpc_decision_tree.py      # Phase 3.5: constraints + canonical sorting
│       ├── cpc_cross_domain_validator.py    # Phase 3.6: cross-domain validation
│       ├── cpc_role_labeling.py      # Phase 8: 3-layer role labeling
│       ├── cpc_role_classifier.py    # Phase 1.5: invention role classification
│       ├── cpc_phase2d_anchor.py     # Phase 2D: subclass structural anchor filter
│       ├── technical_weight_analyzer.py     # TCR analysis
│       ├── cpc_bm25_index.py         # BM25 index builder for CPC cache files
│       ├── cpc_cross_encoder.py      # Cross-encoder for re-ranking
│       ├── cpc_hierarchy_engine.py   # CPC hierarchy traversal utilities
│       ├── prompts.py                # RE-EXPORT HUB — forwards to /prompts/
│       ├── precompute_cpc_embeddings.py  # Builds CPC embeddings for KG (offline)
│       ├── analysing_cpc_old.py      # Legacy CPC analysis utilities (archived)
│       │
│       ├── prompts/                  # LLM PROMPTS (5 sub-modules)
│       │   ├── __init__.py
│       │   ├── shared.py             # label_claims, detect_sections, IMPORTANCE_RUBRIC
│       │   ├── prompt_phase1.py      # Phase 1: passes A/B/C/D + reconciliation + completeness
│       │   ├── prompt_phases_5_6.py  # Phase 5 validation + Phase 6 claim reconciliation
│       │   ├── prompt_phases_35_4_7.py   # Phase 3.5 tie-breaker + Phase 4 sanity check + Phase 7 consistency
│       │   └── prompt_phase8.py      # Phase 8 executive report + at-a-glance card
│       │
│       ├── pipeline/                 # PHASE RUNNERS (10 files + 1 sub-module)
│       │   ├── __init__.py
│       │   ├── phase1_runner.py      # Phase 1: Extraction + completeness scoring
│       │   ├── phase1_2_runner.py    # Phase 1.2: Forensic claims audit
│       │   ├── phase15_tcr_runner.py # Phase 1.5: Role classification + TCR
│       │   ├── phase2_runner.py      # 2A v2 fusion → 2B expansion → 2C scoring → 2D anchor
│       │   ├── phase2b_expander.py   # Phase 2B: Weighted hierarchical CPC expansion
│       │   ├── phase3_runner.py      # Phase 3: Ranking + 3.5 decision tree + 3.6 cross-domain
│       │   ├── phase4_5_runner.py    # Phase 4-5: Hypothesis consolidation + Tri-Pillar
│       │   ├── phase7_runner.py      # Phase 7: Consistency check + feedback loop
│       │   ├── phase8_runner.py      # Phase 8: Role labeling + report generation
│       │   │
│       │   └── phase_2a_v2/          # PHASE 2A v2: Fusion Family Router (4 files)
│       │       ├── __init__.py
│       │       ├── phase_2a_v2_router.py  # Main fusion router (embedding + KG + anchor)
│       │       ├── embedding_router.py    # Embedding similarity scorer (with fallback)
│       │       ├── cpc_kg_client.py       # KG client interface (hierarchy + scoring)
│       │       └── anchor_matcher.py      # Anchor extraction + CPC matching
│       │
│       ├── scoring/                  # SCORING UTILITIES (3 files)
│       │   ├── __init__.py
│       │   ├── tfidf_scorer.py       # Tokenizer, bigrams, score_candidates()
│       │   ├── semantic_scorer.py    # compute_semantic_scores() via KG embeddings
│       │   └── domain_booster.py     # CPC_SYNONYMS, get_synonyms(), get_expanded_terms()
│       │
│       ├── utils/                    # GENERAL UTILITIES (3 files)
│       │   ├── __init__.py
│       │   ├── text_utils.py         # normalize_word(), shorten()
│       │   ├── json_utils.py         # parse_llm_json() with fallbacks
│       │   └── xml_utils.py          # resolve_xml_dir()
│       │
│       ├── tests/                    # TEST SUITES
│       │   ├── run_full_pipeline.py
│       │   ├── run_phase2_simulation.py
│       │   ├── test_phase2_refactor.py
│       │   ├── test_bm25_quick.py
│       │   └── test_hybrid_retrieval.py
│       │
│       └── resources/
│           └── cpc_scheme_2026/      # CPC class definition XML + JSON cache files
│
└── search_core/
    └── ollama_client.py              # Ollama LLM HTTP client
```

---

## Pipeline Phases

```
Phase 1    → Semantic extraction (multi-pass LLM: A/B/C/D + reconciliation)
Phase 1.2  → Mandatory forensic audit (claims-to-domain validation)
Phase 1.5  → Invention role classification (CORE_TECH / SYSTEM / APPLICATION / SUPPORT)
TCR        → Technical Weight Analysis (computational vs physical dominance)
Phase 2A v2→ Fusion Family Router: 0.45×embedding + 0.35×KG + 0.20×anchor
             → All signals aligned to family level (4-char)
             → Anchor-only penalty (0.5×), confidence filtering (MIN_SCORE=0.05, TOP_K=3)
Phase 2B   → Weighted Hierarchical Expansion:
             → KG hierarchy (primary) → graph traversal depth 2-3 (fallback) → XML (last resort)
             → Scoring: 50% inheritance + 30% KG similarity + 20% embedding
             → Expansion balance: each family ≥ 10% of largest
             → Pruning: score < 0.30 removed
             → Output: flat_candidates (Phase 2C) + family_expansions (structured)
Phase 2C   → Hybrid Scoring: TF-IDF (bigrams) + Semantic Embeddings (0.4×TF-IDF + 0.6×Semantic)
             → Cross-domain guardrails, false-friend penalties
Phase 2D   → Subclass Structural Anchor filter (technical layer anchors, excludes G06Q/B60 noise)
             → Find-Until-Full expansion (500 → 1000 → all) until ≥ 20 survivors
Phase 3    → Rank candidates
Phase 3.5  → Decision Tree: domain dominance, canonical sorting, quota guardrail
Phase 3.6  → Cross-Domain Validation: anti-domain collapse, entity consistency
Phase 4    → Hypothesis consolidation (max 2 clusters) + human-readable interpretation engine
Phase 5    → Deterministic resolution + Tri-Pillar FACETS (Goal / Method / Context) with back-scanning
Phase 7    → Internal consistency check + feedback loop (not shown as separate UI step)
Phase 8    → Executive Report: At-a-Glance card, Tech Stack table, Justification, Indexing Codes, Download
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health: LLM status, KG nodes, configured model |
| `POST` | `/classify` | Full classification pipeline |
| `POST` | `/classify_from_phase1` | Classification from pre-computed Phase 1 data |

### `/classify` Request
```json
{
  "text": "Full patent text including claims...",
  "claims": "Optional: separate claims text"
}
```

### `/classify` Response (minimal excerpt)
```json
{
  "phase1": { "technical_object": "...", "core_function": "...", ... },
  "phase2": {
    "phase2b_candidate_count": 242,
    "phase2b_expansion_counts": {"G10L": 200, "G06N": 42},
    "phase2b_output_type": "flat+structured",
    "phase2c_input_size": 242,
    "phase2d_kept_count": 20,
    ...
  },
  "phase3": [ { "symbol": "G10L15/00", "score": 0.8500 }, ... ],
  "phase5": { "pillars": { "pillar1_goal": {...}, ... } },
  "phase8_role_labeling": { "layer1_core": [...], ... },
  "premier": { "symbol": "G10L15/00", "confidence": "high" },
  "formatted_report": "# Executive Patent Classification Report\n..."
}
```

---

## Key Design Principles

1. **Thin orchestrator** — `search_cpc.py` delegates all logic to pipeline runners. Changing scoring? Edit `scoring/tfidf_scorer.py`. Changing prompts? Edit `/prompts/prompt_phase1.py`. The orchestrator just calls phases in order.

2. **Multi-pass Phase 1** — 4 independent LLM passes (function, structure, problem-solution, drawings) are reconciled into a single output with convergence-based confidence scoring.

3. **Phase 2A v2 Fusion Router** — Replaces heuristic keyword routing with multi-source fusion (embedding + KG + anchor). All signals aligned to CPC family level (4-char). Anchor-only penalty applied when no embedding/KG support. Confidence filtering removes low-scoring families.

4. **Phase 2B Weighted Expansion** — Expands families into scored subgroups using KG hierarchy (primary), graph traversal depth 2-3 (fallback), and XML (last resort). Each subclass scored via inheritance (50%) + KG similarity (30%) + embedding (20%). Expansion balance enforced across families.

5. **Hybrid scoring** — Phase 2C blends TF-IDF (with bigrams) and sentence-transformer semantic embeddings at 0.4/0.6 ratio. Falls back to TF-IDF-only when KG unavailable.

6. **Find-Until-Full** — Phase 2D progressively expands the candidate pool (500 → 1000 → all) until at least 20 technical anchor survivors are found.

7. **Tri-Pillar FACETS** — Phase 5 back-scans raw candidates for the best code per functional role: Goal (G06F), Method (G06N), Context (G05B).

8. **Phase 6+7 decommissioned from UI** — consistency check runs internally and feeds into Phase 5 premier code and Phase 8 report, but is not exposed as a separate step.

9. **Re-export hub pattern** — `prompts.py` is a thin re-export of `/prompts/` sub-modules. All existing imports continue to work unchanged.

10. **Flat + structured output contract** — Phase 2B outputs both `flat_candidates` (for Phase 2C consumption) and `family_expansions` (structured, for UI/debugging). Pruning happens after flattening.

---

## Key Files — What They Do

### Orchestrator

| File | Purpose |
|------|---------|
| `search_cpc.py` | Thin orchestrator. `classify()` calls runners in sequence. Contains `_resolve_premier()`, `_build_formatted_report()`, and legacy `classify_from_phase1()`. |

### Pipeline Runners

| File | Function | What it runs |
|------|----------|-------------|
| `phase1_runner.py` | `run_phase1()` | Calls extractor, runs completeness scoring, logs PASS/WARN/FAIL |
| `phase1_2_runner.py` | `run_phase1_2()` | Forensic claims audit: validates Phase 1 domain signals against claims text |
| `phase15_tcr_runner.py` | `run_phase15_tcr()` | Phase 1.5 role classification + TCR weight analysis |
| `phase2_runner.py` | `run_phase2()` | 2A v2 fusion router → 2B weighted expansion → 2C hybrid scoring → 2D anchor filter with Find-Until-Full |
| `phase2b_expander.py` | `Phase2BExpander` | Weighted hierarchical expansion: KG hierarchy → graph traversal → XML fallback |
| `phase3_runner.py` | `run_phase3()` | Sorting + 3.5 decision tree constraints + 3.6 cross-domain validation |
| `phase4_5_runner.py` | `run_phase4_5()` | Hypothesis consolidation + deterministic resolver + Tri-Pillar back-scan |
| `phase7_runner.py` | `run_phase7()` | LLM consistency check → feedback loop re-filtering → premier code resolution |
| `phase8_runner.py` | `run_phase8()` | Role labeling (CORE/SUPPORT/CONTEXT/LEGAL_COVERAGE) + formatted report generation |

### Phase 2A v2 Sub-Module

| File | Purpose |
|------|---------|
| `phase_2a_v2_router.py` | Main fusion router: aligns all signals to family level, fuses (0.45×emb + 0.35×KG + 0.20×anchor), applies anchor-only penalty, filters by confidence |
| `embedding_router.py` | Embedding similarity scorer with fallback (sentence-transformers vs hardcoded CPC descriptions) |
| `cpc_kg_client.py` | KG client interface: hierarchy expansion, graph traversal, semantic scoring — all at family level |
| `anchor_matcher.py` | Extracts structural/action/domain anchors from Phase 1, matches to CPC families |

### Prompts

| File | Contains |
|------|----------|
| `shared.py` | `label_claims()`, `detect_sections()`, `UNIFIED_IMPORTANCE_RUBRIC` |
| `prompt_phase1.py` | `phase1_pass_a/b/c/d_prompt()`, `phase1_reconciliation_prompt()`, `score_phase1_completeness()`, `phase1_prompt()`, `domain_inference_prompt()` |
| `prompt_phases_5_6.py` | `validation_prompt_single()` (Phase 5), `reconciliation_prompt()` (Phase 6) |
| `prompt_phases_35_4_7.py` | `phase35_tiebreaker_prompt()`, `phase4_sanity_check_prompt()`, `phase7_consistency_prompt()` |
| `prompt_phase8.py` | `phase8_report_prompt()`, `phase85_card_prompt()` |

### Scoring

| File | Exports |
|------|---------|
| `tfidf_scorer.py` | `tokenize()`, `tokenize_with_bigrams()`, `make_term_bigrams()`, `score_candidates()` |
| `semantic_scorer.py` | `compute_semantic_scores()` — cosine similarity via KG embeddings |
| `domain_booster.py` | `CPC_SYNONYMS`, `get_synonyms()`, `get_expanded_terms()` |

### Utilities

| File | Exports |
|------|---------|
| `text_utils.py` | `normalize_word()`, `shorten()` |
| `json_utils.py` | `parse_llm_json()` — 3 fallback strategies |
| `xml_utils.py` | `resolve_xml_dir()` |

### Classification Modules

| File | Purpose |
|------|---------|
| `extracting_cpc.py` | `CPCExtractor` — multi-pass Phase 1 extraction with reconciliation |
| `cpc_xml_parser.py` | `CPCXMLParser` — parses CPC scheme XML, expands classes into subgroups |
| `knowledge_graph.py` | `CPCKnowledgeGraph` — embeddings DB, BM25 index, hybrid retrieval (BM25 + embedding + cross-encoder) |
| `cpc_family_router.py` | `CPCFamilyRouter` — legacy family routing (used in classify_from_phase1 path) |
| `cpc_layer_decomposer.py` | `CPCLayerDecomposer` — 5-layer decomposition (application, pure_software, data_reasoning, interaction, control) |
| `cpc_hypothesis_consolidation.py` | `CPCHypothesisConsolidator` — Jaccard clustering + interpretation engine |
| `cpc_hypothesis_resolver.py` | `CPCHypothesisResolver` — deterministic scorer + Tri-Pillar resolution |
| `cpc_decision_tree.py` | `CPCDecisionTreeConstraint` — domain dominance, disambiguation, canonical sort, quota guardrail |
| `cpc_cross_domain_validator.py` | `CrossDomainValidator` — anti-collapse rules, entity consistency, family lock |
| `cpc_role_labeling.py` | `CPCRoleLabeling` — 3-layer model (CORE/SUPPORT/CONTEXT/LEGAL_COVERAGE) |
| `cpc_role_classifier.py` | `CPCRoleClassifier` — Phase 1.5 role classification |
| `cpc_phase2d_anchor.py` | `Phase2DSubclassAnchor` — 4-char subclass prefix filter with family router fallback |
| `technical_weight_analyzer.py` | `TechnicalWeightAnalyzer` — TCR analysis (FORCE_SOFTWARE_CORE vs FORCE_DOMAIN_CORE) |
| `cpc_bm25_index.py` | `CPCBM25Index` — BM25 index builder for CPC cache files |
| `cpc_cross_encoder.py` | Cross-encoder re-ranking utilities |
| `cpc_hierarchy_engine.py` | CPC hierarchy traversal and expansion utilities |

### Supporting Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — `/health`, `/classify`, `/classify_from_phase1` endpoints |
| `precompute_cpc_embeddings.py` | Builds `cpc_embeddings.npz` for KG — runs offline |
| `analysing_cpc_old.py` | Legacy CPC analysis utilities (archived) |
| `prompts.py` (root of cpc_classification/) | Re-export hub for `/prompts/` sub-modules |

---

## How to Run

```bash
# Terminal 1 — FastAPI backend
cd patent_cpc_fastapi
uvicorn app.main:app --port 8000 --reload

# Terminal 2 — MCP proxy (optional, for LLM agent integration)
cd ../MCP_patent_classification
npm run dev

# Terminal 3 — Streamlit UI
cd ..
streamlit run cpc_client/streamlit_app.py --server.port 8502
```

## Environment Variables (.env)

```
LLM_MODEL=phi4-reasoning:plus    # Ollama model for LLM calls
SKIP_KG_REBUILD=1                # Skip knowledge graph build at startup
KG_SECTIONS=E,F,G,H              # (commented) Limit KG sections for faster builds
```

---

## Recent Architecture Changes (v1.1 → v2.0)

- **Phase 2A v2 Fusion Router** — Replaced heuristic keyword routing with multi-source fusion (embedding + KG + anchor). All signals aligned to CPC family level (4-char). Added anchor-only penalty (0.5×) and confidence filtering (MIN_SCORE=0.05, TOP_K=3).
- **Phase 2B Weighted Expansion** — Replaced XML-only expansion with KG hierarchy (primary), graph traversal depth 2-3 (fallback), and XML (last resort). Each subclass scored via inheritance (50%) + KG similarity (30%) + embedding (20%). Expansion balance enforced across families.
- **Pipeline contract fix** — Phase 2B now outputs `flat_candidates` (for Phase 2C) + `family_expansions` (structured). Pruning happens after flattening. Added validation guard before Phase 2C.
- **Phase 2A v2 moved to pipeline/** — `phase_2a_v2/` directory moved into `pipeline/` for better organization.
- **Streamlit UI updated** — Phase 2B display now shows expansion balance, pruned count, and weighted scoring details.
- **Debug trace added** — `phase2b_output_type`, `phase2c_input_size`, `expansion_balance`, `anchor_only_penalty_applied` fields added to pipeline output.
