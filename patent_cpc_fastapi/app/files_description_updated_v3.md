# files_description_updated_v3.md

## Changes from v2 → v3 (2026-05-16)

### Critical Bug Fixes

1. **Phase 2C pre-filter bug fixed** — `if score > 0:` gate at `phase2_runner.py:445` was silently dropping candidates with zero TF-IDF term overlap (typically 94% of candidates). Removed. All 320+ candidates now flow through to Phase 2D which applies the anchor-based filter. Added logging to report non-zero vs zero candidate split.

2. **Phase 2B pruning threshold override** — `_enforce_balance_with_fallback()` in `phase2b_expander.py` now has a 4th cascading step: when graph traversal and standard XML fallback both fail, it calls `_expand_via_xml_fallback_unpruned()` (no scoring threshold) to force balance. Logs the threshold relaxation `0.30 → 0.0`.

### Architecture Simplifications

3. **Phase 2A-Layer LLM decomposer removed** — `cpc_layer_decomposer.py` archived → `.archived`. The LLM-based 5-layer decomposition was expensive, produced misleading output, and didn't influence classification. Replaced with a hardcoded fallback in `phase2_runner.py` that provides the layer structure Phase 2D depends on. Phase 2D's `_build_anchor_set` falls through to the family-router path, reading `phase2a_families` injected after Phase 2A v2 routing.

4. **Phase 1.5 → pure JSON** — `cpc_role_classifier.py` now outputs only `{"role": "...", "confidence": ...}`. Removed: `structural_signals`, `justification`, `claim_format`, `tcr_hint`, `audit`, `reasoning`, all prompt templates (`SYSTEM_PROMPT`, `ROLE_CLASSIFICATION_PROMPT`), `KEYWORD_TO_STRUCTURAL_SIGNAL` mapping. Phase 1.5 is machine-readable only. UI mapping lives in `streamlit_app.py`'s `ROLE_UI_MAP`.

### Phase 1 Output Upgrade

5. **Two new output layers** added to Phase 1 reconciliation prompt:
   - `core_function_generalized`: 3 broad verb-phrase rewordings of the core function (removes over-specific terms for CPC matchability)
   - `cpc_terms`: 6-10 noun phrases in standard patent classification language
   - Fallback generation in `_merge_pass_fallback` when reconciliation LLM fails

6. **Evidence table deduplication** — `deduplicate_evidence_table()` normalizes terms (lowercase, strip punctuation, collapse whitespace), keeps highest weight per unique term, enforces max 8 rows.

7. **Weight validation** — `validate_evidence_weights()` caps structural noise terms (acquisition module, processor, etc.) at weight 4, caps uncited weight ≥7 terms at 6.

8. **Strategy reconstruction forced** — `phase1_runner.py` forces `_ensure_classification_strategy()` reconstruction when strategy is `None` or a plain string, with assertions for `anchor_split` validity.

### UI Changes

9. **Phase 1.2 tab added** to Streamlit dropdown and display section — shows audit status, primary/secondary anchors, signal validations with emoji indicators, conflicts, rejected domains, reasoning, raw JSON expander.

10. **Phase 2A-Layer tab removed** from Streamlit — the LLM decomposer is gone, the explanatory layer view is no longer relevant.

11. **Phase 1.5 UI mapping centralized** — `ROLE_UI_MAP` dict replaces inline conditional formatting. Removed "Role Definition" and "Justification" sections (data no longer emits them).

---

## Pipeline Architecture (v3)

```
Phase 1      → Semantic extraction (multi-pass LLM: A/B/C/D + reconciliation)
               Outputs: technical_object, problem, core_function_precise,
               core_function_generalized[3], cpc_terms[6-10], system_context,
               domain_signals, evidence_table (deduped, max 8 rows), glossary,
               classification_strategy (structured with anchor_split)

Phase 1.2    → Mandatory forensic audit (claims-to-domain validation)
               Validates Phase 1 domain signals against actual claims text.
               Rejects/downgrades signals not supported by claims.

Phase 1.5    → Structural role tagging. Machine-readable only.
               Output: {"role": "CORE_TECH", "confidence": 0.85}
               No descriptive text, no signals, no justification.

TCR          → Technical Weight Analysis (computational vs physical dominance)

Phase 2A v2  → Fusion Family Router: 0.45×embedding + 0.35×KG + 0.20×anchor
               All signals aligned to family level (4-char)
               Anchor-only penalty (0.5×), confidence filtering (MIN_SCORE=0.05, TOP_K=3)

Phase 2B     → Weighted Hierarchical Expansion:
               KG hierarchy (primary) → graph traversal depth 2-3 (fallback) → XML (last resort)
               Scoring: 50% inheritance + 30% KG similarity + 20% embedding
               Expansion balance: each family ≥ 10% of largest
               Pruning: score < 0.30 removed (relaxed to 0.0 as last resort)
               Output: flat_candidates (Phase 2C) + family_expansions (structured)

Phase 2C     → Hybrid Scoring: TF-IDF (bigrams) + Semantic Embeddings (0.4×TF-IDF + 0.6×Semantic)
               ALL candidates flow through (no pre-filter). Zero-score candidates included.
               Cross-domain guardrails, false-friend penalties.

Phase 2D     → Subclass Structural Anchor filter (technical layer anchors via family-router fallback)
               Find-Until-Full expansion (500 → 1000 → all) until ≥ 20 survivors.
               Excludes G06Q/B60 application-layer noise.

Phase 3      → Rank candidates + 3.5 decision tree + 3.6 cross-domain validation

Phase 4      → Hypothesis consolidation (max 2 clusters) + interpretation engine

Phase 5      → Deterministic resolution + Tri-Pillar FACETS (Goal/Method/Context) with back-scanning

Phase 7      → Internal consistency check + feedback loop (not exposed as separate UI step)

Phase 8      → Executive Report: At-a-Glance card, Tech Stack table, Justification, Indexing Codes
```

---

## Directory Structure

```
patent_cpc_fastapi/
├── .env                              # LLM model, KG settings
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container instructions
├── app/
│   ├── main.py                       # FastAPI entry point (POST /classify, GET /health)
│   ├── files_description_updated.md
│   ├── files_description_updated_v0.md
│   ├── files_description_updated_v1.md
│   ├── files_description_updated_v2.md
│   ├── files_description_updated_v3.md  # THIS FILE
│   └── cpc_classification/
│       ├── __init__.py
│       ├── search_cpc.py             # THIN ORCHESTRATOR — calls phase runners
│       ├── extracting_cpc.py         # Phase 1: multi-pass LLM + dedup + weight validation
│       ├── cpc_xml_parser.py         # Parses CPC scheme XML files
│       ├── knowledge_graph.py        # CPC Knowledge Graph (embeddings + BM25 + hybrid)
│       ├── cpc_family_router.py      # Legacy family routing (used in classify_from_phase1)
│       ├── cpc_layer_decomposer.py.archived  # Phase 2A-Layer decomposer (removed, hardcoded fallback)
│       ├── cpc_hypothesis_consolidation.py   # Phase 4: clustering + interpretation
│       ├── cpc_hypothesis_resolver.py        # Phase 5: deterministic resolver + Tri-Pillar
│       ├── cpc_decision_tree.py      # Phase 3.5: constraints + canonical sorting
│       ├── cpc_cross_domain_validator.py     # Phase 3.6: cross-domain validation
│       ├── cpc_role_labeling.py      # Phase 8: 3-layer role labeling (+ UI mapping?)
│       ├── cpc_role_classifier.py    # Phase 1.5: pure JSON role tagging (no explanatory text)
│       ├── cpc_phase2d_anchor.py     # Phase 2D: subclass anchor filter (uses family-router fallback)
│       ├── technical_weight_analyzer.py     # TCR analysis
│       ├── cpc_bm25_index.py         # BM25 index builder for CPC cache
│       ├── cpc_cross_encoder.py      # Cross-encoder for re-ranking
│       ├── cpc_hierarchy_engine.py   # CPC hierarchy traversal utilities
│       ├── prompts.py                # Re-export hub → /prompts/
│       ├── precompute_cpc_embeddings.py  # Builds CPC embeddings (offline)
│       ├── analysing_cpc_old.py      # Legacy utilities (archived)
│       │
│       ├── prompts/                  # LLM PROMPTS (5 sub-modules)
│       │   ├── __init__.py
│       │   ├── shared.py             # label_claims(), detect_sections(), IMPORTANCE_RUBRIC
│       │   ├── prompt_phase1.py      # Phase 1 passes + reconciliation + generalized/cpc_terms output
│       │   ├── prompt_phases_5_6.py  # Phase 5 validation + Phase 6 reconciliation
│       │   ├── prompt_phases_35_4_7.py   # Phase 3.5 tie-breaker + Phase 4 sanity + Phase 7 consistency
│       │   └── prompt_phase8.py      # Phase 8 executive report + at-a-glance card
│       │
│       ├── pipeline/                 # PHASE RUNNERS (9 files + 1 sub-module)
│       │   ├── __init__.py
│       │   ├── phase1_runner.py      # Phase 1: extraction + completeness + forced strategy reconstruction
│       │   ├── phase1_2_runner.py    # Phase 1.2: forensic claims audit
│       │   ├── phase15_tcr_runner.py # Phase 1.5: role + confidence only + TCR
│       │   ├── phase2_runner.py      # 2A v2 fusion → 2B expansion → 2C scoring (all pass) → 2D anchor
│       │   ├── phase2b_expander.py   # Phase 2B: weighted expansion + pruning relaxation fallback
│       │   ├── phase3_runner.py      # Phase 3: ranking + 3.5 + 3.6
│       │   ├── phase4_5_runner.py    # Phase 4-5: consolidation + Tri-Pillar
│       │   ├── phase7_runner.py      # Phase 7: consistency check + feedback loop
│       │   ├── phase8_runner.py      # Phase 8: role labeling + report
│       │   │
│       │   └── phase_2a_v2/          # PHASE 2A v2 Fusion Router (4 files)
│       │       ├── __init__.py
│       │       ├── phase_2a_v2_router.py   # Main fusion router
│       │       ├── embedding_router.py     # Embedding similarity scorer
│       │       ├── cpc_kg_client.py        # KG client interface
│       │       └── anchor_matcher.py       # Anchor extraction + CPC matching
│       │
│       ├── scoring/                  # SCORING UTILITIES (3 files)
│       │   ├── __init__.py
│       │   ├── tfidf_scorer.py       # Tokenizer, bigrams, score_candidates()
│       │   ├── semantic_scorer.py    # compute_semantic_scores() via KG embeddings
│       │   └── domain_booster.py     # Synonyms + expanded terms
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
│       │   ├── test_phase15_role_classifier.py
│       │   ├── test_bm25_quick.py
│       │   └── test_hybrid_retrieval.py
│       │
│       └── resources/
│           └── cpc_scheme_2026/      # CPC XML + JSON cache
│
└── search_core/
    └── ollama_client.py              # Ollama LLM HTTP client
```

---

## Key Files — What They Do (v3 updates)

### Pipeline Runners

| File | Changes in v3 |
|------|---------------|
| `phase1_runner.py` | Added forced `_ensure_classification_strategy()` reconstruction with `anchor_split` assertions when strategy is `None` or plain string. |
| `phase15_tcr_runner.py` | Validates only `role` + `confidence`. Removed all `structural_signals` assertions. Fallback is `{"role": "SYSTEM", "confidence": 0.5}`. |
| `phase2_runner.py` | **Critical fix**: Removed `if score > 0` pre-filter that silently dropped 94% of candidates. All candidates now flow through to Phase 2D. Phase 2A-Layer: hardcoded fallback replaces LLM decomposer. `layer_result` injected with `phase2a_families` for Phase 2D family-router fallback. |
| `phase2b_expander.py` | Added `_expand_via_xml_fallback_unpruned()` (no score threshold). `_enforce_balance_with_fallback()` has 4th cascading step: pruning threshold relaxation `0.30 → 0.0` when all strategies fail. |

### Classification Modules

| File | Changes in v3 |
|------|---------------|
| `extracting_cpc.py` | Added `deduplicate_evidence_table()` (normalize terms, keep highest weight, max 8 rows). Added `validate_evidence_weights()` (structural noise terms capped at 4, uncited weight ≥7 capped at 6). Added `STRUCTURAL_NOISE_TERMS` set. |
| `cpc_role_classifier.py` | **Pure JSON output**: only `role` + `confidence`. Removed `SYSTEM_PROMPT`, `ROLE_CLASSIFICATION_PROMPT`, `KEYWORD_TO_STRUCTURAL_SIGNAL`, `DATA_TRANSFORM_KEYWORDS`, `_compute_tcr_hint`, `_detect_claim_format`, `_build_justification`, `_parse_response`, `CLAIM_FORMAT_PATTERNS`. No explanatory text anywhere. |
| `cpc_layer_decomposer.py.archived` | File archived. LLM-based 5-layer decomposition removed. Replaced by hardcoded fallback in `phase2_runner.py`. |
| `technical_weight_analyzer.py` | Removed reference to `phase1_5_result.get("audit", {})` (field no longer exists in Phase 1.5 output). |

### Prompts

| File | Changes in v3 |
|------|---------------|
| `prompt_phase1.py` | Added TASK 9 with instructions for `core_function_generalized` (3 verb-phrase rewordings) and `cpc_terms` (6-10 CPC-style noun phrases). Updated output format contract. |

### UI

| File | Changes in v3 |
|------|---------------|
| `streamlit_app.py` | Added "Phase 1.2: Forensic Claims Audit" tab with full display block. Removed "Phase 2A-Layer: Explanation" tab. Centralized `ROLE_UI_MAP` dict for Phase 1.5 display. Removed "Role Definition" and "Justification" sections. |

### Tests

| File | Changes in v3 |
|------|---------------|
| `test_phase15_role_classifier.py` | Updated `test_empty_phase1_data` to assert `len(result) == 2` instead of checking `structural_signals`. |

---

## Key Design Principles (v3 additions)

11. **All candidates must flow** — Phase 2C must not pre-filter by score. Even zero-score candidates must reach Phase 2D's anchor filter. The `if score > 0` was a silent pipeline bug dropping 94% of candidates.

12. **Phase 1.5 is pure data, not text** — The role classifier emits only machine-readable JSON: `{"role": "...", "confidence": ...}`. Zero sentences, zero explanations, zero definitions. UI mapping is the frontend's responsibility via `ROLE_UI_MAP`.

13. **Hardcoded fallback over LLM when output isn't consumed** — Phase 2A-Layer was an expensive LLM call producing explanatory-only output that didn't influence classification. Replaced with a static dict. Phase 2D uses the family-router fallback path instead.

14. **Cascading fallback with threshold relaxation** — Phase 2B's balance enforcement now has 4 cascading strategies: hierarchy → graph → XML → threshold relaxation. The final step removes the pruning score floor entirely to force balance.

---

## Recent Architecture Changes (v2 → v3)

- **Phase 2C pre-filter bug fixed** — Removed `if score > 0` gate at `phase2_runner.py:445`. All candidates now flow through to Phase 2D anchor filtering. Added non-zero vs zero candidate logging.
- **Phase 2B pruning relaxation added** — New `_expand_via_xml_fallback_unpruned()` method. `_enforce_balance_with_fallback()` relaxes `MIN_SUBCLASS_SCORE` from 0.30 to 0.0 as final cascading step.
- **Phase 2A-Layer decomposer archived** — `cpc_layer_decomposer.py` → `.archived`. LLM decomposition replaced with hardcoded fallback in `phase2_runner.py`.
- **Phase 1.5 stripped to pure JSON** — `cpc_role_classifier.py` now outputs only `{role, confidence}`. All descriptive text, signals, justifications, and prompts removed.
- **Phase 1 output upgraded** — Two new layers added: `core_function_generalized` (3 rewordings) and `cpc_terms` (6-10 CPC terms).
- **Evidence dedup + weight validation added** — `extracting_cpc.py` deduplicates evidence table, caps structural noise terms at weight 4, caps uncited weight ≥7 at 6.
- **Strategy reconstruction forced** — `phase1_runner.py` now ensures `classification_strategy` is always a structured dict with valid `anchor_split`.
- **Streamlit: Phase 1.2 tab added, Phase 2A-Layer tab removed**.
- **Phase 1.5 UI mapping centralized** — `ROLE_UI_MAP` in `streamlit_app.py` replaces inline conditionals.
- **Import and instantiation of CPCLayerDecomposer removed** from `search_cpc.py`.
- **`phase2a_layers` key removed** from final response dict in `search_cpc.py`.
- **Phase 1.5 debug logging added** to `search_cpc.py` — full role + TCR + fallback warning block.
