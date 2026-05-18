# patent_cpc_API — File Map & Descriptions

## Structure Map

```
patent_cpc_API/
│
├── main.py                          # App entry point — startup, KG init, engine wiring
│
├── config/
│   ├── pipeline_config.yaml         # All phase thresholds, weights, model names
│   └── domains_schema.json          # Domain→CPC mappings, invalid patterns, visual families
│
├── core/
│   ├── engine.py                    # Pipeline orchestrator — runs phases 1A→5B in sequence
│   ├── state_manager.py             # PipelineState dataclass — shared object across all phases
│   └── knowledge_graph.py           # Singleton wrapper — loads CPCKnowledgeGraph once at startup
│
├── shared/
│   ├── base_phase.py                # BasePhase ABC — every phase inherits run(state) → dict
│   └── cpc_filters.py               # is_allocatable(), filter_allocatable() — used in 2B and 3A
│
├── api/
│   ├── models.py                    # Pydantic request/response schemas
│   └── router.py                    # FastAPI routes: POST /classify, GET /health
│
├── phase1a_extraction/
│   ├── extractor.py                 # LLM multi-pass extraction of terms, core function, domain signals
│   └── prompts/__init__.py          # Re-exports pass A/B/C/D prompts + reconciliation
│
├── phase1b_audit/
│   ├── auditor.py                   # Claim-to-domain forensic audit — builds anchor_families + kill_log
│   └── prompts/__init__.py          # Re-exports phase1_2_audit_prompt
│
├── phase1c_character/
│   └── characterizer.py             # TCR score + role classification (CORE_TECH/SYSTEM/APPLICATION)
│
├── phase2a_routing/
│   └── router.py                    # Family-level routing: embedding + KG + anchor blend → family_scores
│
├── phase2b_expansion/
│   └── expander.py                  # KG/XML subgroup expansion + non-allocatable pre-filter
│
├── phase2c_scoring/
│   └── scorer.py                    # BM25 + semantic RRF scoring + family-level normalisation
│
├── phase2d_filter/
│   └── filter.py                    # Top-N filter + anchor promotion + preserves all_raw_candidates pool
│
├── phase3a_constraints/
│   └── constraints.py               # Decision-tree rule layer: domain boosts, invalid-class penalties
│
├── phase3b_validation/
│   └── validator.py                 # Cross-domain validation: anchor boosts, anti-collapse rules
│
├── phase4a_consolidation/
│   ├── consolidator.py              # Clusters candidates into hypotheses (family, coherence, support_weight)
│   └── prompts/__init__.py          # Re-exports phase4_sanity_check_prompt
│
├── phase4b_resolution/
│   ├── resolver.py                  # Scores hypotheses on 3 axes → picks winner + Tri-Pillar output
│   └── prompts/__init__.py          # Re-exports phase35_tiebreaker_prompt
│
├── phase5a_consistency/
│   ├── consistency.py               # LLM sanity check on Phase 4B primary before final labeling
│   └── prompts/__init__.py          # Re-exports phase7_consistency_prompt
│
└── phase5b_labeling/
    ├── labeler.py                   # Role labeling (CORE/SUPPORT/CONTEXT) + LLM justification text
    └── prompts/__init__.py          # Re-exports phase8_labeling_prompt
```

---

## Phase Groups

### Group 1 — Signal Extraction
Reads the patent text. No CPC database access yet.

| Phase | File | Method | What it does |
|-------|------|--------|--------------|
| 1A | `phase1a_extraction/extractor.py` | LLM | Extracts technical terms, core function, technical object, inventive step, domain signals |
| 1B | `phase1b_audit/auditor.py` | LLM | Audits which CPC families are actually evidenced in the claims → `anchor_families`, `kill_log` |
| 1C | `phase1c_character/characterizer.py` | LLM + Deterministic | Computes TCR (Technology Computation Ratio) and classifies the patent's role |

### Group 2 — Candidate Search
Queries the CPC database. Produces up to 500 scored subgroup candidates.

| Phase | File | Method | What it does |
|-------|------|--------|--------------|
| 2A | `phase2a_routing/router.py` | Embedding + KG + Anchors | Ranks CPC families (4-char) → `family_scores` used later for normalisation |
| 2B | `phase2b_expansion/expander.py` | Deterministic | Expands each family into subgroups via KG hierarchy + XML titles |
| 2C | `phase2c_scoring/scorer.py` | BM25 + Semantic + RRF | Scores every subgroup, then normalises per-family so volume doesn't dominate |
| 2D | `phase2d_filter/filter.py` | Deterministic | Keeps top-N candidates; preserves full raw pool for Phase 4B title lookup |

### Group 3 — Constraint Validation
Applies deterministic rules. No new database queries.

| Phase | File | Method | What it does |
|-------|------|--------|--------------|
| 3A | `phase3a_constraints/constraints.py` | Deterministic | Decision-tree: domain boosts, invalid-class penalties, functional boosting |
| 3B | `phase3b_validation/validator.py` | Deterministic | Cross-domain validation: anchor confirmed → boost; no context → penalty |

### Group 4 — Hypothesis
Clusters candidates into competing hypotheses, then picks one winner.

| Phase | File | Method | What it does |
|-------|------|--------|--------------|
| 4A | `phase4a_consolidation/consolidator.py` | Deterministic | Groups candidates by family into hypotheses with coherence + support_weight |
| 4B | `phase4b_resolution/resolver.py` | Deterministic | Scores hypotheses on phase4 (0.5) + functional_alignment (0.3) + technical_coverage (0.2) → Tri-Pillar |

### Group 5 — Output
Validates and labels the final recommendation.

| Phase | File | Method | What it does |
|-------|------|--------|--------------|
| 5A | `phase5a_consistency/consistency.py` | LLM | Sanity-checks the Phase 4B primary against the full candidate pool |
| 5B | `phase5b_labeling/labeler.py` | LLM | Assigns CORE/SUPPORT/CONTEXT roles + writes professional justification text |

---

## Core Files

### `core/state_manager.py` — PipelineState
Single dataclass that flows through every phase. Each phase reads what it needs
and writes its output back as a dict (e.g. `state.phase2c = scorer.run(state)`).
No global variables — all inter-phase communication goes through this object.

### `core/engine.py` — CPCPipelineEngine
Instantiates all 13 phase runners at startup. On each request, creates a fresh
`PipelineState`, runs the sequence 1A→5B, records errors/warnings, and returns
`state.to_dict()`. Phase failures are caught and logged — the pipeline always
returns whatever it completed.

### `core/knowledge_graph.py` — KnowledgeGraphSingleton
Wraps the `CPCKnowledgeGraph` as a singleton. Loads from disk cache on first
`get()` call; rebuilds from XML if cache is stale. All phases that need the KG
receive the same object instance.

### `shared/cpc_filters.py` — Allocatable guards
`is_allocatable(symbol)` returns `True` only when the symbol contains `/` AND
does not match the cross-reference pattern `^[A-Z]\d{2}[A-Z]\d{4}/`.
Applied in Phase 2B (after expansion) and Phase 3A (before constraint rules).

---

## Config Files

### `config/pipeline_config.yaml`
Every numeric threshold in the pipeline is defined here. Phase code reads its
own section (e.g. `phase2c_scoring.rrf_k`) via the `cfg` dict injected at
startup. Changing a threshold never requires touching phase code.

### `config/domains_schema.json`
- `domain_to_cpc` — maps domain keywords to CPC family lists
- `invalid_patterns` — per-domain CPC prefixes that should be penalised
- `visual_families` — families excluded from Pillar 3 when primary is G10L
- `cross_ref_pattern` — regex to detect non-allocatable cross-reference codes
