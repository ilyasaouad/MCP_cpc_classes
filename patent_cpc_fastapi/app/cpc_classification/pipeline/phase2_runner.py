import logging
import re
import math
from collections import Counter
from typing import Dict

from ..scoring.tfidf_scorer import score_candidates
from ..scoring.semantic_scorer import compute_semantic_scores, compute_rrf_fusion
from ..scoring.domain_booster import get_synonyms, get_expanded_terms
from ..utils.text_utils import normalize_word
from ..cpc_phase2d_anchor import Phase2DSubclassAnchor
from .phase_2a_v2 import Phase2AV2Router
from .phase2b_expander import Phase2BExpander

logger = logging.getLogger(__name__)


def _tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {normalize_word(w) for w in words if len(w) > 2}


def run_phase2(classifier, phase1, phase15_result=None, tcr_result=None):
    if phase15_result is None:
        phase15_result = {}
    if tcr_result is None:
        tcr_result = {}

    terms = phase1.get("terms", phase1.get("essential_terms", phase1.get("terms", [])))
    term_importance = {}
    for t in terms:
        if isinstance(t, dict):
            term = t.get("term", "").lower()
            importance = t.get("importance", 5)
            source = t.get("source_section", "unknown")
            section_weight = {
                "claims": 1.2,
                "summary": 1.0,
                "detailed_description": 0.9,
                "abstract": 0.6,
                "background": 0.2,
            }.get(source, 0.5)
            adjusted = min(round(importance * section_weight), 10)
            if term and len(term) > 3:
                term_importance[term] = max(term_importance.get(term, 0), adjusted)
        else:
            term = str(t).lower()
            if term and len(term) > 3:
                term_importance[term] = 5

    system_context = phase1.get("system_context", "").lower()
    core_function = phase1.get("core_function", "").lower()
    # Extract strategy from either object (new) or string (old) format
    strat_obj = phase1.get("classification_strategy", "")
    if isinstance(strat_obj, dict):
        strategy = strat_obj.get("strategy", "").lower()
    else:
        strategy = str(strat_obj).lower()

    # ── Phase 2A-Layer: Hardcoded fallback (LLM decomposer removed) ──
    # Phase 2D depends on the layer structure for anchor filtering,
    # so we provide a static layer definition instead of calling LLM.
    # layers={} forces Phase 2D to use family-router fallback which
    # reads phase2a_families (injected after Phase 2A v2 runs below).
    layer_result = {
        "primary_layer": "data_reasoning",
        "ai_role": "core_method",
        "layer_scores": {
            "application": 0.5,
            "pure_software": 0.3,
            "data_reasoning": 0.7,
            "interaction": 0.4,
            "control": 0.2,
        },
        "layers": {},
        "decomposition_mode": "hardcoded_fallback",
    }

    layers = layer_result.get("layers", {})
    primary_layer = layer_result.get("primary_layer", "unknown")
    layer_scores = layer_result.get("layer_scores", {})
    layer_explanation = {
        "layers": {},
        "primary_layer": primary_layer,
        "layer_scores": layer_scores,
        "is_cpc_authoritative": False,
        "decomposition_mode": "hardcoded_fallback",
    }

    logger.info(
        "Phase 2A-Layer (hardcoded fallback): primary=%s, layers=%s",
        primary_layer,
        list(layers.keys()),
    )
    logger.info(
        "Phase 2A-Layer scores: %s",
        {k: f"{v:.2f}" for k, v in layer_scores.items()},
    )

    # ─────────────────────────────
    # PHASE 2A v2: Authoritative CPC Family Router
    # ─────────────────────────────
    phase2a_v2_result: dict = {}
    final_cpc_families: list = []
    cpc_source: str = ""
    fallback_used: bool = False

    try:
        v2_router = Phase2AV2Router(classifier.knowledge_graph)
        phase2a_v2_result = v2_router.route(phase1, phase15_result, top_k=5)
        if phase2a_v2_result.get("family_names"):
            final_cpc_families = phase2a_v2_result["family_names"]
            cpc_source = "phase_2a_v2"
            logger.info(
                "Phase 2A v2: %d families — %s",
                len(final_cpc_families),
                final_cpc_families,
            )
        else:
            logger.error("Phase 2A v2 returned no families — fallback triggered")
            fallback_used = True
    except Exception as e:
        logger.error("Phase 2A v2 failed: %s — fallback triggered", e)
        fallback_used = True

    if not final_cpc_families:
        final_cpc_families = []
        fallback_used = True

    logger.info(
        "Phase 2A FINAL: source=%s, fallback=%s, families=%s",
        cpc_source or "none",
        fallback_used,
        final_cpc_families,
    )

    # Inject Phase 2A families into layer_result for Phase 2D fallback
    layer_result["phase2a_families"] = final_cpc_families
    layer_result["decomposition_mode"] = "hardcoded_fallback"

    # ─────────────────────────────
    # PHASE 2B: Hierarchical CPC Expansion
    # ─────────────────────────────
    phase2b_expansion_counts = {}

    # Extract family scores from Phase 2A v2 result
    family_scores: Dict[str, float] = {}
    for fam_entry in phase2a_v2_result.get("families", []):
        fam = fam_entry.get("family", "")
        score = fam_entry.get("score", 0.0)
        if fam:
            family_scores[fam] = score

    expander = Phase2BExpander(
        knowledge_graph=classifier.knowledge_graph,
        xml_parser=classifier.xml_parser,
    )
    phase2b_result = expander.expand(
        families=final_cpc_families,
        family_scores=family_scores,
        phase1=phase1,
        max_subgroups=500,
    )

    # ── Consistently read candidates from Phase 2B result ──
    all_subgroups = phase2b_result.get("flat_candidates", phase2b_result.get("expanded_details", []))
    expanded_cpcs = [sg.get("symbol", sg.get("code", "")) for sg in all_subgroups]
    phase2b_candidate_count = len(all_subgroups)
    
    phase2b_source = phase2b_result["source"]
    phase2b_fallback = phase2b_result["fallback_used"]
    phase2b_family_counts = phase2b_result["family_counts"]
    phase2b_family_expansions = phase2b_result["family_expansions"]
    phase2b_pruned_count = phase2b_result["pruned_count"]
    phase2b_raw_family_counts = phase2b_result.get("raw_family_counts", {})
    phase2b_proportional_caps = phase2b_result.get("proportional_caps", {})

    logger.info(
        "Phase 2B: %d subgroups expanded (pruned %d) from %d families (source=%s, fallback=%s)",
        phase2b_candidate_count,
        phase2b_pruned_count,
        len(final_cpc_families),
        phase2b_source,
        phase2b_fallback,
    )

    # Validation guard: Phase 2C must have candidates
    if phase2b_candidate_count == 0:
        logger.error(
            "Phase 2B output empty or misformatted — Phase 2C will receive 0 candidates"
        )

    if phase2b_candidate_count > 0:
        for sg in all_subgroups:
            sym = sg.get("symbol", "")
            prefix = sym[:4] if len(sym) >= 4 else sym[:3]
            phase2b_expansion_counts[prefix] = (
                phase2b_expansion_counts.get(prefix, 0) + 1
            )

        logger.info(
            "Phase 2B: Expanded %d total subgroups across %d families: %s",
            phase2b_candidate_count,
            len(phase2b_expansion_counts),
            {k: v for k, v in sorted(phase2b_expansion_counts.items())},
        )

    # ─────────────────────────────
    # PHASE 2C: Scoring Expanded Subgroups
    # ─────────────────────────────
    candidates = []
    all_candidates = []
    score_margin = 0.0
    confidence_level = "medium"
    phase2c_final_count = 0

    if all_subgroups:
        logger.info(
            "Phase 2C: Received %d candidates from Phase 2B (first: %s)",
            len(all_subgroups),
            all_subgroups[0].get("symbol", "unknown") if all_subgroups else "none",
        )
        title_count = len(all_subgroups)
        # ─────────────────────────────
        # BM25 Scoring (Replacing TF-IDF)
        # ─────────────────────────────
        # Extract cpc_terms with fallback for injection
        cpc_terms = phase1.get("cpc_terms", [])
        if not cpc_terms:
            fallback = []
            cfp = phase1.get("core_function_precise", "")
            if cfp: fallback.append(cfp)
            cfg = phase1.get("core_function_generalized", [])
            if cfg: fallback.extend(cfg)
            cpc_terms = fallback
            
        logger.info("[Phase 2C] Injected %d cpc_terms for BM25: %s", len(cpc_terms), cpc_terms)

        scored = score_candidates(
            all_subgroups=all_subgroups,
            term_importance=term_importance,
            system_context=system_context,
            core_function=core_function,
            strategy=strategy,
            phase2a_result=layer_result,
            negative_signals=phase1.get("negative_signals", []),
            cpc_terms=cpc_terms,
            core_function_precise=phase1.get("core_function_precise", ""),
            core_function_generalized=phase1.get("core_function_generalized", []),
            evidence_table=phase1.get("evidence_table", []),
        )

        scored.sort(key=lambda x: -x[0])

        # ── BM25 + Embedding → RRF fusion ──
        sem_scores = {}
        if classifier.knowledge_graph and classifier.knowledge_graph.embeddings:
            sem_scores = compute_semantic_scores(
                [sg for _, sg, _ in scored],
                patent_text="",  # not used when core_function is provided
                knowledge_graph=classifier.knowledge_graph,
                core_function=phase1.get("core_function", ""),
                evidence_table=phase1.get("evidence_table", []),
            )

        if sem_scores:
            scored = compute_rrf_fusion(scored, sem_scores)
            logger.info(
                "Phase 2C: RRF fusion applied (BM25 + embedding) for %d candidates",
                len(scored),
            )
        else:
            logger.info(
                "Phase 2C: No KG embeddings available — BM25-only ranking kept"
            )

        # ── Transition to Phase 2D: Build candidates + trace scoring split ──
        all_candidates = []
        if scored:
            scores = [s[0] for s in scored]
            max_val = max(scores) if scores else 1.0
            median_val = sorted(scores)[len(scores) // 2] if len(scores) > 1 else max_val
            denom = max(max_val + median_val * 0.5, 1.0)
            
            # Re-calculate margin and confidence
            if len(scores) >= 2:
                score_margin = round((scores[0] - scores[1]) / denom, 6)
                if score_margin > 0.3:
                    confidence_level = "high"
                elif score_margin < 0.1:
                    confidence_level = "low"
                else:
                    confidence_level = "medium"
            
            for score, sg, _ in scored:
                normalized_score = min(score / denom, 1.0)
                all_candidates.append({
                    "symbol": sg.get("symbol", ""),
                    "title": sg.get("title", ""),
                    "level": sg.get("level", 0),
                    "score": round(normalized_score, 6),
                    "full_context": sg.get("full_context", sg.get("title", "")),
                })
        
        # EXACT PRINT STATEMENT REQUESTED BY USER
        candidates = all_candidates # Rename to match user snippet
        non_zero = sum(1 for c in candidates if c.get('score', 0) > 0)
        print(f"Phase 2C Scoring Split: {non_zero} non-zero / {len(candidates) - non_zero} zero-score candidates")

        phase2c_final_count = len(all_candidates)
        logger.info(
            "Phase 2C: Scored %d candidates. Passing all to Phase 2D.",
            phase2c_final_count,
        )

        # ── Family-level score normalization ──
        # Rescale each family's scores so its best candidate = Phase 2A relevance score.
        # Without this, high-subgroup families (G10L) dominate Phase 3 purely by volume,
        # creating a 6× gap vs low-subgroup families (G06N) regardless of actual relevance.
        if family_scores and all_candidates:
            fam_groups: Dict[str, list] = {}
            for i, c in enumerate(all_candidates):
                fam = c.get("symbol", "")[:4]
                fam_groups.setdefault(fam, []).append(i)
            for fam, indices in fam_groups.items():
                max_score = max(all_candidates[i].get("score", 0) for i in indices)
                p2a_weight = family_scores.get(fam, 0.0)
                if max_score > 0 and p2a_weight > 0:
                    for i in indices:
                        all_candidates[i]["score"] = round(
                            (all_candidates[i]["score"] / max_score) * p2a_weight, 6
                        )
            logger.info(
                "Phase 2C: Family normalization applied — %d families rescaled to Phase 2A weights",
                len(fam_groups),
            )

    _TOP_N = 50
    phase2d_result = {}
    find_until_full_log = []
    candidates = []
    try:
        phase2d_result = Phase2DSubclassAnchor().filter(
            candidates=all_candidates,
            layer_result=layer_result,
            max_result=_TOP_N,
        )
        candidates = phase2d_result.get("purified_candidates", [])
        find_until_full_log = [{
            "depth": len(all_candidates),
            "survivors_found": len(candidates),
            "deep_search_triggered": False,
        }]
        logger.info(
            "Phase 2D: Top-%d filter — Input=%d → Kept=%d → Discarded=%d",
            _TOP_N,
            len(all_candidates),
            phase2d_result.get("kept_count", 0),
            phase2d_result.get("discarded_count", 0),
        )
    except Exception as e:
        logger.warning("Phase 2D filter failed: %s", e)

    phase2_dict = {
        "phase2a_layers": layer_result,
        "layer_explanation": layer_explanation,
        "phase2a_v2": phase2a_v2_result,
        "final_cpc_families": final_cpc_families,
        "cpc_source": cpc_source,
        "fallback_used": fallback_used,
        "codes": [node["symbol"] for node in candidates] if candidates else [],
        "reasoning": (
            f"Phase 2A v2: {len(final_cpc_families)} families via "
            f"embedding+KG+anchor fusion. "
            f"Layer explanation: primary={primary_layer}. "
            f"Phase 2B expanded to {phase2b_candidate_count} candidates, "
            f"Phase 2C scored down to {phase2c_final_count}."
        ),
        "score_margin": round(score_margin, 6),
        "confidence_level": confidence_level,
        "phase2a_source": cpc_source,
        "phase2a_reasoning": (
            f"Authoritative CPC families from Phase 2A v2. "
            f"{len(final_cpc_families)} families. "
            f"Fallback used: {fallback_used}."
        ),
        "phase2a_families": final_cpc_families,
        "phase2a_primary_layer": primary_layer,
        "phase2a_layer_scores": {k: round(v, 4) for k, v in layer_scores.items()},
        "phase2b_candidate_count": phase2b_candidate_count,
        "phase2b_expansion_counts": phase2b_expansion_counts,
        "phase2b_source": phase2b_source,
        "phase2b_fallback_used": phase2b_fallback,
        "phase2b_family_counts": phase2b_family_counts,
        "phase2b_family_expansions": phase2b_family_expansions,
        "phase2b_pruned_count": phase2b_pruned_count,
        "phase2b_raw_family_counts": phase2b_raw_family_counts,
        "phase2b_proportional_caps": phase2b_proportional_caps,
        "phase2b_output_type": "flat+structured",
        "phase2c_input_size": phase2b_candidate_count,
        "phase2c_final_count": phase2c_final_count,
        "phase2d_anchor_set": phase2d_result.get("anchor_set", []),
        "phase2d_anchor_source": phase2d_result.get("anchor_source", []),
        "phase2d_kept_count": phase2d_result.get("kept_count", 0),
        "phase2d_discarded_count": phase2d_result.get("discarded_count", 0),
        "phase2d_discard_log": phase2d_result.get("discard_log", []),
        "phase2d_find_until_full": find_until_full_log,
        "phase2c_total_scored": phase2c_final_count,
        "phase2a_v2_result": phase2a_v2_result,
    }

    return candidates, all_candidates, phase2_dict, phase2d_result, find_until_full_log
