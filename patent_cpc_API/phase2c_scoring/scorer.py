"""
phase2c_scoring/scorer.py — Phase 2C: Hybrid Scoring (BM25 + Embeddings + RRF).

Scores every Phase 2B candidate against the Phase 1A query terms using:
  1. BM25 lexical retrieval
  2. Semantic embedding cosine similarity
  3. Reciprocal Rank Fusion (RRF) to merge both lists

Then applies family-level normalisation so each family's best score
equals its Phase 2A routing weight — preventing high-subgroup families
from dominating purely by volume.

Old equivalent: scoring block in patent_cpc_fastapi/pipeline/phase2_runner.py

score_candidates() signature (7 required args):
  (all_subgroups, term_importance, system_context, core_function,
   strategy, phase2a_result, negative_signals, ...)
  Returns: List[Tuple[float, Dict, int]]  — (score, subgroup, match_count)

compute_semantic_scores() signature:
  (candidates, patent_text, knowledge_graph, core_function="", evidence_table=None)
  Returns: Dict[str, float]  — symbol → score

compute_rrf_fusion(bm25_ranked, sem_scores)  — no k arg
  Returns: List[Tuple[float, Dict, int]]
"""

import logging
from typing import Any, Dict, List, Tuple

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase2CScorer(BasePhase):
    """
    BM25 + embedding + RRF scoring with family-level normalisation.

    Config keys used:
      family_normalization (bool) rescale per-family to Phase 2A weight
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.scoring.tfidf_scorer import score_candidates
            from cpc_classification.scoring.semantic_scorer import (
                compute_semantic_scores,
                compute_rrf_fusion,
            )
        except ImportError as exc:
            state.record_error("2c", f"Import failed: {exc}")
            return {"scored_candidates": []}

        phase1 = state.phase1a
        candidates: List[Dict[str, Any]] = state.phase2b.get("expanded_candidates", [])
        family_scores: Dict[str, float] = state.phase2a.get("family_scores", {})

        if not candidates:
            state.record_warning("2c", "No Phase 2B candidates to score")
            return {"scored_candidates": []}

        do_normalise = self.cfg.get("family_normalization", True)

        # Build term importance from Phase 1A terms
        term_importance: Dict[str, int] = {}
        for t in phase1.get("terms", phase1.get("essential_terms", [])):
            if isinstance(t, dict):
                term = t.get("term", "").lower()
                imp = t.get("importance", 5)
                if term and len(term) > 3:
                    term_importance[term] = max(term_importance.get(term, 0), imp)
            else:
                term = str(t).lower()
                if term and len(term) > 3:
                    term_importance[term] = 5

        core_function = phase1.get("core_function", "")
        system_context = phase1.get("system_context", "")
        negative_signals = phase1.get("negative_signals", [])
        strategy = phase1.get("strategy", "LLM_EXTRACTION")
        phase2a_result = state.phase2a

        # BM25 scoring — returns List[Tuple[float, Dict, int]]
        bm25_ranked: List[Tuple[float, Dict, int]] = []
        try:
            bm25_ranked = score_candidates(
                candidates,
                term_importance,
                system_context,
                core_function,
                strategy,
                phase2a_result,
                negative_signals,
                core_function_precise=core_function,
            )
        except Exception as exc:
            state.record_warning("2c", f"BM25 scoring failed: {exc}")
            bm25_ranked = [(0.0, c, 0) for c in candidates]

        # Semantic scoring — returns Dict[str, float] (symbol → score)
        sem_scores: Dict[str, float] = {}
        try:
            query_text = core_function or " ".join(list(term_importance.keys())[:20])
            sem_scores = compute_semantic_scores(
                candidates,
                query_text,
                self.kg,
                core_function=core_function,
            )
        except Exception as exc:
            state.record_warning("2c", f"Semantic scoring failed: {exc}")

        # RRF fusion — returns List[Tuple[float, Dict, int]]
        fused: List[Tuple[float, Dict, int]] = bm25_ranked
        if sem_scores:
            try:
                fused = compute_rrf_fusion(bm25_ranked, sem_scores)
            except Exception as exc:
                state.record_warning("2c", f"RRF fusion failed: {exc}")

        # Convert tuples → dicts and attach score
        all_candidates: List[Dict[str, Any]] = []
        for score, subgroup, match_count in fused:
            entry = dict(subgroup)
            entry["score"] = round(float(score), 6)
            entry["match_count"] = match_count
            all_candidates.append(entry)

        # Family-level normalisation: best in family = Phase 2A weight
        if do_normalise and family_scores and all_candidates:
            fam_groups: Dict[str, List[int]] = {}
            for i, c in enumerate(all_candidates):
                fam = c.get("symbol", "")[:4]
                fam_groups.setdefault(fam, []).append(i)

            for fam, indices in fam_groups.items():
                max_score = max(all_candidates[i].get("score", 0) for i in indices)
                p2a_weight = family_scores.get(fam, 0.0)
                if max_score > 0 and p2a_weight > 0:
                    for i in indices:
                        raw = all_candidates[i].get("score", 0)
                        all_candidates[i]["score"] = round(
                            (raw / max_score) * p2a_weight, 6
                        )

        all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(
            "Phase 2C: %d candidates scored (normalise=%s)", len(all_candidates), do_normalise
        )
        return {"scored_candidates": all_candidates}
