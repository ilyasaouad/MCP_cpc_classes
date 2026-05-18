"""
phase2c_scoring/scorer.py â€” Phase 2C: Hybrid Scoring (BM25 + Embeddings + RRF).

Scores every Phase 2B candidate against the Phase 1A query terms using:
  1. BM25 lexical retrieval
  2. Semantic embedding cosine similarity
  3. Reciprocal Rank Fusion (RRF) to merge both lists

Then applies family-level normalisation so each family's best score
equals its Phase 2A routing weight â€” preventing high-subgroup families
from dominating purely by volume.

Old equivalent: scoring block in patent_cpc_fastapi/pipeline/phase2_runner.py
"""

import logging
from typing import Any, Dict, List

from ..shared.base_phase import BasePhase
from ..core.state_manager import PipelineState

logger = logging.getLogger(__name__)


class Phase2CScorer(BasePhase):
    """
    BM25 + embedding + RRF scoring with family-level normalisation.

    Config keys used:
      rrf_k               (int)   RRF constant k (default 60)
      family_normalization (bool) rescale per-family to Phase 2A weight
    """

    def run(self, state: PipelineState) -> Dict[str, Any]:
        try:
            from cpc_classification.scoring.tfidf_scorer import (
                score_candidates,
            )
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

        rrf_k = self.cfg.get("rrf_k", 60)
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

        try:
            bm25_scored = score_candidates(candidates, term_importance)
        except Exception as exc:
            state.record_warning("2c", f"BM25 scoring failed: {exc}")
            bm25_scored = candidates

        try:
            query_text = " ".join(term_importance.keys())
            sem_scored = compute_semantic_scores(candidates, query_text, self.kg)
            all_candidates = compute_rrf_fusion(bm25_scored, sem_scored, k=rrf_k)
        except Exception as exc:
            state.record_warning("2c", f"Semantic/RRF failed: {exc}")
            all_candidates = bm25_scored

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
        logger.info("Phase 2C: %d candidates scored (RRF k=%d, normalise=%s)",
                    len(all_candidates), rrf_k, do_normalise)
        return {"scored_candidates": all_candidates}
