"""semantic_scorer.py — Semantic scoring + RRF fusion for Phase 2C."""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard RRF constant — 60 is the well-established default from the original paper.
_RRF_K = 60


def _build_query(
    core_function: str,
    evidence_table: List[Dict[str, Any]],
) -> str:
    """
    Build the embedding query from Phase 1's high-signal output.
    Uses core_function + high-weight evidence terms (weight >= 6).
    This is deliberately compact — embedding long text dilutes the signal.
    """
    parts = [core_function.strip()] if core_function.strip() else []
    for row in evidence_table or []:
        if isinstance(row, dict) and row.get("weight", 0) >= 6:
            term = row.get("term", "").strip()
            if term:
                parts.append(term)
    return " ".join(parts[:12])  # cap at 12 parts to keep query focused


def compute_semantic_scores(
    candidates: List[Dict[str, Any]],
    patent_text: str,
    knowledge_graph: Any,
    core_function: str = "",
    evidence_table: List[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute cosine similarity between the patent query and each candidate's
    pre-computed KG embedding.

    Query priority:
      1. core_function + evidence_table terms (if provided) — focused, high-signal
      2. patent_text fallback (legacy — full technical_object + core_function)

    Returns dict: symbol → normalised similarity [0, 1].
    Falls back to empty dict if KG has no embeddings.
    """
    sem_scores: Dict[str, float] = {}
    if not knowledge_graph or not knowledge_graph.embeddings:
        return sem_scores

    try:
        query = _build_query(core_function, evidence_table or [])
        if not query:
            query = patent_text  # legacy fallback

        model = knowledge_graph._get_model()
        query_emb = model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )[0]
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return sem_scores

        for c in candidates:
            symbol = c.get("symbol", "")
            if symbol not in knowledge_graph.embeddings:
                continue
            cand_emb = knowledge_graph.embeddings[symbol]
            cand_norm = np.linalg.norm(cand_emb)
            if cand_norm == 0:
                continue
            sim = float(np.dot(query_emb, cand_emb) / (query_norm * cand_norm))
            sem_scores[symbol] = max(0.0, sim)

        if sem_scores:
            max_val = max(sem_scores.values())
            if max_val > 0:
                for sym in sem_scores:
                    sem_scores[sym] /= max_val

        logger.info(
            "Semantic scores computed for %d/%d candidates (query: '%s...')",
            len(sem_scores),
            len(candidates),
            query[:80],
        )
    except Exception as e:
        logger.warning("Semantic scoring failed: %s", e)

    return sem_scores


def compute_rrf_fusion(
    bm25_ranked: List[Tuple[float, Dict[str, Any], int]],
    sem_scores: Dict[str, float],
) -> List[Tuple[float, Dict[str, Any], int]]:
    """
    Reciprocal Rank Fusion of BM25 and semantic rankings.

    RRF score = 1/(k + rank_bm25) + 1/(k + rank_sem)

    Using rank positions only avoids the incompatible-scale problem that
    makes a weighted sum unreliable when one scorer returns very different
    value distributions than the other.

    Returns the same (score, subgroup, match_count) format as score_candidates(),
    sorted descending by RRF score, normalised to [0, 1].
    """
    if not sem_scores:
        # No semantic scores available — return BM25 ranking unchanged
        return bm25_ranked

    # Build BM25 rank lookup: symbol → 1-based rank
    bm25_ranks: Dict[str, int] = {
        item[1].get("symbol", ""): rank + 1
        for rank, item in enumerate(bm25_ranked)
    }

    # Build semantic rank lookup: symbol → 1-based rank (sorted by score desc)
    sem_ranked_symbols = sorted(sem_scores, key=sem_scores.get, reverse=True)
    sem_ranks: Dict[str, int] = {
        sym: rank + 1 for rank, sym in enumerate(sem_ranked_symbols)
    }

    # Total number of candidates — used as default rank for missing entries
    n = len(bm25_ranked)

    rrf_scores: Dict[str, float] = {}
    for _, sg, _ in bm25_ranked:
        sym = sg.get("symbol", "")
        r_bm25 = bm25_ranks.get(sym, n)
        r_sem = sem_ranks.get(sym, n)
        rrf_scores[sym] = 1.0 / (_RRF_K + r_bm25) + 1.0 / (_RRF_K + r_sem)

    # Normalise to [0, 1]
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    if max_rrf > 0:
        rrf_scores = {sym: v / max_rrf for sym, v in rrf_scores.items()}

    # Rebuild result list preserving (score, subgroup, match_count) format
    fused = [
        (round(rrf_scores.get(sg.get("symbol", ""), 0.0), 6), sg, mc)
        for _, sg, mc in bm25_ranked
    ]
    fused.sort(key=lambda x: -x[0])

    logger.info(
        "RRF fusion complete: %d candidates fused (BM25 + semantic, k=%d)",
        len(fused),
        _RRF_K,
    )
    return fused
