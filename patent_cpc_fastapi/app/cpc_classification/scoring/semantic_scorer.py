"""semantic_scorer.py — Semantic scoring via Knowledge Graph embeddings."""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_semantic_scores(
    candidates: List[Dict[str, Any]],
    patent_text: str,
    knowledge_graph: Any,
) -> Dict[str, float]:
    """Compute semantic similarity scores using KG embeddings.

    Returns dict mapping candidate symbol -> normalised similarity [0, 1].
    Falls back to empty dict if KG is unavailable.
    """
    sem_scores: Dict[str, float] = {}
    if not knowledge_graph or not knowledge_graph.embeddings:
        return sem_scores

    try:
        model = knowledge_graph._get_model()
        query_emb = model.encode(
            [patent_text], show_progress_bar=False, convert_to_numpy=True
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
            "Semantic scores computed for %d/%d candidates",
            len(sem_scores),
            len(candidates),
        )
    except Exception as e:
        logger.warning("Semantic scoring failed, falling back to TF-IDF: %s", e)

    return sem_scores
