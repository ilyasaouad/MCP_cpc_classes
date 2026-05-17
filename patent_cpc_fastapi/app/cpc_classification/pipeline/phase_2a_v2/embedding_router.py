"""
embedding_router.py — Embedding similarity scorer for Phase 2A v2.

Scores CPC families against patent text using the knowledge graph's
sentence-transformer embeddings. Falls back to direct cosine similarity
against hardcoded CPC family descriptions when KG is unavailable.
Pure scoring — no decisions made here.
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Fallback CPC family descriptions for embedding scoring ──
_CPC_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "G10L": "Speech and voice analysis; speech recognition; speech synthesis; "
    "audio signal processing; acoustic processing; phoneme modeling; "
    "speaker recognition; speech coding; speech enhancement",
    "G06N": "Computing arrangements based on biological models; neural networks; "
    "machine learning; deep learning; training algorithms; inference; "
    "model optimization; self-supervised learning; clustering; transformers",
    "G06F": "Digital data processing; program control; information retrieval; "
    "database structures; file management; computing infrastructure",
    "G06T": "Image data processing; image analysis; image enhancement; "
    "image segmentation; computer vision; graphics rendering",
    "G06V": "Video understanding; scene analysis; object detection; motion analysis",
    "H04L": "Data communication networks; transmission of digital information; "
    "protocols; routing; switching; network architectures",
    "H04W": "Wireless communication; radio networks; base station control; "
    "channel allocation; mobile networks",
    "G05B": "Control systems; industrial automation; process control; robotics",
    "A61B": "Medical diagnosis; surgery; clinical instruments; patient monitoring",
    "B60W": "Vehicle control; autonomous driving; steering; braking; cruise control",
    "B60L": "Electric vehicle propulsion; battery management; charging",
    "B60R": "Vehicle fittings; signal systems; safety equipment",
    "F16J": "Sealing; pistons; cylinders; mechanical joints",
    "F16K": "Valves; taps; fluid flow control; pressure regulation",
    "E21B": "Earth drilling; borehole; well completion; oil and gas extraction",
    "G01N": "Chemical/physical analysis; measurement; testing; sensing; detection",
    "B01D": "Separation; filtration; membrane processes; purification",
}


class EmbeddingRouter:
    """
    Scores CPC families by semantic similarity to patent text.

    Input:  patent text representation + CPC family descriptions
    Output: {family: similarity_score} — NO decisions, only scores.
    """

    def __init__(self, kg_client=None):
        self.kg_client = kg_client
        self._fallback_embeddings: Dict[str, np.ndarray] = {}
        self._model = None

    def _get_model(self):
        """Lazy-load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("all-mpnet-base-v2")
            except Exception as e:
                logger.warning("Failed to load sentence transformer: %s", e)
        return self._model

    def _compute_fallback_embeddings(self):
        """Precompute embeddings for all CPC family descriptions."""
        model = self._get_model()
        if model is None:
            return
        texts = [
            _CPC_FAMILY_DESCRIPTIONS.get(f, f"CPC family {f}")
            for f in _CPC_FAMILY_DESCRIPTIONS
        ]
        embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for i, family in enumerate(_CPC_FAMILY_DESCRIPTIONS):
            self._fallback_embeddings[family] = embs[i]

    def _fallback_score(
        self,
        technical_object: str,
        core_function: str,
        candidate_families: Optional[list] = None,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Compute embedding similarity using hardcoded CPC descriptions.
        Used when KG is unavailable.
        """
        model = self._get_model()
        if model is None:
            logger.warning("Embedding router: no model available for fallback scoring")
            return {}

        if not self._fallback_embeddings:
            self._compute_fallback_embeddings()
            if not self._fallback_embeddings:
                return {}

        text = f"{technical_object} {core_function}".strip()
        if not text:
            return {}

        try:
            query_emb = model.encode(
                [text], show_progress_bar=False, convert_to_numpy=True
            )[0]
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return {}

            scores: Dict[str, float] = {}
            for family, cand_emb in self._fallback_embeddings.items():
                cand_norm = np.linalg.norm(cand_emb)
                if cand_norm == 0:
                    continue
                sim = float(np.dot(query_emb, cand_emb) / (query_norm * cand_norm))
                scores[family] = max(0.0, sim)

            # Normalize to [0, 1]
            if scores:
                max_val = max(scores.values())
                if max_val > 0:
                    for sym in scores:
                        scores[sym] /= max_val

            # Filter to candidate families if provided
            if candidate_families:
                candidate_set = set(candidate_families)
                scores = {f: s for f, s in scores.items() if f in candidate_set}

            # Return top-k
            ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            return dict(ranked)

        except Exception as e:
            logger.error("Fallback embedding scoring failed: %s", e)
            return {}

    def score_families(
        self,
        technical_object: str,
        core_function: str,
        candidate_families: Optional[list] = None,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Score CPC families by embedding similarity to patent text.

        Args:
            technical_object: What the invention IS
            core_function:    What the invention DOES
            candidate_families: Optional constraining list (e.g., ['G10L', 'G06N'])
            top_k: Max families to return

        Returns:
            {family: similarity_score} — all scores in [0, 1]
            Never returns empty dict — falls back to direct embedding scoring.
        """
        # ── Primary: KG-based scoring ──
        if self.kg_client and self.kg_client.is_available():
            text = f"{technical_object} {core_function}".strip()
            if not text:
                return {}

            try:
                raw_scores = self.kg_client.graph_semantic_score(text, top_k=top_k)

                # Filter to candidate families if provided
                if candidate_families:
                    candidate_set = set(candidate_families)
                    filtered = {f: s for f, s in raw_scores.items() if f in candidate_set}

                    # Supplement KG scores with fallback for families the KG didn't cover.
                    # This happens when a family has no embedding in the KG index (e.g. G06N).
                    uncovered = candidate_set - set(filtered)
                    if uncovered:
                        fallback = self._fallback_score(
                            technical_object, core_function,
                            candidate_families=list(uncovered), top_k=len(uncovered)
                        )
                        # Scale fallback scores down slightly so KG scores stay dominant
                        for fam, score in fallback.items():
                            if fam not in filtered:
                                filtered[fam] = round(score * 0.85, 4)
                        logger.info(
                            "Embedding router: KG missing %d families %s — filled via fallback",
                            len(uncovered), sorted(uncovered),
                        )

                    if filtered:
                        return filtered

                if raw_scores:
                    return raw_scores

                logger.warning(
                    "Embedding router: KG returned empty scores, using fallback"
                )

            except Exception as e:
                logger.error(
                    "Embedding router KG scoring failed: %s, using fallback", e
                )

        # ── Fallback: direct embedding against CPC descriptions ──
        logger.info("Embedding router: using fallback scoring (KG unavailable)")
        return self._fallback_score(
            technical_object, core_function, candidate_families, top_k
        )
