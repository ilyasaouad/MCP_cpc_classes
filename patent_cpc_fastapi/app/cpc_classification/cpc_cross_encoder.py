"""
cpc_cross_encoder.py - Cross-Encoder Reranker for CPC Retrieval

Reranks top-K CPC candidates using a cross-encoder that jointly attends
over [query, cpc_title] pairs for higher accuracy than bi-encoder alone.

Usage:
    from cpc_cross_encoder import CPCCrossEncoder

    reranker = CPCCrossEncoder()

    # candidates = [(symbol, title), ...] from BM25/KG
    results = reranker.rerank(
        query="neural network quantization",
        candidates=[("G06N3/047", "Quantisation of neural network"), ...],
        top_k=20
    )
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CPCCrossEncoder:
    """
    Cross-encoder reranker for CPC candidate scoring.

    Uses a lightweight cross-encoder model to score (query, cpc_title) pairs.
    Much more accurate than bi-encoder cosine for ranking because it can
    attend to interactions between query and title tokens.
    """

    # Default lightweight model (20MB, fast on CPU)
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Alternative even smaller model (8MB)
    SMALL_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder.

        Args:
            model_name: HuggingFace model name. If None, uses default.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", self.model_name)
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.error("Failed to load cross-encoder: %s", e)
            raise

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 20,
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """
        Rerank CPC candidates using cross-encoder.

        Args:
            query: Patent text or query string
            candidates: List of (symbol, title) tuples from BM25/KG
            top_k: Number of top results to return
            batch_size: Batch size for inference

        Returns:
            List of (symbol, cross_encoder_score) tuples, sorted by score descending
        """
        if not candidates:
            return []

        if not self.model:
            self._load_model()

        # Prepare pairs: [(query, title1), (query, title2), ...]
        pairs = [(query, title) for _, title in candidates]
        symbols = [symbol for symbol, _ in candidates]

        logger.debug("Reranking %d candidates with cross-encoder", len(pairs))

        # Score all pairs
        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )

        # Combine symbols with scores
        results = list(zip(symbols, scores))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return results[:top_k]

    def rerank_from_kg_results(
        self,
        query: str,
        kg_results: List[Tuple[str, float]],
        kg_instance,  # CPCKnowledgeGraph instance
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Convenience method: rerank results from knowledge graph.

        Args:
            query: Patent text
            kg_results: List of (symbol, kg_score) from knowledge graph
            kg_instance: CPCKnowledgeGraph instance to look up titles
            top_k: Number of results to return

        Returns:
            Reranked list of (symbol, cross_encoder_score)
        """
        # Get titles for symbols
        candidates = []
        for symbol, _ in kg_results:
            title = self._get_title(symbol, kg_instance)
            if title:
                candidates.append((symbol, title))

        return self.rerank(query, candidates, top_k=top_k)

    def _get_title(self, symbol: str, kg_instance) -> str:
        """Get title for a CPC symbol from knowledge graph."""
        try:
            node_data = kg_instance.graph.nodes.get(symbol, {})
            return node_data.get("title", "")
        except Exception:
            return ""


def quick_rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Quick function to rerank candidates without instantiating class.

    Args:
        query: Search query
        candidates: List of (symbol, title) tuples
        top_k: Number of results

    Returns:
        Reranked results
    """
    reranker = CPCCrossEncoder()
    return reranker.rerank(query, candidates, top_k=top_k)
