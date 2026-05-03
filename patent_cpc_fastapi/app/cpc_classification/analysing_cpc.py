from typing import List, Dict, Optional
from search_core.ollama_client import OllamaClient


def cosine(a, b):
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5

    return dot / (na * nb + 1e-9)


def weighted_cosine_similarity(
    node_vec: list, term_vecs: List[list], term_importances: List[float]
) -> float:
    """
    Compute weighted similarity score.

    Uses a combination of:
    - Max similarity (best matching term)
    - Weighted average (all terms weighted by importance)
    """
    if not term_vecs or not node_vec:
        return 0.0

    similarities = [cosine(node_vec, tv) for tv in term_vecs]

    # Max similarity (catches strong single-term matches)
    max_sim = max(similarities) if similarities else 0.0

    # Weighted average (captures aggregate relevance)
    if term_importances and sum(term_importances) > 0:
        weighted_avg = sum(s * w for s, w in zip(similarities, term_importances)) / sum(
            term_importances
        )
    else:
        weighted_avg = sum(similarities) / len(similarities) if similarities else 0.0

    # Combine: 60% weighted average + 40% max
    # This rewards codes that match multiple important terms,
    # while still capturing strong single-term matches
    return 0.6 * weighted_avg + 0.4 * max_sim


class CPCAnalyzer:
    """
    Phase 3:
    Score CPC candidates using embeddings with caching and weighted scoring.
    """

    def __init__(self, llm: OllamaClient):
        self.llm = llm
        # Persistent cache for term embeddings across requests
        self._term_cache: Dict[str, list] = {}
        # Persistent cache for title embeddings across requests
        self._title_cache: Dict[str, list] = {}

    def _get_term_embedding(self, term: str) -> Optional[list]:
        """Get term embedding, using cache if available."""
        if term in self._term_cache:
            return self._term_cache[term]

        try:
            vec = self.llm.embeddings(term)
            self._term_cache[term] = vec
            return vec
        except Exception:
            return None

    def _get_title_embedding(self, title: str) -> Optional[list]:
        """Get title embedding, using cache if available."""
        if not title:
            return None

        if title in self._title_cache:
            return self._title_cache[title]

        try:
            vec = self.llm.embeddings(title)
            self._title_cache[title] = vec
            return vec
        except Exception:
            return None

    def score_nodes(self, nodes: List[dict], terms: List[dict]) -> List[dict]:
        """
        Score CPC candidates using embeddings.

        Args:
            nodes: List of candidate dicts with 'symbol', 'title', etc.
            terms: List of term dicts with 'term' and 'importance'

        Returns:
            Nodes sorted by score descending.
        """
        # Extract term texts and importances
        term_texts = []
        term_importances = []

        for t in terms:
            if isinstance(t, dict):
                term_texts.append(t.get("term", ""))
                term_importances.append(t.get("importance", 5))
            else:
                term_texts.append(str(t))
                term_importances.append(5)

        # Get embeddings for all terms (with caching)
        term_vecs = []
        valid_importances = []

        for text, imp in zip(term_texts, term_importances):
            vec = self._get_term_embedding(text)
            if vec:
                term_vecs.append(vec)
                valid_importances.append(imp)

        if not term_vecs:
            # No valid embeddings, return nodes unsorted with 0 score
            for node in nodes:
                node["score"] = 0.0
            return nodes

        def score(node):
            title = node.get("title", "")

            # Get title embedding (with caching)
            node_vec = self._get_title_embedding(title)

            if node_vec:
                sim = weighted_cosine_similarity(node_vec, term_vecs, valid_importances)
            else:
                sim = 0.0

            node["score"] = round(sim, 4)
            return node

        scored = [score(n) for n in nodes]
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def get_cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "term_cache_size": len(self._term_cache),
            "title_cache_size": len(self._title_cache),
        }
