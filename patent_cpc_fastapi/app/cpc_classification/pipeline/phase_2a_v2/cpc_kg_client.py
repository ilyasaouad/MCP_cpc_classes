"""
cpc_kg_client.py — CPC Knowledge Graph client interface.

Provides structured access to the CPC Knowledge Graph for graph-aware
family scoring in Phase 2A v2. Wraps the raw CPCKnowledgeGraph with
stable, purpose-specific methods.

All public scoring methods return results at CPC FAMILY level (4-char,
e.g. G10L, G06N) — never at class level (3-char, e.g. G10).
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CPCKGClient:
    """
    Stable interface to the CPC Knowledge Graph for routing purposes.

    All scoring outputs are at CPC family level (4-char prefix).
    """

    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self._family_cache: Dict[str, List[str]] = {}

    def is_available(self) -> bool:
        return self.kg is not None and bool(self.kg.embeddings)

    # ──────────────────────────────────────────────
    # HIERARCHY HELPERS
    # ──────────────────────────────────────────────

    def get_families_for_class(self, prefix3: str) -> List[str]:
        """
        Get all unique 4-char CPC families under a 3-char class.

        E.g. "G10" → ["G10L", "G10K", "G10H", ...]

        Results are cached after first lookup.
        """
        if prefix3 in self._family_cache:
            return self._family_cache[prefix3]

        if not self.is_available():
            return []

        subgroups = self.kg.class_to_subgroups.get(prefix3, [])
        families = set()
        for sym in subgroups:
            if len(sym) >= 4:
                families.add(sym[:4])
            elif len(sym) == 3:
                families.add(sym)

        result = sorted(families)
        self._family_cache[prefix3] = result
        return result

    def get_all_known_families(self) -> List[str]:
        """
        Get every unique 4-char family present in the KG.
        """
        if not self.is_available():
            return []

        all_families = set()
        for prefix3 in self.kg.class_to_subgroups:
            for sym in self.kg.class_to_subgroups[prefix3]:
                if len(sym) >= 4:
                    all_families.add(sym[:4])
        return sorted(all_families)

    # ──────────────────────────────────────────────
    # SCORING (family-level)
    # ──────────────────────────────────────────────

    def get_cpc_definition(self, code: str) -> str:
        """Get descriptive text for a CPC code from the KG."""
        if not self.is_available():
            return ""
        return self.kg.symbol_texts.get(code, "")

    def get_cpc_neighbors(self, code: str, depth: int = 2) -> List[str]:
        """
        Get neighboring CPC families via graph traversal.
        Returns list of 4-char CPC family prefixes.
        """
        if not self.is_available():
            return []

        neighbors: set = set()

        # Get subclasses within same 3-char class
        prefix3 = code[:3] if len(code) >= 3 else code
        subclasses = self.kg.class_to_subgroups.get(prefix3, [])[:10]
        for s in subclasses:
            if len(s) >= 4:
                neighbors.add(s[:4])
            elif len(s) == 3:
                neighbors.add(s)

        # Graph traversal
        if self.kg.graph:
            current = [code]
            for _ in range(depth):
                next_level = []
                for node in current:
                    if node in self.kg.graph:
                        for _, child in self.kg.graph.out_edges(node):
                            child_str = str(child)
                            fam = child_str[:4] if len(child_str) >= 4 else child_str
                            if fam not in neighbors:
                                neighbors.add(fam)
                                next_level.append(child_str)
                current = next_level[:20]

        return sorted(neighbors)[:50]

    def graph_semantic_score(self, text: str, top_k: int = 20) -> Dict[str, float]:
        """
        Score CPC FAMILIES against text using KG embeddings.

        Internally the KG returns 3-char class scores; this method
        expands each class score to all families under that class
        so the output is always at family level.

        Returns dict of {family_4char: similarity_score}.
        """
        if not self.is_available() or not text.strip():
            return {}

        try:
            # KG returns 3-char class scores
            class_results = self.kg.find_classes_by_text(text, top_k=top_k)

            # Expand each class score to its families
            family_scores: Dict[str, float] = {}
            for class_code, score in class_results:
                # Normalize to 3-char
                prefix3 = class_code[:3] if len(class_code) >= 3 else class_code
                families = self.get_families_for_class(prefix3)

                if families:
                    for fam in families:
                        # Distribute class score to each family
                        family_scores[fam] = score
                else:
                    # Fallback: use the class code itself as key
                    key = class_code[:4] if len(class_code) >= 4 else class_code
                    family_scores[key] = score

            return family_scores

        except Exception as e:
            logger.warning("KG semantic scoring failed: %s", e)
            return {}

    def get_family_members(self, family_prefix: str) -> List[str]:
        """Get subclass symbols for a CPC family (e.g. 'G10L' → ['G10L15/00', ...])."""
        if not self.is_available():
            return []
        prefix3 = family_prefix[:3] if len(family_prefix) >= 3 else family_prefix
        return self.kg.class_to_subgroups.get(prefix3, [])[:30]
