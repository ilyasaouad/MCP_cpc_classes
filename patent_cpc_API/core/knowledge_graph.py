"""
knowledge_graph.py â€” Thin singleton wrapper over CPCKnowledgeGraph.

Loads once at startup and is injected into every phase that needs it.
All heavy graph and embedding state lives here.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Will be populated when the original CPCKnowledgeGraph is available.
# Import lazily so this module can be imported without the full dependency stack.
_KG_CLASS = None


def _load_kg_class():
    global _KG_CLASS
    if _KG_CLASS is None:
        try:
            # Adjust import path to wherever the original KG lives in your deployment
            from cpc_classification.knowledge_graph import (
                CPCKnowledgeGraph,
            )
            _KG_CLASS = CPCKnowledgeGraph
        except ImportError:
            logger.warning(
                "CPCKnowledgeGraph not importable â€” running without graph/embeddings"
            )
    return _KG_CLASS


class KnowledgeGraphSingleton:
    """
    Holds the single shared CPCKnowledgeGraph instance for the API lifetime.

    Usage in engine.py:
        kg = KnowledgeGraphSingleton.get(cache_dir=settings.kg_cache_dir)
    """

    _instance: Optional[object] = None

    @classmethod
    def get(cls, cache_dir: str, model_name: str = "all-mpnet-base-v2"):
        if cls._instance is None:
            KGClass = _load_kg_class()
            if KGClass is None:
                logger.warning("No KG class available; returning None")
                return None
            try:
                kg = KGClass(cache_dir=cache_dir, model_name=model_name)
                loaded = kg.load()
                if not loaded:
                    logger.info("KG cache miss â€” building from XML files")
                    xml_dir = os.path.join(cache_dir, "cpc_scheme_2026")
                    if os.path.isdir(xml_dir):
                        kg.build_from_cache(xml_dir)
                        kg.save()
                    else:
                        logger.warning("XML dir not found: %s", xml_dir)
                cls._instance = kg
                logger.info("KnowledgeGraph loaded (nodes=%d)", len(kg.graph.nodes))
            except Exception as exc:
                logger.error("KG init failed: %s", exc)
                return None
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force re-initialisation on next get() call (useful in tests)."""
        cls._instance = None
