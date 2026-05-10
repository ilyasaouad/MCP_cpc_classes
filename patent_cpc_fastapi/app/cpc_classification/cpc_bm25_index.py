"""
cpc_bm25_index.py - BM25 Index for CPC Retrieval

Builds a BM25 index on CPC subgroup titles for fast keyword-based retrieval.
Used as the first stage in hybrid retrieval (BM25 recall + KG rerank).

Usage:
    from cpc_bm25_index import CPCBM25Index

    # Build once
    index = CPCBM25Index(cache_dir="resources/cpc_scheme_2026")
    index.build()
    index.save("resources/cpc_bm25_index.pkl")

    # Load and search
    index.load("resources/cpc_bm25_index.pkl")
    results = index.search("neural network quantization", top_k=200)
"""

import os
import json
import pickle
import logging
from typing import List, Tuple, Dict
from collections import defaultdict

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class CPCBM25Index:
    """
    BM25 index for CPC subgroup titles.

    Provides fast keyword-based retrieval to expand recall
    before semantic reranking.
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.bm25 = None
        self.documents: List[str] = []  # titles
        self.symbols: List[str] = []  # corresponding CPC symbols
        self.tokenized_docs: List[List[str]] = []

    def build(self, cache_dir: str = None) -> None:
        """
        Build BM25 index from CPC cache JSON files.

        Args:
            cache_dir: Directory containing cpc-cache-*.json files
        """
        if cache_dir is not None:
            self.cache_dir = cache_dir

        if self.cache_dir is None:
            raise ValueError("cache_dir must be provided")

        logger.info("Building BM25 index from %s", self.cache_dir)

        # Load all CPC entries
        entries = self._load_cache_files()
        logger.info("Loaded %d CPC entries", len(entries))

        # Build documents
        self.symbols = []
        self.documents = []

        for entry in entries:
            symbol = entry.get("symbol", "")
            title = entry.get("title", "")

            if symbol and title:
                self.symbols.append(symbol)
                self.documents.append(title)

        logger.info("Indexing %d documents", len(self.documents))

        # Tokenize for BM25
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

        logger.info("BM25 index built successfully")

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """
        Search BM25 index for query.

        Args:
            query: Search query (e.g., patent text or terms)
            top_k: Number of results to return

        Returns:
            List of (symbol, bm25_score) tuples, sorted by score descending
        """
        if not self.bm25:
            raise RuntimeError(
                "BM25 index not built or loaded. Call build() or load() first."
            )

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        # Return (symbol, score) pairs
        results = [(self.symbols[i], float(scores[i])) for i in top_indices]

        return results

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        data = {
            "symbols": self.symbols,
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
        }

        # BM25Okapi can be pickled
        data["bm25"] = self.bm25

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("BM25 index saved to %s (%d entries)", path, len(self.symbols))

    def load(self, path: str) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.symbols = data["symbols"]
            self.documents = data["documents"]
            self.tokenized_docs = data["tokenized_docs"]
            self.bm25 = data["bm25"]

            logger.info(
                "BM25 index loaded from %s (%d entries)", path, len(self.symbols)
            )
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)
            return False

    def _load_cache_files(self) -> List[Dict]:
        """Load all CPC cache JSON files."""
        entries = []

        for filename in os.listdir(self.cache_dir):
            if filename.startswith("cpc-cache-") and filename.endswith(".json"):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            entries.extend(data)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", filename, e)

        return entries

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Lowercase, split on whitespace, remove short tokens.
        """
        text = text.lower()
        # Remove punctuation and split
        tokens = []
        for word in text.split():
            # Remove non-alphanumeric characters
            cleaned = "".join(c for c in word if c.isalnum())
            if len(cleaned) > 2:  # Skip very short tokens
                tokens.append(cleaned)
        return tokens


# Convenience function
def build_or_load_bm25_index(cache_dir: str, index_path: str) -> CPCBM25Index:
    """
    Build BM25 index if not exists, otherwise load from cache.

    Args:
        cache_dir: Directory with cpc-cache-*.json files
        index_path: Path to save/load BM25 index pickle

    Returns:
        Built or loaded CPCBM25Index
    """
    index = CPCBM25Index(cache_dir)

    # Try to load existing
    if os.path.exists(index_path):
        if index.load(index_path):
            return index

    # Build new
    index.build()
    index.save(index_path)
    return index
