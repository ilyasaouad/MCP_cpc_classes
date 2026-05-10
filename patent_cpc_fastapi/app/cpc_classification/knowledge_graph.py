"""
CPC Knowledge Graph with Semantic Embeddings

Builds a NetworkX graph from CPC cache JSON files and generates
sentence-BERT embeddings for semantic search.

Usage:
    graph = CPCKnowledgeGraph(cache_dir="resources/")
    if not graph.load():
        graph.build_from_cache("resources/cpc_scheme_2026/")
        graph.save()

    results = graph.semantic_search("chatbot dialog system", top_k=10)
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CPCKnowledgeGraph:
    """
    Knowledge graph for CPC classification with semantic embeddings.

    Builds graph from existing cpc-cache-*.json files.
    Uses all-mpnet-base-v2 (110MB) for embeddings.
    """

    def __init__(self, cache_dir: str, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the knowledge graph.

        Args:
            cache_dir: Directory to store graph cache files
            model_name: Sentence transformer model name
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model = None

        # Graph structure
        self.graph = nx.DiGraph()

        # Embeddings: symbol -> numpy array
        self.embeddings: Dict[str, np.ndarray] = {}

        # Text mapping: symbol -> concatenated title text
        self.symbol_texts: Dict[str, str] = {}

        # 3-char class mapping: G06F -> [G06F16/00, G06F17/30, ...]
        self.class_to_subgroups: Dict[str, List[str]] = {}

        # File hash for change detection
        self.source_hash = ""

        # BM25 index for hybrid retrieval
        self.bm25_index = None
        self.bm25_index_path = os.path.join(cache_dir, "cpc_bm25_index.pkl")

        # Cross-encoder for reranking
        self.cross_encoder = None

        logger.info("CPCKnowledgeGraph initialized (cache_dir: %s)", cache_dir)

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self.model is None:
            logger.info("Loading sentence transformer model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        return self.model

    def build_from_xml(
        self,
        cpc_xml_dir: str,
        sections: Optional[List[str]] = None,
    ):
        """
        Build graph directly from CPC scheme XML files.

        Args:
            cpc_xml_dir: Directory containing cpc-scheme-*.xml files
            sections: List of sections to build (e.g., ['G', 'H'] for computing/telecom)
                     None = all sections
        """
        import glob
        from lxml import etree as ET

        logger.info("Building knowledge graph from XML files in %s", cpc_xml_dir)

        # Find all XML files
        xml_files = glob.glob(os.path.join(cpc_xml_dir, "cpc-scheme-*.xml"))

        if sections:
            xml_files = [
                f
                for f in xml_files
                if any(
                    os.path.basename(f).startswith(f"cpc-scheme-{s}") for s in sections
                )
            ]
            logger.info("Filtered to sections %s: %d files", sections, len(xml_files))
        else:
            logger.info("Found %d XML files", len(xml_files))

        if not xml_files:
            logger.error("No XML files found!")
            return

        all_subgroups = []
        for i, xml_file in enumerate(xml_files):
            try:
                subgroups = self._parse_xml_file(xml_file)
                all_subgroups.extend(subgroups)
                if (i + 1) % 50 == 0 or i == len(xml_files) - 1:
                    logger.info(
                        "  Parsed %d/%d files (%d subgroups)",
                        i + 1,
                        len(xml_files),
                        len(all_subgroups),
                    )
            except Exception as e:
                logger.warning("Failed to parse %s: %s", os.path.basename(xml_file), e)

        logger.info("Total subgroups extracted: %d", len(all_subgroups))

        if not all_subgroups:
            logger.error("No subgroups extracted!")
            return

        # Build graph and embeddings
        self._build_graph_structure(all_subgroups)
        self._generate_embeddings()

        logger.info("Knowledge graph build complete")
        logger.info("  - Graph nodes: %d", self.graph.number_of_nodes())
        logger.info("  - Graph edges: %d", self.graph.number_of_edges())
        logger.info("  - Embeddings: %d", len(self.embeddings))

    def _parse_xml_file(self, xml_file: str) -> List[dict]:
        """Parse a single CPC scheme XML file."""
        from lxml import etree as ET

        tree = ET.parse(xml_file)
        root = tree.getroot()
        subgroups = []

        # NO NAMESPACE - plain element names
        items = list(root.iter("classification-item"))

        for item in items:
            symbol_elem = item.find("classification-symbol")
            title_elem = item.find("class-title")

            if symbol_elem is None or title_elem is None:
                continue

            symbol = symbol_elem.text or ""

            # Extract all text from title element
            title = " ".join(title_elem.itertext()).strip()

            if not symbol or not title:
                continue

            # Get parent chain by walking up the tree
            parent_chain = []
            parent = item.getparent()
            while parent is not None and parent.tag == "classification-item":
                parent_symbol = parent.find("classification-symbol")
                if parent_symbol is not None and parent_symbol.text:
                    parent_chain.insert(0, parent_symbol.text)
                parent = parent.getparent()

            # Check if allocatable (not-allocatable="false" means it IS allocatable)
            not_alloc = item.get("not-allocatable", "false").lower() == "true"
            is_allocatable = not not_alloc

            subgroups.append(
                {
                    "symbol": symbol,
                    "title": title,
                    "level": len(parent_chain),
                    "parent_chain": parent_chain,
                    "is_allocatable": is_allocatable,
                }
            )

        return subgroups

    def build_from_cache(
        self,
        cpc_cache_dir: str,
        max_subgroups: Optional[int] = None,
        sections: Optional[List[str]] = None,
    ):
        """
        Build graph from existing cpc-cache-*.json files.

        Args:
            cpc_cache_dir: Directory containing cpc-cache-*.json files
            max_subgroups: Limit for testing (None = all)
        """
        logger.info("Building knowledge graph from cache files in %s", cpc_cache_dir)

        # Step 1: Load all cache files
        cache_files = sorted(
            [
                f
                for f in os.listdir(cpc_cache_dir)
                if f.startswith("cpc-cache-") and f.endswith(".json")
            ]
        )

        logger.info("Found %d cache files", len(cache_files))

        all_subgroups = []
        for i, cache_file in enumerate(cache_files):
            filepath = os.path.join(cpc_cache_dir, cache_file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_subgroups.extend(data)
                    if (i + 1) % 50 == 0 or i == len(cache_files) - 1:
                        logger.info(
                            "  Loaded %d/%d cache files (%d subgroups so far)",
                            i + 1,
                            len(cache_files),
                            len(all_subgroups),
                        )
            except Exception as e:
                logger.warning("Failed to load %s: %s", cache_file, e)

        logger.info("Total subgroups loaded: %d", len(all_subgroups))

        if max_subgroups:
            all_subgroups = all_subgroups[:max_subgroups]
            logger.info("Limited to %d subgroups for testing", max_subgroups)

        # Step 2: Build graph nodes and hierarchy
        self._build_graph_structure(all_subgroups)

        # Step 3: Generate embeddings
        self._generate_embeddings()

        # Step 4: Compute source hash for change detection
        self.source_hash = self._compute_source_hash(cpc_cache_dir)

        logger.info("Knowledge graph build complete")
        logger.info("  - Graph nodes: %d", self.graph.number_of_nodes())
        logger.info("  - Graph edges: %d", self.graph.number_of_edges())
        logger.info("  - Embeddings: %d", len(self.embeddings))

    def _build_graph_structure(self, subgroups: List[dict]):
        """Build NetworkX graph from subgroup data."""
        logger.info("Building graph structure from %d subgroups...", len(subgroups))

        # CRITICAL OPTIMIZATION: Build symbol-to-title dictionary for O(1) lookups
        # Instead of O(n) linear search for every parent lookup
        logger.info("  Creating symbol index for fast parent lookups...")
        symbol_to_title = {}
        for sg in subgroups:
            sym = sg.get("symbol", "")
            if sym:
                symbol_to_title[sym] = sg.get("title", "").strip()
        logger.info("  Symbol index created: %d entries", len(symbol_to_title))

        # Track 3-char classes
        char3_classes = set()
        total = len(subgroups)
        last_progress = 0

        for idx, sg in enumerate(subgroups):
            symbol = sg.get("symbol", "")
            title = sg.get("title", "").strip()
            level = sg.get("level", 0)
            parent_chain = sg.get("parent_chain", [])
            is_allocatable = sg.get("is_allocatable", False)

            if not symbol or not title:
                continue

            # Extract 3-char prefix (e.g., G06F -> G06)
            prefix3 = symbol[:3] if len(symbol) >= 3 else symbol
            char3_classes.add(prefix3)

            # Extract 4-char class (e.g., G06F)
            prefix4 = symbol[:4] if len(symbol) >= 4 else symbol

            # Add subgroup node
            self.graph.add_node(
                symbol,
                type="subgroup",
                title=title,
                level=level,
                is_allocatable=is_allocatable,
                prefix3=prefix3,
                prefix4=prefix4,
            )

            # Build full context text (concatenate parent titles)
            context_parts = []
            for parent in parent_chain:
                # FAST O(1) lookup using dictionary
                parent_title = symbol_to_title.get(parent, "")
                if parent_title:
                    context_parts.append(parent_title)
            context_parts.append(title)
            full_text = " | ".join(filter(None, context_parts))
            self.symbol_texts[symbol] = full_text

            # Map to 3-char class
            if prefix3 not in self.class_to_subgroups:
                self.class_to_subgroups[prefix3] = []
            self.class_to_subgroups[prefix3].append(symbol)

            # Hierarchy edges (parent-child)
            if parent_chain:
                immediate_parent = parent_chain[-1]
                self.graph.add_edge(
                    immediate_parent, symbol, relation="parent_of", weight=1.0
                )

            # Progress logging every 10%
            progress = int((idx + 1) / total * 100)
            if progress >= last_progress + 10:
                logger.info(
                    "  Graph build: %d%% (%d/%d subgroups)", progress, idx + 1, total
                )
                last_progress = progress

        # Add 3-char class nodes
        for cls in char3_classes:
            self.graph.add_node(
                cls, type="class_3char", title=f"CPC Class {cls}", level=2
            )

            # Link subgroups to their 3-char class
            for sg_symbol in self.class_to_subgroups.get(cls, []):
                self.graph.add_edge(cls, sg_symbol, relation="contains", weight=0.5)

        logger.info(
            "Graph structure built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _generate_embeddings(self):
        """Generate embeddings for all subgroup texts."""
        logger.info("Generating embeddings for %d subgroups...", len(self.symbol_texts))

        model = self._get_model()

        symbols = list(self.symbol_texts.keys())
        texts = [self.symbol_texts[s] for s in symbols]

        # Generate in batches to avoid memory issues
        batch_size = 256
        all_embeddings = []

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = model.encode(
                batch_texts, show_progress_bar=False, convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

            batch_num = i // batch_size + 1
            if (
                batch_num % max(1, total_batches // 10) == 0
                or batch_num == total_batches
            ):
                logger.info(
                    "  Embedded %d/%d batches (%.0f%%) — %d/%d subgroups",
                    batch_num,
                    total_batches,
                    batch_num / total_batches * 100,
                    min(i + len(batch_texts), len(texts)),
                    len(texts),
                )

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)

        # Store as dict
        for idx, symbol in enumerate(symbols):
            self.embeddings[symbol] = all_embeddings[idx]

        logger.info(
            "Embeddings generated: %d vectors of dimension %d",
            len(self.embeddings),
            all_embeddings.shape[1],
        )

    def semantic_search(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find CPC subgroups most similar to the given text.

        Args:
            text: Patent text or query
            top_k: Number of results to return

        Returns:
            List of (symbol, similarity_score) tuples
        """
        if not self.embeddings:
            logger.warning("No embeddings available for semantic search")
            return []

        model = self._get_model()
        query_embedding = model.encode([text], convert_to_numpy=True)

        # Compute similarities
        symbols = list(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[s] for s in symbols])

        similarities = cosine_similarity(query_embedding, embedding_matrix)[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(symbols[i], float(similarities[i])) for i in top_indices]

        return results

    def find_classes_by_text(
        self, text: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find 3-char CPC classes most relevant to the text.

        Returns aggregated scores at the 3-char class level.
        """
        # Get subgroup-level similarities
        subgroup_results = self.semantic_search(text, top_k=50)

        # Aggregate to 3-char class level
        class_scores = {}
        class_counts = {}

        for symbol, score in subgroup_results:
            # Get 3-char prefix
            node_data = self.graph.nodes.get(symbol, {})
            prefix3 = node_data.get(
                "prefix3", symbol[:3] if len(symbol) >= 3 else symbol
            )

            if prefix3 not in class_scores:
                class_scores[prefix3] = 0.0
                class_counts[prefix3] = 0

            class_scores[prefix3] += score
            class_counts[prefix3] += 1

        # Normalize by count and sort
        aggregated = []
        for prefix3, total_score in class_scores.items():
            avg_score = total_score / class_counts[prefix3]
            aggregated.append((prefix3, avg_score))

        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:top_k]

    def get_related_classes(self, symbol: str, max_depth: int = 2) -> List[str]:
        """
        Find CPC classes related to the given symbol via graph edges.

        Args:
            symbol: CPC symbol (e.g., G06F16/3329)
            max_depth: Maximum traversal depth

        Returns:
            List of related CPC symbols
        """
        if symbol not in self.graph:
            return []

        related = set()

        # BFS traversal
        visited = {symbol}
        queue = [(symbol, 0)]

        while queue:
            current, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.add(neighbor)
                    queue.append((neighbor, depth + 1))

            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return list(related)

    def find_initial_classes(
        self, patent_text: str, extracted_terms: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find initial CPC classes using hybrid approach.

        Combines:
        - Semantic search on full patent text (60% weight)
        - Concept matching on extracted terms (40% weight)

        Args:
            patent_text: Full patent text
            extracted_terms: Key terms from Phase 1
            top_k: Number of classes to return

        Returns:
            List of (class_symbol, score) tuples
        """
        # Semantic search on full text
        semantic_results = self.find_classes_by_text(patent_text, top_k=20)

        # Term-based search
        term_results = []
        if extracted_terms:
            term_text = " ".join(extracted_terms)
            term_results = self.find_classes_by_text(term_text, top_k=20)

        # Combine scores
        combined = {}

        # Semantic scores (60%)
        for cls, score in semantic_results:
            combined[cls] = score * 0.6

        # Term scores (40%)
        for cls, score in term_results:
            if cls in combined:
                combined[cls] += score * 0.4
            else:
                combined[cls] = score * 0.4

        # Sort and return top-k
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_subgroups_for_class(self, class_symbol: str) -> List[str]:
        """Get all subgroup symbols belonging to a 3-char class."""
        return self.class_to_subgroups.get(class_symbol, [])

    def save(self):
        """Save graph and embeddings to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)

        # Save graph
        graph_path = os.path.join(self.cache_dir, "cpc_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph, f)

        # Save embeddings
        embeddings_path = os.path.join(self.cache_dir, "cpc_embeddings.npz")
        np.savez_compressed(
            embeddings_path,
            symbols=list(self.embeddings.keys()),
            embeddings=np.array(list(self.embeddings.values())),
        )

        # Save metadata
        meta_path = os.path.join(self.cache_dir, "cpc_graph_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_hash": self.source_hash,
                    "num_nodes": self.graph.number_of_nodes(),
                    "num_edges": self.graph.number_of_edges(),
                    "num_embeddings": len(self.embeddings),
                    "model_name": self.model_name,
                    "class_to_subgroups": {
                        k: len(v) for k, v in self.class_to_subgroups.items()
                    },
                },
                f,
                indent=2,
            )

        logger.info("Knowledge graph saved to %s", self.cache_dir)

    def load(self) -> bool:
        """
        Load graph and embeddings from disk.

        Returns:
            True if loaded successfully, False if cache doesn't exist
        """
        graph_path = os.path.join(self.cache_dir, "cpc_graph.pkl")
        embeddings_path = os.path.join(self.cache_dir, "cpc_embeddings.npz")
        meta_path = os.path.join(self.cache_dir, "cpc_graph_meta.json")

        if not all(os.path.exists(p) for p in [graph_path, embeddings_path, meta_path]):
            logger.info("Knowledge graph cache not found, will build from scratch")
            return False

        try:
            # Load graph
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)

            # Load embeddings
            data = np.load(embeddings_path)
            symbols = data["symbols"].tolist()
            embeddings = data["embeddings"]

            self.embeddings = {
                symbol: embeddings[i] for i, symbol in enumerate(symbols)
            }

            # Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.source_hash = meta.get("source_hash", "")
                self.model_name = meta.get("model_name", self.model_name)

            # Rebuild class_to_subgroups mapping
            self._rebuild_class_mapping()

            logger.info("Knowledge graph loaded from cache")
            logger.info("  - Nodes: %d", self.graph.number_of_nodes())
            logger.info("  - Edges: %d", self.graph.number_of_edges())
            logger.info("  - Embeddings: %d", len(self.embeddings))

            return True

        except Exception as e:
            logger.error("Failed to load knowledge graph: %s", e)
            return False

    def _rebuild_class_mapping(self):
        """Rebuild class_to_subgroups from graph nodes."""
        self.class_to_subgroups = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "subgroup":
                prefix3 = data.get("prefix3", node[:3] if len(node) >= 3 else node)
                if prefix3 not in self.class_to_subgroups:
                    self.class_to_subgroups[prefix3] = []
                self.class_to_subgroups[prefix3].append(node)

    def check_source_changed(self, cpc_cache_dir: str) -> bool:
        """
        Check if source XML cache files have changed since last build.

        Returns:
            True if source files changed (need rebuild)
        """
        current_hash = self._compute_source_hash(cpc_cache_dir)
        return current_hash != self.source_hash

    def _compute_source_hash(self, cpc_cache_dir: str) -> str:
        """Compute hash of all source cache files for change detection."""
        hasher = hashlib.md5()

        cache_files = sorted(
            [
                f
                for f in os.listdir(cpc_cache_dir)
                if f.startswith("cpc-cache-") and f.endswith(".json")
            ]
        )

        for filename in cache_files:
            filepath = os.path.join(cpc_cache_dir, filename)
            stat = os.stat(filepath)
            hasher.update(filename.encode())
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime).encode())

        return hasher.hexdigest()

    # ───────────────────────────────────────────
    # HYBRID RETRIEVAL: BM25 + Embedding + Cross-Encoder
    # ───────────────────────────────────────────

    def init_hybrid_retrieval(self, cpc_cache_dir: str = None) -> None:
        """
        Initialize BM25 index and cross-encoder for hybrid retrieval.

        Args:
            cpc_cache_dir: Directory with cpc-cache-*.json files
        """
        if cpc_cache_dir is None:
            # Try to infer from cache_dir
            cpc_cache_dir = os.path.join(self.cache_dir, "cpc_scheme_2026")

        # Initialize BM25 index
        self._init_bm25(cpc_cache_dir)

        # Initialize cross-encoder
        self._init_cross_encoder()

    def _init_bm25(self, cpc_cache_dir: str) -> None:
        """Load or build BM25 index."""
        try:
            from .cpc_bm25_index import build_or_load_bm25_index

            self.bm25_index = build_or_load_bm25_index(
                cache_dir=cpc_cache_dir,
                index_path=self.bm25_index_path,
            )
            logger.info("BM25 index ready (%d entries)", len(self.bm25_index.symbols))
        except Exception as e:
            logger.warning("Failed to initialize BM25 index: %s", e)
            self.bm25_index = None

    def _init_cross_encoder(self) -> None:
        """Initialize cross-encoder for reranking."""
        try:
            from .cpc_cross_encoder import CPCCrossEncoder

            self.cross_encoder = CPCCrossEncoder()
            logger.info("Cross-encoder ready")
        except Exception as e:
            logger.warning("Failed to initialize cross-encoder: %s", e)
            self.cross_encoder = None

    def hybrid_search(
        self,
        text: str,
        top_k: int = 20,
        bm25_k: int = 200,
        use_cross_encoder: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search: BM25 recall + semantic scoring + optional cross-encoder rerank.

        Pipeline:
        1. BM25: Retrieve top bm25_k candidates by keyword matching
        2. Semantic: Score BM25 candidates using embeddings
        3. Cross-encoder (optional): Rerank top 50 for highest precision
        4. Aggregate to 3-char class level

        Args:
            text: Patent text or query
            top_k: Number of final results
            bm25_k: Number of candidates from BM25 recall
            use_cross_encoder: Whether to use cross-encoder reranking

        Returns:
            List of (class_symbol, score) tuples
        """
        if not self.embeddings:
            logger.warning("No embeddings available, falling back to semantic_search")
            return self.find_classes_by_text(text, top_k=top_k)

        # Step 1: BM25 recall expansion
        bm25_candidates = []
        if self.bm25_index:
            bm25_results = self.bm25_index.search(text, top_k=bm25_k)
            bm25_candidates = [symbol for symbol, _ in bm25_results]
            logger.debug("BM25 recalled %d candidates", len(bm25_candidates))

        if not bm25_candidates:
            # Fallback to all embeddings
            logger.debug("BM25 not available, using all embeddings")
            return self.find_classes_by_text(text, top_k=top_k)

        # Step 2: Semantic scoring on BM25 candidates only
        candidate_scores = {}

        # Get query embedding
        model = self._get_model()
        query_embedding = model.encode([text], convert_to_numpy=True)

        for symbol in bm25_candidates:
            if symbol in self.embeddings:
                emb = self.embeddings[symbol].reshape(1, -1)
                sim = cosine_similarity(query_embedding, emb)[0][0]
                candidate_scores[symbol] = float(sim)

        if not candidate_scores:
            return []

        # Step 3: Cross-encoder reranking (optional)
        if use_cross_encoder and self.cross_encoder:
            # Get top 50 by semantic score for reranking
            top_semantic = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:50]

            # Prepare candidates with titles
            rerank_candidates = []
            for symbol, _ in top_semantic:
                title = self._get_symbol_title(symbol)
                if title:
                    rerank_candidates.append((symbol, title))

            if rerank_candidates:
                reranked = self.cross_encoder.rerank(text, rerank_candidates, top_k=50)

                # Blend semantic and cross-encoder scores
                rerank_dict = {s: score for s, score in reranked}
                for symbol, sem_score in candidate_scores.items():
                    if symbol in rerank_dict:
                        # Weighted blend: 60% semantic, 40% cross-encoder
                        candidate_scores[symbol] = (
                            0.6 * sem_score + 0.4 * rerank_dict[symbol]
                        )

        # Step 4: Aggregate to 3-char class level
        class_scores = {}
        class_counts = {}

        for symbol, score in candidate_scores.items():
            prefix3 = symbol[:3] if len(symbol) >= 3 else symbol

            if prefix3 not in class_scores:
                class_scores[prefix3] = 0
                class_counts[prefix3] = 0

            class_scores[prefix3] += score
            class_counts[prefix3] += 1

        # Average by count
        for prefix in class_scores:
            class_scores[prefix] /= class_counts[prefix]

        # Return top-k
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_classes[:top_k]

    def _get_symbol_title(self, symbol: str) -> str:
        """Get title for a symbol from graph or symbol_texts."""
        # Try graph first
        if symbol in self.graph.nodes:
            return self.graph.nodes[symbol].get("title", "")
        # Fallback to symbol_texts
        return self.symbol_texts.get(symbol, "")
