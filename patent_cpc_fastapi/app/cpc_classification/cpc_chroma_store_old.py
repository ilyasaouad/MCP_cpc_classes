"""
CPC ChromaDB Vector Store - On-demand Section Loading

Loads only the sections needed for each query, making initialization fast.
Sections are cached persistently in ChromaDB for subsequent use.
"""

import os
import time
import fnmatch
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings

from .cpc_xml_parser import CPCXMLParser


class CPCChromaStore:
    """
    ChromaDB-based store for CPC classification codes.

    Features:
    - Loads sections on-demand (only what's needed)
    - Persistent caching of loaded sections
    - Fast queries after sections are loaded
    """

    def __init__(self, persist_dir: str, xml_dir: str):
        """
        Initialize ChromaDB store.

        Args:
            persist_dir: Directory to persist ChromaDB data
            xml_dir: Directory containing cpc-scheme-*.xml files
        """
        self.persist_dir = persist_dir
        self.xml_dir = xml_dir
        self.xml_parser = CPCXMLParser(xml_dir)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="cpc_scheme", metadata={"hnsw:space": "cosine"}
        )

        # Track which sections are loaded
        self._loaded_sections = set()
        self._check_loaded_sections()

        print(f"ChromaDB initialized with {self.collection.count()} documents")
        print(f"Loaded sections: {sorted(self._loaded_sections)}")

    def _check_loaded_sections(self) -> None:
        """Check which sections are already loaded in ChromaDB."""
        if self.collection.count() == 0:
            return

        # Sample some documents to check sections
        try:
            result = self.collection.get(limit=1000)
            if result and result["metadatas"]:
                for metadata in result["metadatas"]:
                    section = metadata.get("section", "")
                    if section:
                        self._loaded_sections.add(section)
        except Exception:
            pass

    def load_section(self, section: str) -> None:
        """
        Load all XML files for a given section into ChromaDB.

        Args:
            section: Section letter (e.g., 'G', 'A', 'C')
        """
        if section in self._loaded_sections:
            return

        print(f"Loading section {section} into ChromaDB...")
        start_time = time.time()

        # Find all XML files for this section
        all_files = os.listdir(self.xml_dir)
        pattern = f"cpc-scheme-{section}*.xml"
        matching_files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]

        if not matching_files:
            print(f"  No XML files found for section {section}")
            return

        print(f"  Found {len(matching_files)} files for section {section}")

        # Collect documents
        documents = []
        metadatas = []
        ids = []

        for xml_file in matching_files:
            class_code = xml_file.replace("cpc-scheme-", "").replace(".xml", "")

            try:
                subgroups = self.xml_parser.parse_file(class_code)

                for sg in subgroups:
                    if sg["is_allocatable"] and sg["title"]:
                        doc_text = f"{sg['symbol']}: {sg['title']}"

                        documents.append(doc_text)
                        metadatas.append(
                            {
                                "symbol": sg["symbol"],
                                "title": sg["title"],
                                "level": sg["level"],
                                "section": section,
                                "class_code": class_code,
                            }
                        )
                        ids.append(sg["symbol"])

            except Exception as e:
                print(f"  Warning: Failed to parse {xml_file}: {e}")

        # Add to ChromaDB in batches
        batch_size = 1000
        total = len(documents)

        if total > 0:
            print(f"  Adding {total} documents...")
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx],
                )

        elapsed = time.time() - start_time
        print(f"  Section {section} loaded in {elapsed:.1f}s ({total} documents)")

        self._loaded_sections.add(section)

    def query(
        self, query_texts: List[str], sections: List[str], n_results: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for similar CPC codes.

        Args:
            query_texts: List of query texts (e.g., extracted terms)
            sections: List of section codes to filter by (e.g., ['G', 'H'])
            n_results: Number of results to return

        Returns:
            List of dicts with symbol, title, score, level
        """
        if not query_texts:
            return []

        # Ensure sections are loaded
        for section in sections:
            if section and section not in self._loaded_sections:
                self.load_section(section)

        # Build where filter
        where_filter = None
        if sections:
            # Filter to loaded sections only
            valid_sections = [s for s in sections if s in self._loaded_sections]
            if valid_sections:
                if len(valid_sections) == 1:
                    where_filter = {"section": valid_sections[0]}
                else:
                    where_filter = {"section": {"$in": valid_sections}}

        # Join query texts
        query = " ".join(query_texts)

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score
                similarity = 1.0 - distance

                formatted.append(
                    {
                        "symbol": metadata["symbol"],
                        "title": metadata["title"],
                        "level": metadata["level"],
                        "score": round(similarity, 4),
                        "class_code": metadata.get("class_code", ""),
                    }
                )

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            "total_documents": self.collection.count(),
            "loaded_sections": sorted(list(self._loaded_sections)),
            "persist_dir": self.persist_dir,
        }

    def reset(self) -> None:
        """Reset the store."""
        self.client.delete_collection("cpc_scheme")
        self.collection = self.client.get_or_create_collection(
            name="cpc_scheme", metadata={"hnsw:space": "cosine"}
        )
        self._loaded_sections = set()
        print("ChromaDB reset complete")
