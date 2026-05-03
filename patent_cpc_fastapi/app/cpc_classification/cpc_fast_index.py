"""
CPC Fast Index - JSON-based keyword search

Lightweight alternative to ChromaDB:
- Parses XML once to build a JSON index
- Loads index at startup (1-2 seconds)
- Uses fast keyword matching for candidate selection
- No embeddings needed, very fast queries
"""

import json
import os
import time
import fnmatch
from typing import Dict, List, Any

from .cpc_xml_parser import CPCXMLParser


class CPCFastIndex:
    """
    Fast JSON-based index for CPC codes.

    Features:
    - Builds index from XML files (one-time, ~30s)
    - Loads in 1-2 seconds
    - Keyword-based search
    - Section filtering
    """

    def __init__(self, xml_dir: str, cache_path: str = None):
        """
        Initialize fast index.

        Args:
            xml_dir: Directory containing cpc-scheme-*.xml files
            cache_path: Path to JSON cache file (auto-generated if None)
        """
        self.xml_dir = xml_dir

        if cache_path is None:
            self.cache_path = os.path.join(xml_dir, "cpc_fast_index.json")
        else:
            self.cache_path = cache_path

        # Load or build index
        self.index = self._load_or_build()
        print(f"Fast index loaded: {len(self.index)} codes")

    def _load_or_build(self) -> Dict[str, Dict]:
        """Load index from cache or build from XML."""
        # Try to load from cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Loaded index from {self.cache_path}")
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}")

        # Build from XML
        return self._build_index()

    def _build_index(self) -> Dict[str, Dict]:
        """Build index from all XML files."""
        print("Building CPC index from XML...")
        start_time = time.time()

        parser = CPCXMLParser(self.xml_dir)

        # Find all XML files
        all_files = os.listdir(self.xml_dir)
        xml_files = [f for f in all_files if f.endswith(".xml")]

        index = {}

        for xml_file in xml_files:
            class_code = xml_file.replace("cpc-scheme-", "").replace(".xml", "")
            section = class_code[0] if class_code else ""

            try:
                subgroups = parser.parse_file(class_code)

                for sg in subgroups:
                    if sg["is_allocatable"] and sg["title"]:
                        index[sg["symbol"]] = {
                            "title": sg["title"],
                            "level": sg["level"],
                            "section": section,
                            "class_code": class_code,
                            "parent_chain": sg.get("parent_chain", []),
                        }
            except Exception as e:
                print(f"  Warning: Failed to parse {xml_file}: {e}")

        # Save to cache
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"Index built in {elapsed:.1f}s: {len(index)} codes")
        print(f"Saved to {self.cache_path}")

        return index

    def search(
        self, query_terms: List[str], sections: List[str] = None, top_n: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search index using keyword matching.

        Args:
            query_terms: List of search terms
            sections: Optional section filter
            top_n: Number of results to return

        Returns:
            List of matching codes with scores
        """
        if not query_terms:
            return []

        # Normalize query terms
        query_terms = [t.lower() for t in query_terms]

        # Score each code
        scored = []

        for symbol, data in self.index.items():
            # Section filter
            if sections and data["section"] not in sections:
                continue

            title = data["title"].lower()

            # Calculate score: count matching terms
            score = 0
            for term in query_terms:
                if term in title:
                    score += len(term)  # Weight by term length
                # Also check individual words for multi-word terms
                if " " in term:
                    words = term.split()
                    score += sum(1 for word in words if word in title and len(word) > 3)

            if score > 0:
                scored.append((score, symbol, data))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Format results
        results = []
        for score, symbol, data in scored[:top_n]:
            # Normalize score to 0-1 range
            normalized_score = min(score / 50.0, 1.0)

            results.append(
                {
                    "symbol": symbol,
                    "title": data["title"],
                    "level": data["level"],
                    "score": round(normalized_score, 4),
                    "class_code": data["class_code"],
                }
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        sections = set()
        for data in self.index.values():
            sections.add(data["section"])

        return {
            "total_codes": len(self.index),
            "sections": sorted(list(sections)),
            "cache_path": self.cache_path,
        }
