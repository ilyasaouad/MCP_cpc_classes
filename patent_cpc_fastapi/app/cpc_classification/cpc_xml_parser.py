"""
CPC XML Parser

Parses local CPC XML scheme files and extracts the full classification hierarchy.
Caches parsed results to JSON for fast reloads.
"""

import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional


class CPCXMLParser:
    """
    Parser for CPC XML scheme files.

    Extracts nested classification hierarchy with full titles.
    Caches results to JSON for persistence across restarts.
    """

    def __init__(self, xml_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize parser.

        Args:
            xml_dir: Directory containing cpc-scheme-*.xml files
            cache_dir: Directory to store parsed JSON caches (default: same as xml_dir)
        """
        self.xml_dir = xml_dir
        self.cache_dir = cache_dir or xml_dir
        self._cache = {}  # In-memory cache

        # Create cache dir if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, class_code: str) -> str:
        """Get path to JSON cache file for a given class code."""
        return os.path.join(self.cache_dir, f"cpc-cache-{class_code}.json")

    def _load_from_cache(self, class_code: str) -> Optional[List[Dict[str, Any]]]:
        """Load parsed data from JSON cache if it exists."""
        cache_path = self._get_cache_path(class_code)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_to_cache(self, class_code: str, data: List[Dict[str, Any]]) -> None:
        """Save parsed data to JSON cache."""
        cache_path = self._get_cache_path(class_code)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache for {class_code}: {e}")

    def _extract_title(self, class_title_elem) -> str:
        """Extract full title text from <class-title> element."""
        parts = []

        for title_part in class_title_elem.findall(".//title-part"):
            # Get direct text
            text_elem = title_part.find("text")
            if text_elem is not None and text_elem.text:
                parts.append(text_elem.text.strip())

            # Also check for CPC-specific-text
            cpc_text = title_part.find("CPC-specific-text")
            if cpc_text is not None:
                text_elem = cpc_text.find("text")
                if text_elem is not None and text_elem.text:
                    parts.append(text_elem.text.strip())

        return " ".join(parts) if parts else ""

    def _parse_item(
        self,
        item_elem,
        parent_chain: Optional[List[str]] = None,
        parent_titles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Parse a single <classification-item> element."""
        if parent_chain is None:
            parent_chain = []
        if parent_titles is None:
            parent_titles = []

        # Extract symbol
        symbol_elem = item_elem.find("classification-symbol")
        symbol = symbol_elem.text.strip() if symbol_elem is not None else ""

        # Extract attributes
        level = int(item_elem.get("level", 0))
        is_allocatable = item_elem.get("not-allocatable", "false") == "false"

        # Extract title
        class_title = item_elem.find("class-title")
        title = self._extract_title(class_title) if class_title is not None else ""

        # Build full parent chain and parent titles
        full_chain = parent_chain + [symbol]
        full_titles = parent_titles + [title]

        # Combine all parent titles for richer matching context
        full_context = " ".join(filter(None, full_titles))

        return {
            "symbol": symbol,
            "title": title,
            "full_context": full_context,
            "level": level,
            "is_allocatable": is_allocatable,
            "parent_chain": parent_chain.copy(),
            "parent_titles": parent_titles.copy(),
        }

    def _parse_recursive(
        self,
        item_elem,
        parent_chain: Optional[List[str]] = None,
        parent_titles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recursively parse <classification-item> and all its children.
        Returns flat list of all items in the hierarchy.
        """
        if parent_chain is None:
            parent_chain = []
        if parent_titles is None:
            parent_titles = []

        results = []

        # Parse current item
        item_data = self._parse_item(item_elem, parent_chain, parent_titles)
        results.append(item_data)

        # Recursively parse children
        # Children are <classification-item> elements that are direct children
        for child in item_elem.findall("classification-item"):
            child_results = self._parse_recursive(
                child,
                item_data["parent_chain"] + [item_data["symbol"]],
                item_data["parent_titles"] + [item_data["title"]],
            )
            results.extend(child_results)

        return results

    def parse_file(
        self, class_code: str, force_reload: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Parse a single CPC XML file and return all subgroups.

        Args:
            class_code: The class code (e.g., 'G06N', 'G06F')
            force_reload: If True, ignore cache and re-parse XML

        Returns:
            List of dicts with symbol, title, level, is_allocatable, parent_chain
        """
        # Check in-memory cache
        if not force_reload and class_code in self._cache:
            return self._cache[class_code]

        # Check file cache
        if not force_reload:
            cached = self._load_from_cache(class_code)
            if cached is not None:
                self._cache[class_code] = cached
                return cached

        # Try direct file match first (e.g., G06N → cpc-scheme-G06N.xml)
        xml_path = os.path.join(self.xml_dir, f"cpc-scheme-{class_code}.xml")

        # If not found, try wildcard match for section-level codes
        # (e.g., A61 → cpc-scheme-A61B.xml, cpc-scheme-A61K.xml, etc.)
        xml_files = []
        if os.path.exists(xml_path):
            xml_files.append(xml_path)
        else:
            import fnmatch

            all_files = os.listdir(self.xml_dir)
            pattern = f"cpc-scheme-{class_code}*.xml"
            matching = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
            xml_files = [os.path.join(self.xml_dir, f) for f in matching]

        if not xml_files:
            return []

        try:
            results = []
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Find all top-level classification items
                for item in root.findall("classification-item"):
                    results.extend(self._parse_recursive(item))

            # Cache results
            self._cache[class_code] = results
            self._save_to_cache(class_code, results)

            return results

        except Exception as e:
            print(f"Error parsing XML for {class_code}: {e}")
            return []

    def expand_classes(
        self, class_codes: List[str], include_non_allocatable: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Expand a list of broad class codes to all their subgroups.

        Args:
            class_codes: List of class codes from Phase 1 (e.g., ['G06N', 'G06F'])
            include_non_allocatable: If False, filter out codes that cannot be assigned

        Returns:
            Flat list of all subgroups with full titles and hierarchy context
        """
        all_candidates = []
        seen_symbols = set()

        for code in class_codes:
            subgroups = self.parse_file(code)

            for sg in subgroups:
                symbol = sg["symbol"]

                # Skip duplicates
                if symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)

                # Skip non-allocatable unless requested
                if not include_non_allocatable and not sg["is_allocatable"]:
                    continue

                all_candidates.append(sg)

        return all_candidates

    def get_class_summary(self, class_code: str) -> Dict[str, Any]:
        """
        Get a summary of a parsed class file.

        Returns:
            Dict with total_codes, max_level, allocatable_codes, etc.
        """
        subgroups = self.parse_file(class_code)

        if not subgroups:
            return {"error": f"No data found for {class_code}"}

        allocatable = [sg for sg in subgroups if sg["is_allocatable"]]
        levels = [sg["level"] for sg in subgroups]

        return {
            "class_code": class_code,
            "total_codes": len(subgroups),
            "allocatable_codes": len(allocatable),
            "non_allocatable_codes": len(subgroups) - len(allocatable),
            "max_level": max(levels) if levels else 0,
            "min_level": min(levels) if levels else 0,
            "sample_codes": [sg["symbol"] for sg in allocatable[:5]],
        }


# Convenience function for quick testing
def parse_cpc_class(
    class_code: str, xml_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Quick function to parse a single CPC class."""
    if xml_dir is None:
        # Default to project resources directory
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        xml_dir = os.path.join(base_dir, "resources", "cpc_scheme_2026")

    parser = CPCXMLParser(xml_dir)
    return parser.parse_file(class_code)
