"""
cpc_filters.py — Shared predicates for filtering non-allocatable CPC symbols.

Used in Phase 2B, Phase 3A, and Phase 4B to guarantee only real subgroups
(those with a "/" in the symbol) enter scoring and output.
"""

import re
from typing import List, Dict, Any

# Cross-reference indexing codes embed a 4-digit year: G10L2015/xxxx
_CROSS_REF_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]\d{4}/")


def is_allocatable(symbol: str) -> bool:
    """Return True only if `symbol` is a valid allocatable CPC subgroup."""
    if not symbol:
        return False
    if "/" not in symbol:
        return False                          # class / subclass / group node
    if _CROSS_REF_PATTERN.match(symbol):
        return False                          # cross-reference indexing code
    return True


def filter_allocatable(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove non-allocatable entries from a candidate list."""
    return [c for c in candidates if is_allocatable(c.get("symbol", ""))]
