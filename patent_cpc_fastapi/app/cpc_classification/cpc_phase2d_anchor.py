"""
Phase 2D — Top-N Score Filter

Filters Phase 2C candidates by score, keeping only the top N highest-scoring
subgroups for Phase 3 ranking. Eliminates low-confidence candidates that would
dilute Phase 3's ranking quality.

Strategy:
    1. Sort all Phase 2C candidates by hybrid score (descending).
    2. Keep top N (default 50) — discard the rest.
    3. Provides clean, focused input to Phase 3 with reduced noise.
"""

import logging
import re
from typing import Dict, Any, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Layer categories for anchor extraction
# ─────────────────────────────────────────────────────────────────────

TECHNICAL_LAYERS = frozenset(
    {
        "pure_software",
        "data_reasoning",
        "interaction",
        "control",
    }
)

EXCLUDED_LAYERS = frozenset(
    {
        "application",
    }
)

# 3-digit class prefixes that are too broad to serve as anchors
EXCLUDED_BROAD_PREFIXES: Set[str] = set()

# 4-char pattern: A-Z prefix + 2 digits (e.g., G06F, B60W, H04L)
_SUBCLASS_PATTERN = re.compile(r"^([A-Z]\d{2}[A-Z])")

# Non-technical / business-method families to explicitly exclude
# even if they somehow appear in technical layers
_NON_TECHNICAL_FAMILIES = frozenset(
    {
        "G06Q",  # Business methods / administration
        "G06C",  # Digital computers (miscellaneous)
        "G07F",  # Coin-freed apparatus
        "G07G",  # Cash registers
        "G09F",  # Display / advertising
        "G09B",  # Educational appliances
        "A63F",  # Games
    }
)

# Mapping of 3-char CPC families → known 4-char subclasses
# Used as fallback when Phase 2A uses family router instead of layer decomposer
_FAMILY_TO_SUBCLASS: Dict[str, List[str]] = {
    "G06": ["G06F", "G06N", "G06T", "G06V", "G06K", "G06Q"],
    "G10": ["G10L", "G10H", "G10K"],
    "H04": ["H04L", "H04W", "H04N", "H04B", "H04M", "H04R"],
    "G05": ["G05B", "G05D", "G05F"],
    "B60": ["B60W", "B60R", "B60L", "B60T", "B60N", "B60K", "B60Q"],
    "A61": ["A61B", "A61C", "A61F", "A61H", "A61J", "A61K", "A61M", "A61N", "A61P"],
    "B23": ["B23B", "B23C", "B23D", "B23K", "B23P", "B23Q"],
    "H02": ["H02J", "H02K", "H02M", "H02P", "H02H", "H02G"],
    "E21": ["E21B", "E21C", "E21D", "E21F"],
    "A01": ["A01B", "A01C", "A01D", "A01G", "A01K", "A01M", "A01N"],
    "H01": ["H01L", "H01M", "H01R", "H01F", "H01H", "H01J"],
    "G01": [
        "G01N",
        "G01S",
        "G01R",
        "G01L",
        "G01B",
        "G01C",
        "G01D",
        "G01F",
        "G01M",
        "G01V",
    ],
    "F16": ["F16H", "F16K", "F16L", "F16B", "F16D", "F16F", "F16J"],
    "B65": ["B65D", "B65G", "B65H"],
    "C08": ["C08F", "C08G", "C08J", "C08K", "C08L"],
    "H05": ["H05B", "H05K", "H05H"],
    "F02": ["F02B", "F02D", "F02F", "F02M", "F02P"],
    "F01": ["F01D", "F01N", "F01P"],
    "B01": ["B01D", "B01F", "B01J", "B01L"],
}


class Phase2DSubclassAnchor:
    """
    Filters CPC candidates through a subclass anchor derived from Phase 2A
    technical layers, eliminating semantic drift into irrelevant domains.
    """

    def __init__(
        self,
        technical_layers: frozenset = TECHNICAL_LAYERS,
        excluded_layers: frozenset = EXCLUDED_LAYERS,
        excluded_families: frozenset = _NON_TECHNICAL_FAMILIES,
    ):
        self.technical_layers = technical_layers
        self.excluded_layers = excluded_layers
        self.excluded_families = excluded_families

    # ── public API ──────────────────────────────────────────────────

    def filter(
        self,
        candidates: List[Dict[str, Any]],
        layer_result: Dict[str, Any],
        max_result: int = 50,
    ) -> Dict[str, Any]:
        """
        Top-N score filter: keep the highest-scoring Phase 2C candidates.

        Args:
            candidates: Phase 2C candidates (list of dicts with 'symbol', 'score', ...)
            layer_result: Unused — kept for API compatibility with the runner.
            max_result: Number of top candidates to keep (default 50).

        Returns:
            Dict with:
                - purified_candidates: Top-N candidates sorted by score
                - kept_count / discarded_count: Statistics
                - discard_log: Symbols that were cut
                - anchor_set / anchor_source: Empty (legacy fields, kept for compatibility)
        """
        if not candidates:
            return {
                "purified_candidates": [],
                "anchor_set": [],
                "anchor_source": ["score_filter"],
                "kept_count": 0,
                "discarded_count": 0,
                "discard_log": [],
            }

        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0), reverse=True
        )

        kept = sorted_candidates[:max_result]
        discarded = sorted_candidates[max_result:]

        discard_log = [
            {"symbol": c.get("symbol", ""), "reason": "below_top_n_threshold"}
            for c in discarded
        ]

        logger.info(
            "Phase 2D: Top-%d filter — Input=%d → Kept=%d → Discarded=%d "
            "(score range: %.4f – %.4f)",
            max_result,
            len(candidates),
            len(kept),
            len(discarded),
            kept[-1].get("score", 0) if kept else 0,
            kept[0].get("score", 0) if kept else 0,
        )

        return {
            "purified_candidates": kept,
            "anchor_set": [],
            "anchor_source": ["score_filter"],
            "kept_count": len(kept),
            "discarded_count": len(discarded),
            "discard_log": discard_log,
        }

    # ── internal helpers ────────────────────────────────────────────

    def _build_anchor_set(
        self, layer_result: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Build the anchor set from Phase 2A technical layers.

        Supports two result formats:
        1. Layer decomposition (has 'layers' dict) — extracts 4-char subclasses
           from technical layers, excluding application layer.
        2. Family router (has 'families' list) — extracts 4-char subclasses
           from the routed family list.

        Strictly excludes:
        - application layer candidates
        - 3-digit broad classes (e.g., B60, H04)
        - non-technical families (G06Q, etc.)

        Returns: (anchor_set, anchor_source_layers)
        """
        anchor_set: Set[str] = set()
        source_layers: Set[str] = set()

        layers = layer_result.get("layers", {})

        if layers:
            # ── Layer decomposition path ──
            for layer_name, layer_candidates in layers.items():
                if layer_name in self.excluded_layers:
                    continue
                if layer_name not in self.technical_layers:
                    continue
                if not layer_candidates:
                    continue

                source_layers.add(layer_name)

                for candidate in layer_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    symbol = candidate.get("symbol", "")
                    prefix = self._extract_subclass_prefix(symbol)
                    if prefix is None:
                        continue
                    if prefix in self.excluded_families:
                        continue
                    anchor_set.add(prefix)
        else:
            # ── Family router fallback ──
            families = layer_result.get("families", [])
            if not families:
                families = layer_result.get("phase2a_families", [])
            for fam in families:
                if not fam or not isinstance(fam, str):
                    continue
                if len(fam) == 4:
                    if fam not in self.excluded_families:
                        anchor_set.add(fam)
                        source_layers.add("family_router")
                elif len(fam) == 3:
                    source_layers.add(f"family_router:{fam}")
                    for sub in _FAMILY_TO_SUBCLASS.get(fam, []):
                        if sub not in self.excluded_families:
                            anchor_set.add(sub)
                else:
                    prefix = self._extract_subclass_prefix(fam)
                    if prefix and prefix not in self.excluded_families:
                        anchor_set.add(prefix)
                        source_layers.add("family_router")

        return anchor_set, source_layers

    @staticmethod
    def _extract_subclass_prefix(symbol: str) -> Optional[str]:
        """
        Extract the 4-character CPC subclass prefix from a symbol.

        Examples:
            'G06F40/30'   → 'G06F'
            'B60W50/08'   → 'B60W'
            'G06N'        → 'G06N'
            'B60'         → None  (3-digit class, too broad)

        Returns None for symbols without a valid 4-char subclass prefix.
        """
        if not symbol:
            return None
        match = _SUBCLASS_PATTERN.match(symbol)
        if match:
            prefix = match.group(1)
            # Ensure it's exactly 4 characters (not a 3-char class)
            if len(prefix) == 4:
                return prefix
        return None


# ─────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────


def apply_subclass_anchor_filter(
    candidates: List[Dict[str, Any]],
    layer_result: Dict[str, Any],
    max_result: int = 50,
) -> Dict[str, Any]:
    """Apply Phase 2D subclass anchor filter (convenience wrapper)."""
    anchor = Phase2DSubclassAnchor()
    return anchor.filter(candidates, layer_result, max_result=max_result)
