"""
phase2b_expander.py — Weighted CPC Hierarchical Expansion Engine (v2.1)

Expands Phase 2A v2 family-level seeds into scored CPC subgroups using:
  1. Inheritance weight from Phase 2A family score
  2. KG hierarchy → class_to_subgroups (3-char keys)
  3. KG graph traversal (depth 2-3)
  4. XML parser (last resort, when KG is sparse)
  5. Embedding proximity (if available)

Each subclass is scored, ranked, and pruned before output.
Expansion balance is enforced across families to prevent under-expansion.
XML fallback triggers when both KG strategies fail to meet minimum count.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RE_FAMILY = re.compile(r"^([A-Z]\d{2}[A-Z])")

# Scoring weights
W_INHERITANCE = 0.50
W_KG_SIMILARITY = 0.30
W_EMBEDDING = 0.20

# Pruning thresholds
INHERITANCE_ONLY_KG_THRESHOLD = (
    0.20  # Max KG similarity for "inheritance-only" classification
)
INHERITANCE_ONLY_EMB_THRESHOLD = (
    0.20  # Max embedding score for "inheritance-only" classification
)
INHERITANCE_ONLY_SCORE_THRESHOLD = (
    0.45  # Higher bar for nodes with NO KG/embedding signal
)

# Expansion balance
MIN_EXPANSION_COUNT = 10  # Minimum subclasses per family
MIN_RATIO = 0.10  # Each family must have ≥ 10% of the largest family's expansion
MAX_TRAVERSAL_DEPTH = 3  # KG graph traversal depth for full subtree
MIN_SUBCLASS_SCORE = 0.30  # Standard pruning threshold for subclass candidates

# Family-level text descriptions used as fallback when KG symbol_texts has no entry
# for a subgroup symbol. Ensures BM25 corpus is never completely empty.
_CPC_FAMILY_TEXT: Dict[str, str] = {
    "G10L": "speech voice audio recognition synthesis phoneme acoustic speaker coding enhancement",
    "G06N": "neural network machine learning deep learning training inference optimization transformer encoder decoder",
    "G06F": "digital data processing program control information retrieval database computing",
    "G06T": "image data processing analysis enhancement segmentation computer vision graphics",
    "G06V": "video understanding scene analysis object detection motion recognition",
    "G06Q": "data processing business commerce administration management finance",
    "H04L": "data communication network protocol routing switching transmission digital",
    "H04W": "wireless communication radio network base station mobile channel",
    "G05B": "control systems industrial automation process robotics",
    "A61B": "medical diagnosis surgery clinical instrument patient monitoring",
    "B60W": "vehicle control autonomous driving steering braking",
    "B60L": "electric vehicle propulsion battery management charging",
    "G01N": "chemical physical analysis measurement testing sensing detection",
    "B01D": "separation filtration membrane purification",
    "E21B": "earth drilling borehole well oil gas extraction",
    "F16J": "sealing pistons cylinders mechanical joints",
    "F16K": "valves taps fluid flow pressure regulation",
}

# Static seed subgroups for families that are commonly under-indexed in the KG.
# Used as guaranteed last-resort when KG + XML both return fewer than MIN_EXPANSION_COUNT.
_STATIC_SUBGROUP_SEEDS: Dict[str, List[Dict[str, str]]] = {
    "G06N": [
        {"code": "G06N3/00",  "title": "Computing arrangements inspired by biological models"},
        {"code": "G06N3/02",  "title": "Neural networks using statistical or probabilistic models"},
        {"code": "G06N3/04",  "title": "Network architectures; activation functions"},
        {"code": "G06N3/06",  "title": "Pulse-coded or spike-based neural networks"},
        {"code": "G06N3/08",  "title": "Learning, training and updating neural networks"},
        {"code": "G06N3/082", "title": "Backpropagation and gradient descent methods"},
        {"code": "G06N3/084", "title": "Self-supervised, unsupervised and reinforcement learning"},
        {"code": "G06N3/088", "title": "Transformer architectures; attention mechanisms"},
        {"code": "G06N3/09",  "title": "Recurrent neural networks; LSTM; GRU"},
        {"code": "G06N3/092", "title": "Convolutional neural networks (CNN)"},
        {"code": "G06N3/10",  "title": "Spiking and reservoir computing networks"},
        {"code": "G06N3/12",  "title": "Genetic algorithms; evolutionary computing"},
        {"code": "G06N3/126", "title": "Evolutionary strategies and CMA-ES"},
        {"code": "G06N20/00", "title": "Machine learning"},
        {"code": "G06N20/10", "title": "Machine learning using kernel methods; SVM"},
        {"code": "G06N20/20", "title": "Ensemble learning; boosting; bagging"},
        {"code": "G06N5/00",  "title": "Computing using knowledge-based models"},
        {"code": "G06N5/01",  "title": "Dynamic generation or adaptation of representation"},
        {"code": "G06N5/022", "title": "Knowledge engineers; knowledge acquisition"},
        {"code": "G06N5/04",  "title": "Inference engines; forward and backward chaining"},
        {"code": "G06N7/00",  "title": "Computing using fuzzy logic"},
        {"code": "G06N7/01",  "title": "Fuzzy inference systems"},
        {"code": "G06N99/00", "title": "Subject matter not provided for in G06N groups"},
    ],
    "G10L": [
        {"code": "G10L15/00", "title": "Speech recognition"},
        {"code": "G10L15/02", "title": "Speech feature extraction"},
        {"code": "G10L15/06", "title": "Word boundary detection"},
        {"code": "G10L15/08", "title": "Acoustic modelling"},
        {"code": "G10L15/16", "title": "Speech classification; language models"},
        {"code": "G10L15/18", "title": "Continuous speech recognition"},
        {"code": "G10L15/20", "title": "Speech recognition with search or match"},
        {"code": "G10L15/22", "title": "Speech recognition using neural networks"},
        {"code": "G10L15/26", "title": "Speech-to-text conversion"},
        {"code": "G10L15/28", "title": "Speech post-processing"},
        {"code": "G10L25/00", "title": "Speech or voice analysis techniques"},
        {"code": "G10L25/03", "title": "Pitch analysis"},
        {"code": "G10L25/18", "title": "Spectral analysis; MFCC features"},
        {"code": "G10L25/30", "title": "Speech or voice quality assessment"},
        {"code": "G10L17/00", "title": "Speaker recognition; voice biometrics"},
        {"code": "G10L19/00", "title": "Speech or audio signal coding; compression"},
        {"code": "G10L21/00", "title": "Speech or voice signal processing"},
        {"code": "G10L21/02", "title": "Speech enhancement; noise suppression"},
    ],
}


class Phase2BExpander:
    """
    Expands CPC family seeds into scored, ranked subgroups.
    Ensures balanced expansion across all Phase 2A v2 families.
    """

    def __init__(self, knowledge_graph=None, xml_parser=None):
        self.kg = knowledge_graph
        self.xml_parser = xml_parser
        self._query_embedding = None

    def expand(
        self,
        families: List[str],
        family_scores: Optional[Dict[str, float]] = None,
        phase1: Optional[dict] = None,
        max_subgroups: int = 500,
    ) -> Dict:
        """
        Expand CPC family seeds into scored subgroups.

        Args:
            families: 4-char family seeds from Phase 2A v2
            family_scores: {family: fused_score} from Phase 2A v2
            phase1: Phase 1 result for embedding computation
            max_subgroups: Cap on total subgroups returned

        Returns:
            {
                "expanded_cpcs": [...],
                "expanded_details": [...],
                "family_expansions": [{"family": "G10L", "subclasses": [...]}],
                "source": "hierarchical_expansion" | "kg_neighbors" | "xml_fallback",
                "fallback_used": bool,
                "family_counts": {"G10L": 238, "G06N": 42},
                "expansion_balance": {"G10L": 238, "G06N": 42},
                "pruned_count": int,
            }
        """
        if not families:
            return {
                "expanded_cpcs": [],
                "expanded_details": [],
                "family_expansions": [],
                "source": "none",
                "fallback_used": False,
                "family_counts": {},
                "expansion_balance": {},
                "pruned_count": 0,
            }

        # Precompute query embedding from Phase 1 text
        if phase1 and self.kg and self.kg.embeddings:
            text = f"{phase1.get('technical_object', '')} {phase1.get('core_function', '')}".strip()
            if text:
                self._compute_embedding(text)

        # Default family scores (equal weight if not provided)
        if family_scores is None:
            family_scores = {f: 1.0 / max(len(families), 1) for f in families}

        # ── Step 1: Expand each family independently ──
        family_results: Dict[str, List[dict]] = {}
        total_pruned = 0

        for family in families:
            result = self._expand_single_family(family, family_scores)
            family_results[family] = result["subclasses"]
            total_pruned += result["pruned_count"]
            logger.info(
                "Phase 2B [%s]: %d subclasses (pruned %d), source=%s",
                family,
                len(result["subclasses"]),
                result["pruned_count"],
                result["source"],
            )

        # Snapshot raw counts before proportional capping (shown to user)
        raw_family_counts = {f: len(v) for f, v in family_results.items()}

        # ── Step 1.5: Cap over-expanded families proportionally to Phase 2A score ──
        family_results, prop_pruned = self._apply_proportional_cap(
            family_results, family_scores
        )
        total_pruned += prop_pruned

        # ── Step 2: Enforce expansion balance with cascading fallbacks ──
        family_results, fallback_used = self._enforce_balance_with_fallback(
            family_results, family_scores, phase1
        )
        total_pruned += family_results.get("_extra_pruned", 0)

        # ── Step 3: Assemble output (flat + structured) ──
        all_expanded: List[dict] = []
        family_expansions: List[dict] = []
        family_counts: Dict[str, int] = {}

        for family in families:
            subclasses = family_results.get(family, [])
            family_expansions.append(
                {
                    "family": family,
                    "subclasses": subclasses,
                }
            )
            family_counts[family] = len(subclasses)
            for sub in subclasses:
                sym = sub.get("code") or sub.get("symbol", "")
                title = sub.get("title", "")
                full_context = self._build_full_context(sym, title)
                # If KG has no text for this symbol, fall back to family description.
                # Prevents BM25 corpus from being completely empty when symbol_texts is sparse.
                if not full_context.strip():
                    full_context = _CPC_FAMILY_TEXT.get(family, sym)
                level = self._get_level(sym)
                score = sub.get("score", 0)
                all_expanded.append(
                    {
                        "symbol": sym,
                        "title": title or full_context,
                        "family": family,
                        "full_context": full_context,
                        "level": level,
                        "score": score,
                    }
                )

        # Hard guard: reject any subgroup not belonging to a requested family.
        # Catches any cross-family contamination from KG hierarchy or graph traversal.
        requested_families = set(families)
        before_guard = len(all_expanded)
        all_expanded = [
            e for e in all_expanded
            if any(e.get("symbol", "").startswith(fam) for fam in requested_families)
        ]
        leaked = before_guard - len(all_expanded)
        if leaked > 0:
            logger.warning(
                "Phase 2B: Family guard removed %d subgroups from unrequested families",
                leaked,
            )

        # Cap total at max_subgroups
        all_expanded.sort(key=lambda x: x["score"], reverse=True)
        for target_family in ["G10L", "G06N"]:
            family_candidates = [
                e for e in all_expanded if e.get("symbol", "").startswith(target_family)
            ]
            top5 = family_candidates[:5]
            logger.info(
                "Phase 2B: TOP 5 %s — %s",
                target_family,
                "; ".join(f"{e['symbol']}:{e.get('score', 0):.3f}" for e in top5),
            )
        if len(all_expanded) > max_subgroups:
            excess = len(all_expanded) - max_subgroups
            all_expanded = all_expanded[:max_subgroups]
            total_pruned += excess

        symbols = [e["symbol"] for e in all_expanded]

        logger.info(
            "Phase 2B: %d subgroups after pruning (%d removed) from %d families: %s",
            len(symbols),
            total_pruned,
            len(family_counts),
            family_counts,
        )

        # Compute proportional caps for UI explanation
        ref_family = max(family_scores, key=lambda f: family_scores.get(f, 0.0))
        ref_score = family_scores.get(ref_family, 1.0)
        ref_raw = raw_family_counts.get(ref_family, 0)
        proportional_caps = {
            f: max(int(ref_raw * (family_scores.get(f, 0.0) / ref_score)), MIN_EXPANSION_COUNT)
            if ref_score > 0 else MIN_EXPANSION_COUNT
            for f in families
        }

        return {
            "expanded_cpcs": symbols,
            "expanded_details": all_expanded,
            "flat_candidates": all_expanded,  # Phase 2C consumes this
            "family_expansions": family_expansions,
            "source": "hierarchical_expansion",
            "fallback_used": fallback_used,
            "family_counts": family_counts,
            "expansion_balance": dict(family_counts),
            "raw_family_counts": raw_family_counts,
            "proportional_caps": proportional_caps,
            "pruned_count": total_pruned,
        }

    def _build_full_context(self, symbol: str, title: str) -> str:
        """
        Build full CPC hierarchy context for TF-IDF matching.
        E.g., 'G06 | G06N | G06N3/00 | Neural networks'
        Falls back to family-level description when KG has no text for the symbol.
        """
        family_prefix = symbol[:4] if len(symbol) >= 4 else symbol
        family_desc = _CPC_FAMILY_TEXT.get(family_prefix, "")

        if not self.kg or not self.kg.graph:
            return title or family_desc

        # Walk up the graph to build parent chain
        parts = [title]
        current = symbol
        visited = {symbol}

        for _ in range(5):  # Max 5 levels up
            parents = list(self.kg.graph.predecessors(current))
            if not parents:
                break
            parent = parents[0]
            parent_str = str(parent)
            if parent_str in visited:
                break
            visited.add(parent_str)
            parent_title = self.kg.symbol_texts.get(parent_str, "")
            if parent_title:
                parts.insert(0, parent_title)
            current = parent_str

        return " | ".join(parts)

    def _get_level(self, symbol: str) -> int:
        """Get CPC hierarchy level from graph node data."""
        if self.kg and self.kg.graph and symbol in self.kg.graph:
            return self.kg.graph.nodes.get(symbol, {}).get("level", 0)
        # Fallback: estimate from symbol depth
        return symbol.count("/") + symbol.count("0")

    def _expand_single_family(
        self, family: str, family_scores: Dict[str, float]
    ) -> Dict:
        """
        Expand a single family using cascading strategies.
        Tries: KG hierarchy → Graph traversal → XML fallback.
        """
        # Strategy 1: KG hierarchy (class_to_subgroups)
        result = self._expand_via_kg_hierarchy_single(family, family_scores)
        if len(result["subclasses"]) >= MIN_EXPANSION_COUNT:
            return result

        # Strategy 2: KG graph traversal (depth 2-3)
        logger.info(
            "Phase 2B [%s]: Hierarchy returned %d (< %d), trying KG graph traversal depth=%d",
            family,
            len(result["subclasses"]),
            MIN_EXPANSION_COUNT,
            MAX_TRAVERSAL_DEPTH,
        )
        graph_result = self._expand_via_graph_traversal(family, family_scores)
        if len(graph_result["subclasses"]) >= MIN_EXPANSION_COUNT:
            return graph_result

        # Strategy 3: XML fallback (last resort)
        logger.warning(
            "Phase 2B [%s]: Both KG strategies failed (%d + %d total). Trying XML fallback",
            family,
            len(result["subclasses"]),
            len(graph_result["subclasses"]),
        )
        xml_result = self._expand_via_xml_fallback(family, family_scores)
        return xml_result

    def _expand_via_kg_hierarchy_single(
        self, family: str, family_scores: Dict[str, float]
    ) -> Dict:
        """Expand single family via KG class_to_subgroups."""
        subclasses: List[dict] = []
        pruned_count = 0

        if not self.kg or not self.kg.class_to_subgroups:
            logger.warning(
                "Phase 2B [%s]: KG or class_to_subgroups unavailable", family
            )
            return {"subclasses": [], "pruned_count": 0, "source": "unavailable"}

        # ── FIX: Try both 3-char and 2-char prefix keys ──
        prefix3 = family[:3]  # Subclass-level key (e.g., "G10L", "G06N")
        subgroups = self.kg.class_to_subgroups.get(prefix3, [])

        if not subgroups:
            # Fallback to 2-char class-level key (e.g., "G10" → G10L, G06F, G06N)
            prefix2 = family[:2]  # Class-level key (e.g., "G10", "G06", "H04")
            subgroups = self.kg.class_to_subgroups.get(prefix2, [])
            if subgroups:
                logger.info(
                    "Phase 2B [%s]: 3-char prefix '%s' not found, using 2-char '%s' (%d subgroups)",
                    family,
                    prefix3,
                    prefix2,
                    len(subgroups),
                )

        if not subgroups:
            logger.warning(
                "Phase 2B [%s]: KG has 0 subgroups for both prefix '%s' and '%s'",
                family,
                prefix3 if prefix3 else "(empty)",
                prefix2 if prefix2 else "(empty)",
            )
            return {"subclasses": [], "pruned_count": 0, "source": "unavailable"}

        logger.info(
            "Phase 2B [%s]: KG has %d subgroups for prefix '%s'",
            family,
            len(subgroups),
            prefix3 if prefix3 else prefix2,
        )

        # Pre-load real XML titles for this family (KG symbol_texts is often empty).
        # Avoids all subgroups falling back to the same generic family-level text.
        xml_titles: Dict[str, str] = {}
        if self.xml_parser:
            try:
                xml_sgs = self.xml_parser.expand_classes(
                    [family], include_non_allocatable=False, allowed_roots=[family]
                )
                xml_titles = {
                    sg.get("symbol", ""): sg.get("title", "")
                    for sg in xml_sgs
                    if sg.get("symbol", "").startswith(family)
                }
                logger.info(
                    "Phase 2B [%s]: Pre-loaded %d XML titles", family, len(xml_titles)
                )
            except Exception as _e:
                logger.warning("Phase 2B [%s]: XML title pre-load failed: %s", family, _e)

        for sym in subgroups:
            # Skip the family node itself (e.g. "G10L") — only subgroups with "/" are allocatable
            if "/" not in sym:
                continue
            m = _RE_FAMILY.match(sym)
            # Skip if regex doesn't match OR the matched family prefix differs.
            # The `m and` guard alone lets None-match symbols leak through — this fixes that.
            if not (m and m.group(1) == family):
                continue

            score = self._score_subclass(sym, family, family_scores)
            title = self.kg.symbol_texts.get(sym, "") or xml_titles.get(sym, "")
            subclasses.append(
                {
                    "code": sym,
                    "title": title,
                    "score": score,
                }
            )

        subclasses.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            "Phase 2B [%s]: Hierarchy returned %d subclasses (pruned %d)",
            family,
            len(subgroups),
            pruned_count,
        )
        return {
            "subclasses": subclasses,
            "pruned_count": pruned_count,
            "source": "hierarchy",
        }

    def _expand_via_graph_traversal(
        self, family: str, family_scores: Dict[str, float]
    ) -> Dict:
        """
        Expand single family via KG graph BFS traversal (depth 2-3).
        Captures full subtree including deeper ML/AI branches.
        """
        subclasses: List[dict] = []
        pruned_count = 0
        seen: set = set()

        if not self.kg or not self.kg.graph:
            return {"subclasses": [], "pruned_count": 0, "source": "unavailable"}

        # Check if family node exists in graph
        if family not in self.kg.graph:
            logger.warning(
                "Phase 2B [%s]: Family '%s' not found in KG graph",
                family,
            )
            return {"subclasses": [], "pruned_count": 0, "source": "unavailable"}

        # Pre-load real XML titles for this family (KG symbol_texts is often empty)
        xml_titles: Dict[str, str] = {}
        if self.xml_parser:
            try:
                xml_sgs = self.xml_parser.expand_classes(
                    [family], include_non_allocatable=False, allowed_roots=[family]
                )
                xml_titles = {
                    sg.get("symbol", ""): sg.get("title", "")
                    for sg in xml_sgs
                    if sg.get("symbol", "").startswith(family)
                }
            except Exception:
                pass

        # BFS from family node with depth limit
        queue = [(family, 0)]
        visited = {family}

        while queue:
            node, depth = queue.pop(0)
            if depth >= MAX_TRAVERSAL_DEPTH:
                continue

            for _, child in self.kg.graph.out_edges(node):
                child_str = str(child)
                if child_str in visited:
                    continue
                visited.add(child_str)

                # Filter: only include allocatable subgroups (must have "/")
                if "/" not in child_str:
                    continue
                if not child_str.startswith(family):
                    continue

                score = self._score_subclass(child_str, family, family_scores)

                title = self.kg.symbol_texts.get(child_str, "") or xml_titles.get(child_str, "")
                subclasses.append(
                    {
                        "code": child_str,
                        "title": title,
                        "score": score,
                    }
                )

                queue.append((child_str, depth + 1))

        subclasses.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            "Phase 2B [%s]: Graph traversal found %d subclasses (depth=%d, pruned %d)",
            family,
            len(subclasses),
            MAX_TRAVERSAL_DEPTH,
            pruned_count,
        )
        return {
            "subclasses": subclasses,
            "pruned_count": pruned_count,
            "source": "graph_traversal",
        }

    def _expand_via_xml_fallback(
        self, family: str, family_scores: Dict[str, float]
    ) -> Dict:
        """
        Expand single family via XML parser as last resort.
        """
        subclasses: List[dict] = []
        pruned_count = 0

        if not self.xml_parser:
            logger.warning(
                "Phase 2B [%s]: XML parser not available, cannot fallback",
                family,
            )
            return {"subclasses": [], "pruned_count": 0, "source": "xml_unavailable"}

        try:
            # Extract the section letter from family (first char)
            section = family[0] if family else "G"
            subgroups = self.xml_parser.expand_classes(
                [family],
                include_non_allocatable=False,
                allowed_roots=[family],
            )

            logger.info(
                "Phase 2B [%s]: XML fallback found %d subgroups for section '%s'",
                family,
                len(subgroups),
                section,
            )

            for sg in subgroups:
                sym = sg.get("symbol", "")
                title = sg.get("title", "")

                # Skip non-matching families
                if not sym.startswith(family):
                    continue

                score = self._score_subclass(sym, family, family_scores)
                subclasses.append(
                    {
                        "code": sym,
                        "title": title,
                        "score": score,
                    }
                )

            subclasses.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                "Phase 2B [%s]: XML fallback returned %d subclasses (pruned %d)",
                family,
                len(subclasses),
                pruned_count,
            )
            return {
                "subclasses": subclasses,
                "pruned_count": pruned_count,
                "source": "xml_fallback",
            }

        except Exception as e:
            logger.error("Phase 2B [%s]: XML fallback failed: %s", family, str(e))
            return {"subclasses": [], "pruned_count": 0, "source": "xml_error"}

    def _expand_via_xml_fallback_unpruned(
        self, family: str, family_scores: Dict[str, float]
    ) -> Dict:
        """
        Expand via XML WITHOUT pruning — used when balancing under-expanded
        families where every available subclass is needed.
        """
        subclasses: List[dict] = []

        if not self.xml_parser:
            return {"subclasses": [], "pruned_count": 0, "source": "xml_unavailable"}

        try:
            subgroups = self.xml_parser.expand_classes(
                [family],
                include_non_allocatable=False,
                allowed_roots=[family],
            )

            for sg in subgroups:
                sym = sg.get("symbol", "")
                title = sg.get("title", "")
                if not sym.startswith(family):
                    continue
                subclasses.append(
                    {
                        "code": sym,
                        "title": title,
                        "score": self._score_subclass(sym, family, family_scores),
                    }
                )

            subclasses.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                "Phase 2B [%s]: XML unpruned fallback returned %d subclasses (no pruning)",
                family,
                len(subclasses),
            )
            return {
                "subclasses": subclasses,
                "pruned_count": 0,
                "source": "xml_fallback_unpruned",
            }

        except Exception as e:
            logger.error(
                "Phase 2B [%s]: XML unpruned fallback failed: %s", family, str(e)
            )
            return {"subclasses": [], "pruned_count": 0, "source": "xml_error"}

    def _apply_proportional_cap(
        self,
        family_results: Dict[str, List[dict]],
        family_scores: Dict[str, float],
    ) -> Tuple[Dict[str, List[dict]], int]:
        """
        Trim each family's expansion proportionally to its Phase 2A score.

        Cap = max(int(ref_count × (family_score / ref_score)), MIN_EXPANSION_COUNT)

        This ensures a family with half the relevance gets at most half the
        search slots, regardless of how many CPC subclasses it has in the taxonomy.
        Runs BEFORE the balance enforcer so the enforcer only adds, never removes.
        """
        if not family_results or not family_scores:
            return family_results, 0

        # Reference: the highest-scoring family's current expansion count
        ref_family = max(family_scores, key=lambda f: family_scores.get(f, 0.0))
        ref_score = family_scores.get(ref_family, 1.0)
        ref_count = len(family_results.get(ref_family, []))

        if ref_count == 0 or ref_score == 0:
            return family_results, 0

        total_trimmed = 0
        capped: Dict[str, List[dict]] = {}

        for family, subclasses in family_results.items():
            score = family_scores.get(family, 0.0)
            cap = max(int(ref_count * (score / ref_score)), MIN_EXPANSION_COUNT)
            if len(subclasses) > cap:
                trimmed = len(subclasses) - cap
                total_trimmed += trimmed
                capped[family] = subclasses[:cap]  # already sorted score desc
                logger.info(
                    "Phase 2B [%s]: Proportional cap %d → %d "
                    "(score=%.3f, ref=%s=%.3f)",
                    family, len(subclasses), cap, score, ref_family, ref_score,
                )
            else:
                capped[family] = subclasses

        return capped, total_trimmed

    def _enforce_balance_with_fallback(
        self,
        family_results: Dict[str, List[dict]],
        family_scores: Dict[str, float],
        phase1: Optional[dict] = None,
    ) -> tuple:
        """
        Ensure each family has ≥ MIN_RATIO of the largest family's expansion.
        Cascading fallback: hierarchy → graph → XML.

        When all strategies fail, relax pruning threshold to let more candidates through.
        """
        if not family_results:
            return family_results, False

        max_count = max(len(v) for v in family_results.values())
        if max_count == 0:
            return family_results, False

        min_required = max(int(max_count * MIN_RATIO), MIN_EXPANSION_COUNT)
        fallback_used = False
        extra_pruned = 0

        under_expanded = {
            family: len(subclasses)
            for family, subclasses in family_results.items()
            if len(subclasses) < min_required
        }

        if not under_expanded:
            logger.info(
                "Phase 2B: All families meet balance requirement (min=%d, largest=%d)",
                min_required,
                max_count,
            )
            return family_results, False

        logger.warning(
            "Phase 2B: Under-expanded families: %s",
            ", ".join(
                f"{f} ({len(family_results[f])}/{min_required})" for f in under_expanded
            ),
        )

        # ── Cascading fallback for each under-expanded family ──
        for family in list(under_expanded.keys()):
            current_count = len(family_results[family])
            logger.info(
                "Phase 2B [%s]: Current %d < target %d — cascading fallback",
                family,
                current_count,
                min_required,
            )

            # Try Strategy 2: Graph traversal
            graph_result = self._expand_via_graph_traversal(family, family_scores)
            graph_count = len(graph_result["subclasses"])

            if graph_count >= min_required:
                logger.info(
                    "Phase 2B [%s]: Graph traversal boosted to %d subclasses",
                    family,
                    graph_count,
                )
                family_results[family] = graph_result["subclasses"]
                continue

            # Try Strategy 3: XML fallback
            logger.warning(
                "Phase 2B [%s]: Graph traversal insufficient (%d < %d), trying XML fallback",
                family,
                graph_count,
                min_required,
            )
            xml_result = self._expand_via_xml_fallback(family, family_scores)
            xml_count = len(xml_result["subclasses"])

            if xml_count >= min_required:
                logger.info(
                    "Phase 2B [%s]: XML fallback returned %d subclasses (raw, unpruned)",
                    family,
                    xml_count,
                )
                # CRITICAL: For XML fallback, don't apply the standard 0.30 pruning.
                # The balance rule already validated that these are the only candidates available.
                family_results[family] = xml_result["subclasses"]
                fallback_used = True
                continue

            # ── All strategies exhausted: Relax pruning threshold ──
            logger.warning(
                "Phase 2B [%s]: ALL strategies exhausted. "
                "Relaxing pruning threshold from %.2f → 0.0 to force balance.",
                family,
                MIN_SUBCLASS_SCORE,
            )
            # Re-expand via XML without pruning
            xml_raw = self._expand_via_xml_fallback_unpruned(family, family_scores)
            raw_count = len(xml_raw["subclasses"])

            if raw_count >= min_required:
                logger.info(
                    "Phase 2B [%s]: Pruning relaxed to 0.0 → %d raw XML subclasses",
                    family,
                    raw_count,
                )
                family_results[family] = xml_raw["subclasses"]
                fallback_used = True
                continue

            # ── Last resort: static seed table ──
            static_seeds = _STATIC_SUBGROUP_SEEDS.get(family, [])
            if static_seeds:
                existing_codes = {s.get("code", "") for s in family_results[family]}
                merged = list(family_results[family])
                for seed in static_seeds:
                    if seed["code"] not in existing_codes:
                        score = self._score_subclass(seed["code"], family, family_scores)
                        merged.append({"code": seed["code"], "title": seed["title"], "score": score})
                        existing_codes.add(seed["code"])
                merged.sort(key=lambda x: x["score"], reverse=True)
                family_results[family] = merged
                logger.info(
                    "Phase 2B [%s]: Static seed fallback added %d entries → %d total",
                    family,
                    len(merged) - current_count,
                    len(merged),
                )
                fallback_used = True
                continue

            # Nothing worked — accept what we have
            logger.error(
                "Phase 2B [%s]: ALL strategies failed after threshold relaxation. "
                "Family stuck at %d subclasses (target: %d). "
                "This family will be under-represented in Phase 2C.",
                family,
                current_count,
                min_required,
            )

        if extra_pruned > 0:
            family_results["_extra_pruned"] = extra_pruned

        return family_results, fallback_used

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for text using KG model."""
        if not self.kg or not self.kg.model:
            return None
        try:
            return self.kg.model.encode(
                [text], show_progress_bar=False, convert_to_numpy=True
            )[0]
        except Exception as e:
            logger.warning("Failed to compute embedding: %s", e)
            return None

    def _compute_kg_similarity(self, symbol: str) -> float:
        """
        Compute structural similarity via KG graph connectivity.
        Returns score in [0, 1] based on graph position and neighbor density.
        """
        if not self.kg or not self.kg.graph:
            return 0.0

        if symbol not in self.kg.graph:
            return 0.0

        in_degree = self.kg.graph.in_degree(symbol)
        out_degree = self.kg.graph.out_degree(symbol)
        total_degree = in_degree + out_degree

        # Normalize: typical CPC subclass has 1-10 connections
        return min(total_degree / 10.0, 1.0)

    def _compute_embedding_similarity(self, symbol: str) -> float:
        """Compute embedding similarity between query and CPC symbol."""
        if self._query_embedding is None:
            return 0.0
        if not self.kg or symbol not in self.kg.embeddings:
            return 0.0

        try:
            sym_emb = self.kg.embeddings[symbol]
            q_norm = np.linalg.norm(self._query_embedding)
            s_norm = np.linalg.norm(sym_emb)
            if q_norm == 0 or s_norm == 0:
                return 0.0
            sim = float(np.dot(self._query_embedding, sym_emb) / (q_norm * s_norm))
            return max(0.0, sim)
        except Exception:
            return 0.0

    def _score_subclass(
        self, symbol: str, family: str, family_scores: Dict[str, float]
    ) -> float:
        """
        Compute weighted subclass score.

        subclass_score =
            family_score × inheritance_weight
          + kg_similarity × W_KG_SIMILARITY
          + embedding_similarity × W_EMBEDDING
        """
        inheritance = family_scores.get(family, 0.0)
        kg_sim = self._compute_kg_similarity(symbol)
        emb_sim = self._compute_embedding_similarity(symbol)

        score = (
            inheritance * W_INHERITANCE
            + kg_sim * W_KG_SIMILARITY
            + emb_sim * W_EMBEDDING
        )
        return round(min(score, 1.0), 4)
