"""
cpc_hypothesis_resolver.py - Phase 5: CPC Hypothesis Resolution + Tri-Pillar Classification

Deterministic resolver that selects the best CPC hypothesis from Phase 4 output.
Also produces a Tri-Pillar classification: Primary Function (G06F), Methodology (G06N),
Application Domain (G05B) — by back-scanning Phase 2/3 raw candidates.

KEY DESIGN:
- NOT a free-form classifier
- NOT validating individual CPC codes
- IS a deterministic scorer that ranks Phase 4 hypotheses
- Tri-Pillar: finds highest-scoring champion per functional role
- LLM used ONLY for tie-breaking or targeted lookups when candidates missing
"""

import logging
import re
from typing import Dict, List, Any, Optional

# Matches CPC cross-reference / indexing codes that embed a 4-digit year, e.g. G10L2015/0631.
# These are not allocatable subgroups and must be excluded from champion selection.
_CROSS_REF_PATTERN = re.compile(r'^[A-Z]\d{2}[A-Z]\d{4}/')

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Tri-Pillar Definitions
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_PILLAR_DEFINITIONS = {
    "pillar1_goal": {
        "label": "Primary Facet (Core Purpose)",
        "description": "The core technical result/output — what the invention PRODUCES.",
        "families": ["G06F", "G06Q"],
        "fallback_family": "G06F",
    },
    "pillar2_method": {
        "label": "Methodological Facet (Implementation)",
        "description": "The AI/ML implementation strategy — how the invention WORKS.",
        "families": ["G06N"],
        "fallback_family": "G06N",
    },
    "pillar3_context": {
        "label": "Application Facet (Domain)",
        "description": "The secondary domain — where the invention APPLIES.",
        "families": ["G05B", "B60W", "A61B", "H02J"],
        "fallback_family": "G05B",
    },
}


def _build_pillar_definitions(phase2a_families: List[str]) -> Dict[str, Any]:
    """
    Build Tri-Pillar definitions dynamically from Phase 2A selected families.

    pillar1_goal  → primary Phase 2A family (highest relevance)
    pillar2_method → G06N always (ML/AI methodology), or primary if G06N absent
    pillar3_context → remaining Phase 2A families (secondary domain)
    """
    if not phase2a_families:
        return dict(_DEFAULT_PILLAR_DEFINITIONS)

    METHOD_FAMILY = "G06N"
    primary = phase2a_families[0]
    others = [f for f in phase2a_families[1:] if f != METHOD_FAMILY]

    # pillar1: primary domain family
    p1_families = [primary] if primary != METHOD_FAMILY else phase2a_families[:1]

    # pillar2: always G06N if present, else first family
    p2_families = [METHOD_FAMILY] if METHOD_FAMILY in phase2a_families else [primary]

    # Families whose domain clearly contradicts a speech/audio primary family (G10L).
    # G06T (image/video) and G06V (computer vision) are visual-domain families that
    # should not appear as the Application facet when the primary is acoustic/speech.
    _VISUAL_FAMILIES = {"G06T", "G06V", "G06K"}
    primary_is_speech = primary.startswith("G10L")
    p3_candidates = [
        f for f in others
        if not (primary_is_speech and f in _VISUAL_FAMILIES)
    ]

    # pillar3: remaining families after domain filtering.
    # Fallback to primary family rather than re-including blocked families —
    # the original else branch was re-adding G06T for speech patents.
    p3_families = p3_candidates if p3_candidates else p1_families

    return {
        "pillar1_goal": {
            "label": "Primary Facet (Core Purpose)",
            "description": "The core technical result/output — what the invention PRODUCES.",
            "families": p1_families,
            "fallback_family": p1_families[0],
        },
        "pillar2_method": {
            "label": "Methodological Facet (Implementation)",
            "description": "The AI/ML implementation strategy — how the invention WORKS.",
            "families": p2_families,
            "fallback_family": p2_families[0],
        },
        "pillar3_context": {
            "label": "Application Facet (Domain)",
            "description": "The secondary domain — where the invention APPLIES.",
            "families": p3_families if p3_families else p1_families,
            "fallback_family": p3_families[0] if p3_families else p1_families[0],
        },
    }


# Keep PILLAR_DEFINITIONS as module-level alias for backwards compatibility
PILLAR_DEFINITIONS = _DEFAULT_PILLAR_DEFINITIONS


class CPCHypothesisResolver:
    """
    Phase 5: Deterministic hypothesis resolver.

    Evaluates Phase 4 hypotheses as whole units and selects the best.
    """

    def __init__(self, use_llm_tiebreak: bool = False, llm_client=None):
        self.use_llm_tiebreak = use_llm_tiebreak
        self.llm = llm_client

    def resolve(
        self,
        phase4_result: Dict[str, Any],
        phase1_data: Dict[str, Any],
        all_raw_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve Phase 4 hypotheses into final CPC selection + Tri-Pillar classification.

        Args:
            phase4_result: Output from Phase 4 consolidation
            phase1_data: Phase 1 semantic extraction
            all_raw_candidates: Optional Phase 2C raw scored candidates for pillar back-scanning

        Returns:
            Structured output with primary, optional secondary, pillars, and decision logic
        """
        hypotheses = phase4_result.get("phase4_hypotheses", [])

        if not hypotheses:
            logger.warning("Phase 5: No hypotheses to resolve")
            return self._empty_result()

        logger.info("Phase 5: Resolving %d hypotheses", len(hypotheses))

        # Build symbol→title lookup from the raw candidate pool so _score_hypothesis
        # can access real CPC titles even though supporting_codes stores only strings.
        symbol_to_title: Dict[str, str] = {}
        for c in (all_raw_candidates or []):
            sym = c.get("symbol", "")
            title = c.get("title", "") or c.get("full_context", "")
            if sym and title:
                symbol_to_title[sym] = title

        # Step 1: Score each hypothesis
        scored_hypotheses = []
        for hyp in hypotheses:
            scores = self._score_hypothesis(hyp, phase1_data, symbol_to_title)
            scored_hypotheses.append({**hyp, **scores})

        # Step 2: Rank by final_score
        ranked = sorted(scored_hypotheses, key=lambda x: x["final_score"], reverse=True)

        # Step 3: Select primary
        primary = ranked[0]

        # Step 4: Optionally select secondary
        secondary = None
        if len(ranked) > 1:
            gap = primary["final_score"] - ranked[1]["final_score"]
            if gap < 0.25:
                secondary = ranked[1]
                logger.info(
                    "Phase 5: Secondary accepted (gap=%.3f < 0.25): %s",
                    gap,
                    secondary["family"],
                )
            else:
                logger.info("Phase 5: Secondary rejected (gap=%.3f >= 0.25)", gap)

        # Step 5: Build result
        result = {
            "primary": {
                "family": primary["family"],
                "final_score": round(primary["final_score"], 4),
                "phase4_score": round(
                    primary.get("normalized_score", primary.get("score", 0)), 4
                ),
                "functional_alignment": round(primary["functional_alignment"], 4),
                "technical_coverage": round(primary["technical_coverage"], 4),
                "specificity_match": round(primary["specificity_match"], 4),
                "confidence": self._score_to_confidence(primary["final_score"]),
                "reasoning": primary.get("reasoning", ""),
                "supporting_codes": primary.get("supporting_codes", []),
            },
            "decision_logic": {
                "score_gap": round(
                    primary["final_score"]
                    - (secondary["final_score"] if secondary else 0),
                    4,
                ),
                "was_tiebreak_needed": False,
                "num_hypotheses_evaluated": len(ranked),
                "selection_method": "deterministic_scoring",
            },
        }

        if secondary:
            result["secondary"] = {
                "family": secondary["family"],
                "final_score": round(secondary["final_score"], 4),
                "phase4_score": round(
                    secondary.get("normalized_score", secondary.get("score", 0)), 4
                ),
                "functional_alignment": round(secondary["functional_alignment"], 4),
                "technical_coverage": round(secondary["technical_coverage"], 4),
                "specificity_match": round(secondary["specificity_match"], 4),
                "confidence": self._score_to_confidence(secondary["final_score"]),
                "reasoning": secondary.get("reasoning", ""),
                "supporting_codes": secondary.get("supporting_codes", []),
            }
            result["decision_logic"]["secondary_accepted"] = True
        else:
            result["decision_logic"]["secondary_accepted"] = False

        logger.info(
            "Phase 5: Primary=%s (score=%.3f), Secondary=%s, Gap=%.3f",
            result["primary"]["family"],
            result["primary"]["final_score"],
            result.get("secondary", {}).get("family", "None"),
            result["decision_logic"]["score_gap"],
        )

        # ─────────────────────────────────────────────────────────────
        # Step 6: Tri-Pillar Resolution — back-scan raw candidates for
        #         highest-scoring champion per functional role.
        #         Always run even if pool is empty (shows fallback targets).
        # ─────────────────────────────────────────────────────────────
        raw_pool = all_raw_candidates if all_raw_candidates is not None else []
        phase2a_families = phase1_data.get("phase2a_families", [])
        result["pillars"] = self._resolve_pillars(raw_pool, phase1_data, phase2a_families)

        return result

    def _resolve_pillars(
        self,
        all_raw_candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        phase2a_families: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find the highest-scoring CPC champion for each of the three pillars.

        Back-scans the full Phase 2C raw candidate pool (pre-2D filtering).
        For each pillar family (e.g., G06F, G06N, G05B), picks the top-scoring
        candidate whose symbol starts with that family prefix.

        Falls back to the ranked candidates from Phase 3 if no raw candidate found.
        """
        pillars = {}
        pillar_defs = _build_pillar_definitions(phase2a_families or [])

        logger.info(
            "Phase 5 Tri-Pillar: Pool size=%d, first 3 symbols=%s, phase2a_families=%s",
            len(all_raw_candidates),
            [c.get("symbol", "")[:12] for c in all_raw_candidates[:3]],
            phase2a_families,
        )

        used_symbols: set = set()
        for pillar_key, pillar_def in pillar_defs.items():
            families = pillar_def["families"]
            label = pillar_def["label"]
            desc = pillar_def["description"]

            # Scan raw candidates — exclude symbols already claimed by earlier pillars
            # so Primary and Application facets don't return the same subgroup.
            champion = self._find_champion_in_pool(
                all_raw_candidates, families, exclude=used_symbols
            )

            if champion:
                used_symbols.add(champion.get("symbol", ""))
                pillars[pillar_key] = {
                    "symbol": champion.get("symbol", ""),
                    "title": champion.get("title", ""),
                    "score": champion.get("score", 0),
                    "family": families[0] if families else "",
                    "label": label,
                    "description": desc,
                    "source": "phase2c_back_scan",
                }
                logger.info(
                    "Pillar '%s': Champion=%s (score=%.4f, source=back_scan)",
                    pillar_key,
                    champion.get("symbol", ""),
                    champion.get("score", 0),
                )
            else:
                # Fallback: placeholder — could trigger LLM lookup
                fallback_family = pillar_def.get(
                    "fallback_family", families[0] if families else ""
                )
                pillars[pillar_key] = {
                    "symbol": "",
                    "title": f"Target: {fallback_family} subclass (not found in candidate pool)",
                    "score": 0,
                    "family": fallback_family,
                    "label": label,
                    "description": desc,
                    "source": "not_found",
                }
                logger.warning(
                    "Pillar '%s': No champion found in %s families %s",
                    pillar_key,
                    len(all_raw_candidates),
                    families,
                )

        standard_count = sum(
            1 for v in pillars.values() if v.get("source") != "not_found"
        )
        logger.info(
            "Phase 5 Tri-Pillar: Found %d/3 pillar champions",
            standard_count,
        )

        return pillars

    @staticmethod
    def _find_champion_in_pool(
        candidates: List[Dict[str, Any]],
        families: List[str],
        exclude: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the highest-scoring candidate whose symbol starts with
        any of the given family prefixes.
        """
        excluded = exclude or set()
        matches = [
            c
            for c in candidates
            if any(c.get("symbol", "").startswith(fam) for fam in families)
            # Exclude CPC cross-reference/indexing codes (e.g. G10L2015/xxxx) — not allocatable
            and not _CROSS_REF_PATTERN.match(c.get("symbol", ""))
            # Exclude symbols already claimed by earlier pillars
            and c.get("symbol", "") not in excluded
        ]
        if not matches:
            return None
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        return matches[0]

    def _score_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        phase1_data: Dict[str, Any],
        symbol_to_title: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Score a hypothesis against Phase 1 context.

        Computes:
        - functional_alignment (0-1)
        - technical_coverage (0-1)
        - specificity_match (0-1)
        - final_score = 0.5*phase4_score + 0.3*functional_alignment + 0.2*technical_coverage
        """
        phase4_score = hypothesis.get("normalized_score", hypothesis.get("score", 0))

        # Get terms and context from Phase 1
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        core_function = phase1_data.get("core_function", "").lower()
        technical_object = phase1_data.get("technical_object", "").lower()
        system_context = phase1_data.get("system_context", "").lower()

        family = hypothesis.get("family", "")

        # Build title text from supporting_codes.
        # Codes are stored as plain strings — look up real CPC titles from the
        # symbol_to_title map built from the raw Phase 2C candidate pool.
        supporting_codes = hypothesis.get("supporting_codes", [])
        candidate_titles = []
        for code in supporting_codes:
            if isinstance(code, dict):
                t = code.get("title", "")
            else:
                t = (symbol_to_title or {}).get(str(code), "")
            if t:
                candidate_titles.append(t.lower())

        # Fallback to reasoning text if no titles found (sparse KG)
        if not candidate_titles:
            candidate_titles = [hypothesis.get("reasoning", "").lower()]

        all_titles = " ".join(candidate_titles)

        # Compute functional alignment
        functional_alignment = self._compute_functional_alignment(
            core_function, technical_object, all_titles
        )

        # Compute technical coverage
        technical_coverage = self._compute_technical_coverage(terms, all_titles)

        # Compute specificity match
        specificity_match = self._compute_specificity_match(hypothesis, phase1_data)

        # Final score
        final_score = (
            0.5 * min(phase4_score, 1.0)
            + 0.3 * functional_alignment
            + 0.2 * technical_coverage
        )

        logger.debug(
            "Hypothesis %s: phase4=%.3f, func_align=%.3f, tech_cov=%.3f, "
            "spec=%.3f, final=%.3f",
            family,
            phase4_score,
            functional_alignment,
            technical_coverage,
            specificity_match,
            final_score,
        )

        return {
            "functional_alignment": functional_alignment,
            "technical_coverage": technical_coverage,
            "specificity_match": specificity_match,
            "final_score": final_score,
        }

    def _compute_functional_alignment(
        self, core_function: str, technical_object: str, titles: str
    ) -> float:
        """
        Compute how well hypothesis titles align with the invention's core function.

        Uses keyword overlap between Phase 1 function and CPC titles.
        """
        # Use regex to extract clean alpha words — avoids punctuation like "speech," ≠ "speech"
        function_words = set(re.findall(r'[a-zA-Z]+', (core_function + " " + technical_object).lower()))
        title_words = set(re.findall(r'[a-zA-Z]+', titles.lower()))

        if not function_words or not title_words:
            return 0.5

        # Filter to meaningful words
        stopwords = {
            "the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
            "with", "by", "from", "as", "is", "are", "be", "being", "been",
            "method", "apparatus", "system", "device", "comprising",
        }
        function_keywords = {
            w for w in function_words if len(w) > 3 and w not in stopwords
        }
        title_keywords = {w for w in title_words if len(w) > 3 and w not in stopwords}

        if not function_keywords:
            return 0.5

        overlap = function_keywords & title_keywords
        alignment = len(overlap) / len(function_keywords)

        # Boost for strong matches
        return min(1.0, alignment * 1.5)

    def _compute_technical_coverage(self, terms: List[Any], titles: str) -> float:
        """
        Compute how many extracted terms are covered by the hypothesis.

        Counts term matches in CPC titles.
        """
        if not terms:
            return 0.5

        title_lower = titles.lower()
        matched = 0
        total_weight = 0

        for term_obj in terms:
            if isinstance(term_obj, dict):
                term = term_obj.get("term", "").lower()
                weight = term_obj.get("importance", 5) / 10.0
            else:
                term = str(term_obj).lower()
                weight = 0.5

            if term and len(term) > 2:
                total_weight += weight
                if term in title_lower:
                    # Exact phrase match — full credit
                    matched += weight
                elif " " in term:
                    # Multi-word term: give partial credit proportional to
                    # how many of its words appear individually in the titles.
                    # E.g. "feature extraction" → "feature" + "extraction" both in titles → 0.7×
                    term_words = [w for w in term.split() if len(w) > 2]
                    if term_words:
                        hits = sum(1 for w in term_words if w in title_lower)
                        matched += weight * (hits / len(term_words)) * 0.7

        if total_weight == 0:
            return 0.5

        coverage = matched / total_weight
        return min(1.0, coverage * 1.2)

    def _compute_specificity_match(
        self, hypothesis: Dict[str, Any], phase1_data: Dict[str, Any]
    ) -> float:
        """
        Compute how specific the hypothesis is to the invention.

        Prefers clusters with high coherence and appropriate size.
        """
        coherence = hypothesis.get("coherence", 1.0)
        count = hypothesis.get("candidate_count", 1)

        # Optimal cluster size: 2-5 candidates
        size_score = 1.0
        if count == 1:
            size_score = 0.7  # Too small
        elif count > 6:
            size_score = 0.85  # Too broad

        return coherence * 0.6 + size_score * 0.4

    def _score_to_confidence(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.75:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when no hypotheses available."""
        return {
            "primary": {
                "family": "",
                "final_score": 0.0,
                "phase4_score": 0.0,
                "functional_alignment": 0.0,
                "technical_coverage": 0.0,
                "specificity_match": 0.0,
                "confidence": "low",
                "reasoning": "No hypotheses available for resolution.",
                "supporting_codes": [],
            },
            "decision_logic": {
                "score_gap": 0.0,
                "was_tiebreak_needed": False,
                "num_hypotheses_evaluated": 0,
                "selection_method": "none",
                "secondary_accepted": False,
            },
        }


def resolve_cpc_hypotheses(
    phase4_result: Dict[str, Any],
    phase1_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Quick function to resolve Phase 4 hypotheses."""
    resolver = CPCHypothesisResolver()
    return resolver.resolve(phase4_result, phase1_data)
