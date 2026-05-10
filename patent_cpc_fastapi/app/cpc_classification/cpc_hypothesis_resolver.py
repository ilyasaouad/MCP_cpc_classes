"""
cpc_hypothesis_resolver.py - Phase 5: CPC Hypothesis Resolution

Deterministic resolver that selects the best CPC hypothesis from Phase 4 output.

KEY DESIGN:
- NOT a free-form classifier
- NOT validating individual CPC codes
- IS a deterministic scorer that ranks Phase 4 hypotheses
- LLM used ONLY for tie-breaking, not for classification

Constraints:
- MUST choose ONLY from Phase 4 hypotheses
- MUST NOT generate new CPC codes
- MUST select exactly 1 primary
- Secondary is optional (max 1, only if gap < 0.25)
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


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
    ) -> Dict[str, Any]:
        """
        Resolve Phase 4 hypotheses into final CPC selection.

        Args:
            phase4_result: Output from Phase 4 consolidation
            phase1_data: Phase 1 semantic extraction

        Returns:
            Structured output with primary, optional secondary, and decision logic
        """
        hypotheses = phase4_result.get("phase4_hypotheses", [])

        if not hypotheses:
            logger.warning("Phase 5: No hypotheses to resolve")
            return self._empty_result()

        logger.info("Phase 5: Resolving %d hypotheses", len(hypotheses))

        # Step 1: Score each hypothesis
        scored_hypotheses = []
        for hyp in hypotheses:
            scores = self._score_hypothesis(hyp, phase1_data)
            scored_hypotheses.append(
                {
                    **hyp,
                    **scores,
                }
            )

        # Step 2: Rank by final_score
        ranked = sorted(
            scored_hypotheses,
            key=lambda x: x["final_score"],
            reverse=True,
        )

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

        return result

    def _score_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        phase1_data: Dict[str, Any],
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

        # Get hypothesis data
        family = hypothesis.get("family", "")
        candidate_titles = [
            c.get("title", "").lower()
            for c in hypothesis.get("supporting_codes", [])
            if isinstance(c, dict)
        ]
        # If supporting_codes are strings, we don't have titles
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
        function_words = set(core_function.split()) | set(technical_object.split())
        title_words = set(titles.split())

        if not function_words or not title_words:
            return 0.5

        # Filter to meaningful words
        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "for",
            "and",
            "or",
            "in",
            "on",
            "to",
            "with",
            "by",
            "from",
            "as",
            "is",
            "are",
            "be",
            "being",
            "been",
            "method",
            "apparatus",
            "system",
            "device",
            "comprising",
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
                    matched += weight

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
