import logging

from ..cpc_hypothesis_consolidation import CPCHypothesisConsolidator
from ..cpc_hypothesis_resolver import CPCHypothesisResolver

logger = logging.getLogger(__name__)


def run_phase4_5(classifier, ranked, phase1, all_candidates, term_importance=None):
    if term_importance is None:
        term_importance = {}

    phase4_result = {}
    try:
        consolidator = CPCHypothesisConsolidator(max_hypotheses=3)
        phase4_result = consolidator.consolidate(
            ranked_candidates=ranked,
            term_importance=term_importance,
        )
        logger.info(
            "Phase 4: Consolidated %d candidates into %d hypotheses. Primary=%s, confidence=%s",
            len(ranked),
            len(phase4_result.get("phase4_hypotheses", [])),
            phase4_result.get("phase4_primary_family", "N/A"),
            phase4_result.get("phase4_confidence", "N/A"),
        )
    except Exception as e:
        logger.warning("Phase 4 consolidation failed: %s", e)

    phase5_result = {}
    validated_candidates = []
    filtered_out = []
    best_code = None

    try:
        resolver = CPCHypothesisResolver()
        phase5_result = resolver.resolve(phase4_result, phase1, all_candidates)

        primary = phase5_result.get("primary", {})
        if primary and primary.get("family"):
            for candidate in ranked:
                if candidate["symbol"].startswith(primary["family"]):
                    validated_candidates.append(
                        {
                            "symbol": candidate["symbol"],
                            "title": candidate["title"],
                            "validation": "PASS",
                            "confidence": primary.get("confidence", "high"),
                            "justification": primary.get("reasoning", ""),
                        }
                    )

            best_code = {
                "symbol": validated_candidates[0]["symbol"]
                if validated_candidates
                else "",
                "title": validated_candidates[0]["title"]
                if validated_candidates
                else "",
                "confidence": primary.get("confidence", "high"),
                "reasoning": primary.get("reasoning", ""),
            }

        logger.info(
            "Phase 5: Primary=%s (score=%.3f), Secondary=%s",
            primary.get("family", "N/A"),
            primary.get("final_score", 0),
            phase5_result.get("secondary", {}).get("family", "None"),
        )
    except Exception as e:
        logger.warning("Phase 5 resolution failed: %s", e)
        if ranked:
            validated_candidates = [
                {
                    "symbol": node["symbol"],
                    "title": node["title"],
                    "validation": "PASS",
                    "confidence": "medium",
                    "justification": "Resolution skipped due to error",
                }
                for node in ranked[:3]
            ]
            best_code = {
                "symbol": ranked[0]["symbol"],
                "title": ranked[0]["title"],
                "confidence": "medium",
                "reasoning": "Resolution skipped due to error",
            }

    return phase4_result, phase5_result
