import logging

from ..prompts.prompt_phase1 import score_phase1_completeness

logger = logging.getLogger(__name__)


def run_phase1(classifier, description, labeled_claims):
    try:
        phase1 = classifier.extractor.extract(description, labeled_claims)

        if isinstance(phase1, dict) and phase1.get("technical_object"):
            completeness = score_phase1_completeness(phase1, labeled_claims)
            phase1["phase1_completeness"] = completeness
            status = completeness["status"]
            score = completeness["score"]
            issues = "; ".join(completeness["issues"])
            if status == "FAIL":
                logger.warning(
                    "Phase 1 completeness FAIL (%d/100): %s — downstream may be unreliable",
                    score,
                    issues,
                )
            elif status == "WARN":
                logger.info("Phase 1 completeness WARN (%d/100): %s", score, issues)
            else:
                logger.info("Phase 1 completeness PASS (%d/100)", score)
    except Exception as e:
        logger.error("Phase 1 extraction failed: %s", str(e))
        return {
            "error": str(e),
            "phase1": {},
            "phase2": {},
            "phase3": [],
            "phase4": {},
            "phase5": {},
            "cpc": [],
        }

    if (
        not phase1
        or not phase1.get("technical_object")
        or not phase1.get("core_function")
    ):
        error_msg = (
            "Phase 1 extraction returned incomplete data. "
            "The LLM did not produce valid semantic extraction. "
            "Possible causes: model timeout, invalid JSON response, or model not loaded."
        )
        logger.error(error_msg)
        return {
            "error": error_msg,
            "phase1": phase1 if phase1 else {},
            "phase2": {},
            "phase3": [],
            "phase4": {},
            "phase5": {},
            "cpc": [],
        }

    return phase1
