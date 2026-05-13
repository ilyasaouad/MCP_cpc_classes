import logging

from ..cpc_role_classifier import CPCRoleClassifier, apply_role_scoring
from ..technical_weight_analyzer import TechnicalWeightAnalyzer

logger = logging.getLogger(__name__)


def run_phase15_tcr(classifier, phase1):
    phase15_result = {}
    try:
        role_classifier = CPCRoleClassifier(classifier.llm)
        phase15_result = role_classifier.classify_role(phase1)
        logger.info(
            "Phase 1.5: Role=%s (conf=%.2f)",
            phase15_result.get("role", "UNKNOWN"),
            phase15_result.get("confidence", 0),
        )
    except Exception as e:
        logger.warning("Phase 1.5 role classification failed: %s", e)
        phase15_result = {"role": "SYSTEM", "confidence": 0.5}

    tcr_result = {}
    try:
        tcr_analyzer = TechnicalWeightAnalyzer()
        tcr_result = tcr_analyzer.analyze(phase1)
        logger.info(
            "TCR Analysis: TCR=%.3f, force_flag=%s, comp_weight=%.2f, phys_weight=%.2f",
            tcr_result.get("tcr", 1.0),
            tcr_result.get("force_flag", "HYBRID_INVENTION"),
            tcr_result.get("computational_weight", 0),
            tcr_result.get("physical_weight", 0),
        )
    except Exception as e:
        logger.warning("Technical weight analysis failed: %s", e)
        tcr_result = {"tcr": 1.0, "force_flag": "HYBRID_INVENTION"}

    return phase15_result, tcr_result
