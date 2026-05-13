import logging

from ..cpc_decision_tree import CPCDecisionTreeConstraint
from ..cpc_cross_domain_validator import CrossDomainValidator

logger = logging.getLogger(__name__)


def run_phase3(classifier, candidates, phase1, layer_result, tcr_result):
    ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

    phase35_result = {}
    try:
        dt_constraint = CPCDecisionTreeConstraint()
        phase35_result = dt_constraint.apply_constraints(
            ranked, phase1, layer_result, tcr_result
        )
        ranked = phase35_result.get("phase35_candidates", ranked)
        logger.info(
            "Phase 3.5: Applied %d constraint rules. Domain=%s (conf=%.2f). Layer-mode=%s",
            phase35_result.get("phase35_adjustments", 0),
            phase35_result.get("phase35_domain", "unknown"),
            phase35_result.get("phase35_domain_confidence", 0),
            phase35_result.get("phase35_layer_mode", False),
        )
    except Exception as e:
        logger.warning("Phase 3.5 decision tree failed: %s", e)

    phase36_result = {}
    try:
        validator = CrossDomainValidator()
        phase36_result = validator.validate(
            ranked,
            phase1,
            phase35_result,
        )
        ranked = phase36_result.get("phase36_candidates", ranked)
        logger.info(
            "Phase 3.6: Cross-domain validation complete. Verified=%s, adjustments=%d",
            phase36_result.get("phase36_domain_verified", False),
            phase36_result.get("phase36_adjustments", 0),
        )
    except Exception as e:
        logger.warning("Phase 3.6 cross-domain validation failed: %s", e)

    return ranked, phase35_result, phase36_result
