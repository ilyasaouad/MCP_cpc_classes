import logging

from ..cpc_decision_tree import CPCDecisionTreeConstraint
from ..cpc_cross_domain_validator import CrossDomainValidator

logger = logging.getLogger(__name__)


def run_phase3(classifier, candidates, phase1, layer_result, tcr_result):
    ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

    # Phase 3.5 always runs — it handles domain disambiguation, functional boosting,
    # hierarchy priority, and invalid-class filtering across all patent fields.
    # Especially valuable for ambiguous cross-domain patents (e.g. mechanical+software,
    # medical+computing, automotive+AI) where Phase 2C scoring alone can drift.
    #
    # Could be skipped when:
    #   - All top-5 candidates share the same 4-char CPC family (single-domain dominance)
    #   - Phase 2A domain confidence >= 0.85 (already unambiguous)
    #   - Example: a pure G10L speech patent where Phase 2C already ranked correctly.
    # To add the short-circuit:
    #   top_families = {c.get("symbol","")[:4] for c in ranked[:5]}
    #   if len(top_families) == 1 and phase2a_confidence >= 0.85: skip
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
