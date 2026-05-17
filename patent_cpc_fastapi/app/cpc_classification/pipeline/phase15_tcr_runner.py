import logging

from ..cpc_role_classifier import CPCRoleClassifier
from ..technical_weight_analyzer import TechnicalWeightAnalyzer

logger = logging.getLogger(__name__)

_VALID_ROLES = {"CORE_TECH", "SYSTEM", "APPLICATION", "SUPPORT"}


def run_phase15_tcr(classifier, phase1):
    phase15_result = {}
    try:
        role_classifier = CPCRoleClassifier(classifier.llm)
        phase15_result = role_classifier.classify_role(phase1)

        assert phase15_result["role"] in _VALID_ROLES, (
            f"Invalid role: {phase15_result['role']}. Must be one of {_VALID_ROLES}"
        )
        assert 0.0 < phase15_result["confidence"] <= 0.92, (
            f"Confidence {phase15_result['confidence']} out of range (0.0, 0.92]"
        )

        logger.info(
            "Phase 1.5: Role=%s (conf=%.2f)",
            phase15_result.get("role", "UNKNOWN"),
            phase15_result.get("confidence", 0),
        )
    except AssertionError:
        raise
    except Exception as e:
        logger.warning("Phase 1.5 role classification failed: %s", e)
        phase15_result = {"role": "SYSTEM", "confidence": 0.5}

    tcr_result = {}
    try:
        tcr_analyzer = TechnicalWeightAnalyzer()
        tcr_result = tcr_analyzer.analyze(
            phase1,
            phase1_5_result=phase15_result,
            domain_signals=phase1.get("domain_signals", []),
        )
        logger.info(
            "TCR Analysis: TCR=%.3f, bias=%.3f, comp_score=%.2f, phys_score=%.2f, force_mode=%s, override=%s",
            tcr_result.get("tcr", 1.0),
            tcr_result.get("tcr_bias", 0.0),
            tcr_result.get("computational_score", 0),
            tcr_result.get("physical_score", 0),
            tcr_result.get("force_mode", "unknown"),
            tcr_result.get("override_applied", False),
        )

        # TCR feedback: boost role confidence when TCR strongly corroborates the role.
        # FORCE_SOFTWARE_CORE + high TCR bias → confident CORE_TECH signal.
        tcr_bias = tcr_result.get("tcr_bias", 0.0)
        force_mode = tcr_result.get("force_mode", "")
        tcr_conf = tcr_result.get("confidence", 0.0)
        current_conf = phase15_result.get("confidence", 0.5)

        if (
            force_mode in ("FORCE_SOFTWARE_CORE", "FORCE_COMPUTATIONAL")
            and phase15_result.get("role") == "CORE_TECH"
            and tcr_bias >= 0.7
            and tcr_conf >= 0.85
        ):
            boosted = min(max(current_conf, 0.88), 0.92)
            if boosted > current_conf:
                logger.info(
                    "Phase 1.5: TCR corroboration boosted CORE_TECH confidence %.2f → %.2f",
                    current_conf, boosted,
                )
                phase15_result["confidence"] = boosted
                phase15_result["tcr_corroborated"] = True

    except Exception as e:
        logger.warning("Technical weight analysis failed: %s", e)
        tcr_result = {"tcr": 1.0, "tcr_bias": 0.0, "confidence": 0.0}

    return phase15_result, tcr_result
