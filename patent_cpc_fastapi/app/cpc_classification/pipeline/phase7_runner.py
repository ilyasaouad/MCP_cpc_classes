import logging
import re

from ..prompts.prompt_phases_35_4_7 import (
    phase7_consistency_prompt as consistency_check_prompt,
)
from ..utils.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


def run_phase7(
    classifier,
    phase1,
    phase5_result,
    ranked,
    candidates,
    phase4_result,
    tcr_result,
    term_importance=None,
):
    if term_importance is None:
        term_importance = {}

    # Build validated_candidates from Phase 5
    validated_candidates = []
    filtered_out = []
    best_code = None

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
            "symbol": validated_candidates[0]["symbol"] if validated_candidates else "",
            "title": validated_candidates[0]["title"] if validated_candidates else "",
            "confidence": primary.get("confidence", "high"),
            "reasoning": primary.get("reasoning", ""),
        }

    # ─────────────────────────────
    # FINAL CONSISTENCY CHECK
    # ─────────────────────────────
    consistency_result = {}
    try:
        if validated_candidates:
            selected = [
                {"symbol": v["symbol"], "title": v["title"]}
                for v in validated_candidates[:3]
            ]
            consist_prompt = consistency_check_prompt(phase1, selected)
            consist_response = classifier.llm.chat(
                system_prompt="You are performing a final coherence check on CPC classifications.",
                user_message=consist_prompt,
                temperature=0.1,
                max_tokens=1500,
            )
            consist_data = parse_llm_json(consist_response)
            consistency_result = {
                "coherent": consist_data.get("coherent", True),
                "issues": consist_data.get("issues", []),
                "recommended_primary": consist_data.get("recommended_primary", ""),
                "recommended_secondary": consist_data.get("recommended_secondary", []),
                "reasoning": consist_data.get("reasoning", ""),
            }
            logger.info(
                "Final Consistency: coherent=%s",
                consistency_result.get("coherent", True),
            )
    except Exception as e:
        logger.warning("Final consistency check failed: %s", e)

    # ─────────────────────────────
    # FEEDBACK LOOP
    # ─────────────────────────────
    feedback_applied = False
    feedback_constrained_families = []

    try:
        issues = consistency_result.get("issues", [])
        recommended_primary = consistency_result.get("recommended_primary", "")
        recommended_secondary = consistency_result.get("recommended_secondary", [])

        has_warnings = (
            not consistency_result.get("coherent", True)
            or len(issues) > 0
            or recommended_primary
        )

        if has_warnings and recommended_primary:
            logger.info(
                "Feedback loop triggered. Issues=%d, Recommended Primary=%s",
                len(issues),
                recommended_primary,
            )

            feedback_constrained_families = [recommended_primary[:4]]
            for sec in recommended_secondary:
                if sec and len(sec) >= 4:
                    feedback_constrained_families.append(sec[:4])

            for issue in issues:
                issue_str = str(issue).upper()
                family_patterns = re.findall(r"[A-Z]\d{2}[A-Z]?", issue_str)
                for fp in family_patterns:
                    if fp not in feedback_constrained_families:
                        feedback_constrained_families.append(fp)

            logger.info(
                "Feedback loop: Constrained families: %s",
                feedback_constrained_families,
            )

            if validated_candidates:
                original_count = len(validated_candidates)
                validated_candidates = [
                    c
                    for c in validated_candidates
                    if any(
                        c.get("symbol", "").startswith(f)
                        for f in feedback_constrained_families
                    )
                ]
                filtered_count = len(validated_candidates)

                if filtered_count < original_count:
                    logger.info(
                        "Feedback loop: Filtered candidates %d -> %d",
                        original_count,
                        filtered_count,
                    )

                if not validated_candidates:
                    primary_cpc = phase1.get("primary_domain", {}).get(
                        "cpc_class", ""
                    ) or phase1.get("primary_domain", {}).get("name", "")
                    primary_prefix = (
                        primary_cpc[:3] if len(primary_cpc) >= 3 else primary_cpc
                    )
                    validated_candidates = [
                        c
                        for c in ranked[:10]
                        if c.get("symbol", "").startswith(primary_prefix)
                    ] or ranked[:5]
                    logger.warning(
                        "Feedback loop: No candidates matched feedback filter. "
                        "Using top ranked from primary domain %s",
                        primary_prefix,
                    )

                feedback_applied = True

    except Exception as e:
        logger.warning("Phase 7.5 feedback loop failed: %s", e)

    # ─────────────────────────────
    # Premier resolution
    # ─────────────────────────────
    premier_data = classifier._resolve_premier(
        consistency_result, best_code, ranked, validated_candidates
    )

    return (
        consistency_result,
        validated_candidates,
        filtered_out,
        best_code,
        premier_data,
        {
            "feedback_applied": feedback_applied,
            "feedback_constrained_families": feedback_constrained_families,
        },
    )
