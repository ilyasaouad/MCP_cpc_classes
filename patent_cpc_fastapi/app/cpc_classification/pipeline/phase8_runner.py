import logging

from ..cpc_role_labeling import CPCRoleLabeling

logger = logging.getLogger(__name__)


def run_phase8(
    classifier,
    phase1,
    phase5_result,
    ranked,
    enhanced_candidates,
    consistency_result,
    tcr_result,
    phase36_result=None,
    premier_data=None,
):
    if phase36_result is None:
        phase36_result = {}

    # Build enhanced candidates list including Phase 7 recommendations
    enhanced = list(ranked)

    recommended_primary = consistency_result.get("recommended_primary", "")
    recommended_secondary = consistency_result.get("recommended_secondary", [])

    if recommended_primary:
        if isinstance(recommended_primary, dict):
            sym = recommended_primary.get("symbol", "")
        else:
            sym = recommended_primary

        if sym and not any(
            c.get("symbol", "").startswith(sym[:6]) for c in enhanced[:10]
        ):
            enhanced.insert(
                0,
                {
                    "symbol": sym,
                    "title": f"Recommended by Phase 7: {recommended_primary.get('title', 'Primary recommendation')}"
                    if isinstance(recommended_primary, dict)
                    else sym,
                    "score": 1.0,
                    "contribution_match": "primary",
                },
            )
            logger.info("Phase 8: Added Phase 7 recommended primary: %s", sym)

    for sec in recommended_secondary:
        if isinstance(sec, dict):
            sym = sec.get("symbol", "")
            title = sec.get("title", sym)
        else:
            sym = sec
            title = sec

        if sym and not any(
            c.get("symbol", "").startswith(sym[:6]) for c in enhanced[:10]
        ):
            enhanced.insert(
                1,
                {
                    "symbol": sym,
                    "title": f"Recommended by Phase 7: {title}",
                    "score": 0.95,
                    "contribution_match": "secondary",
                },
            )
            logger.info("Phase 8: Added Phase 7 recommended secondary: %s", sym)

    role_labeling_result = {}
    try:
        labeler = CPCRoleLabeling(llm=classifier.llm)
        role_labeling_result = labeler.label_roles(
            candidates=enhanced,
            phase1_data=phase1,
            phase36_result=phase36_result,
            tcr_result=tcr_result,
        )
        logger.info(
            "Phase 8: Role labeling complete. Core=%d, Support=%d, Context=%d, Coverage=%d",
            len(role_labeling_result.get("layer1_core", [])),
            len(role_labeling_result.get("layer2_support", [])),
            len(role_labeling_result.get("layer2_context", [])),
            len(role_labeling_result.get("layer3_coverage", [])),
        )
    except Exception as e:
        logger.warning("Phase 8 role labeling failed: %s", e)

    # Build formatted report
    formatted_report = classifier._build_formatted_report(
        phase1,
        role_labeling_result,
        consistency_result,
        tcr_result,
        pillars=phase5_result.get("pillars", {}),
        premier=premier_data,
    )

    return role_labeling_result, formatted_report
