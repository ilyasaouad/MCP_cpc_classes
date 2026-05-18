"""Phase 1A prompts â€” signal extraction passes A/B/C/D + reconciliation."""

try:
    from cpc_classification.prompts.prompt_phase1 import (
        phase1_pass_a_prompt,
        phase1_pass_b_prompt,
        phase1_pass_c_prompt,
        phase1_reconciliation_prompt,
        score_phase1_completeness,
    )
    __all__ = [
        "phase1_pass_a_prompt",
        "phase1_pass_b_prompt",
        "phase1_pass_c_prompt",
        "phase1_reconciliation_prompt",
        "score_phase1_completeness",
    ]
except ImportError:
    pass
