"""
prompts.py — Re-export hub for all prompt modules.

All prompt functions live in /prompts/ sub-package:
  prompts/shared.py                     — label_claims, detect_sections, UNIFIED_IMPORTANCE_RUBRIC
  prompts/prompt_phase1.py              — Phase 1 multi-pass extraction + reconciliation
  prompts/prompt_phases_5_6.py          — Phase 5 single-candidate validation + Phase 6 claim reconciliation
  prompts/prompt_phases_35_4_7.py       — Phase 3.5 tie-breaker, Phase 4 sanity check, Phase 7 consistency
  prompts/prompt_phase8.py              — Phase 8 executive report + at-a-glance card prompts

This module re-exports everything with backward-compatible names.
"""

from .prompts.shared import label_claims, detect_sections, UNIFIED_IMPORTANCE_RUBRIC

from .prompts.prompt_phase1 import (
    phase1_pass_a_prompt,
    phase1_pass_b_prompt,
    phase1_pass_c_prompt,
    phase1_pass_d_prompt,
    phase1_reconciliation_prompt,
    score_phase1_completeness,
    phase1_prompt,
    domain_inference_prompt,
)

from .prompts.prompt_phases_5_6 import (
    validation_prompt_single,
    reconciliation_prompt,
)

from .prompts.prompt_phases_35_4_7 import (
    phase35_tiebreaker_prompt,
    phase4_sanity_check_prompt,
    phase7_consistency_prompt,
)

from .prompts.prompt_phase8 import (
    phase8_report_prompt,
    phase85_card_prompt,
)

# Backward-compatible alias
consistency_check_prompt = phase7_consistency_prompt
