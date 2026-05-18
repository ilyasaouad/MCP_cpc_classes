"""Phase 1B prompts â€” claim-to-domain forensic audit."""

try:
    from cpc_classification.prompts.prompt_phase1_2 import (
        phase1_2_audit_prompt,
    )
    __all__ = ["phase1_2_audit_prompt"]
except ImportError:
    pass
