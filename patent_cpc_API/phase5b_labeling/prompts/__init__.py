"""Phase 5B prompts â€” professional labeling and justification."""

try:
    from cpc_classification.prompts.prompt_phase8 import (
        phase8_labeling_prompt,
    )
    __all__ = ["phase8_labeling_prompt"]
except ImportError:
    pass
