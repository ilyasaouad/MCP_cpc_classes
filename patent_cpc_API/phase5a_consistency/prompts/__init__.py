"""Phase 5A prompts â€” LLM consistency check."""

try:
    from cpc_classification.prompts.prompt_phases_35_4_7 import (
        phase7_consistency_prompt,
    )
    __all__ = ["phase7_consistency_prompt"]
except ImportError:
    pass
