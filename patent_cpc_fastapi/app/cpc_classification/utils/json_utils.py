"""json_utils.py — JSON parsing with fallback strategies for LLM responses."""

import json
import re
from typing import Any, Dict


def parse_llm_json(response: Any) -> Dict[str, Any]:
    """Parse JSON from LLM response with multiple fallback strategies."""
    if not response:
        return {}
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except Exception:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}
