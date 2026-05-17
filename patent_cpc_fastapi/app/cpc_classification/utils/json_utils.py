"""json_utils.py — JSON parsing with fallback strategies for LLM responses."""

import json
import re
from typing import Any, Dict


_CPC_ANCHOR_RE = re.compile(r"^[A-H][0-9]{2}[A-Z]$")


def _normalize_json_keywords(text: str) -> str:
    """Normalize uppercase JSON keywords (NULL, TRUE, FALSE) to lowercase."""
    text = re.sub(r":\s*NULL([\s,\n\r\}])", r": null\1", text)
    text = re.sub(r":\s*TRUE([\s,\n\r\}])", r": true\1", text)
    text = re.sub(r":\s*FALSE([\s,\n\r\}])", r": false\1", text)
    return text


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
    cleaned = _normalize_json_keywords(cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        cleaned_match = _normalize_json_keywords(match.group(0))
        try:
            return json.loads(cleaned_match)
        except Exception:
            pass
    return {}


def parse_phase1_1_anchor(response: Any) -> Dict[str, Any]:
    """Parse Phase 1.1 emergency anchor response with strict validation.

    Validates the emergency_anchor against ^[A-H][0-9]{2}[A-Z]$.
    If the LLM returns a longer code (e.g. G06F 17/00), truncates to
    the first 4 characters.

    If parsing or validation fails, returns a hard-coded fallback (G06F)
    so the API never returns an error.
    """
    parsed = parse_llm_json(response)

    raw_anchor = parsed.get("emergency_anchor", "")
    confidence = parsed.get("confidence", 0.5)
    reasoning = parsed.get("reasoning", "")

    if raw_anchor and len(raw_anchor) >= 4:
        anchor = raw_anchor[:4]
    else:
        anchor = raw_anchor

    if _CPC_ANCHOR_RE.match(anchor):
        return {
            "emergency_anchor": anchor,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    return {
        "emergency_anchor": "G06F",
        "confidence": 0.3,
        "reasoning": "Hard-coded fallback: LLM Phase 1.1 output was malformed. "
        f"Raw response: {raw_anchor}",
    }
