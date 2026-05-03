import json
import re


def safe_json_parse(text: str) -> dict:
    """
    Safely parse LLM JSON output by removing markdown fences.
    """
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        return {}