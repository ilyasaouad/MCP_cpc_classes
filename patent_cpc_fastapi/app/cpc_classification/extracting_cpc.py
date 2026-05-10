import json
import os
import re
from typing import Dict, Any, List
from search_core.ollama_client import OllamaClient
from .prompts import phase1_prompt


def _normalize_terms(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize term extraction from both description_terms and claims_terms.
    Claims terms get 2x weight multiplier.
    """
    terms = []

    # New format: description_terms + claims_terms
    desc_raw = data.get("description_terms", [])
    claims_raw = data.get("claims_terms", [])

    # Process description terms (normal weight)
    if isinstance(desc_raw, list):
        for item in desc_raw:
            if isinstance(item, dict) and "term" in item:
                terms.append(
                    {
                        "term": item.get("term", ""),
                        "importance": item.get("importance", 5),
                        "justification": item.get("justification", ""),
                        "source": item.get("source", "description"),
                    }
                )

    # Process claims terms (2x weight)
    if isinstance(claims_raw, list):
        for item in claims_raw:
            if isinstance(item, dict) and "term" in item:
                # Claims terms get 2x weight
                base_importance = item.get("importance", 5)
                terms.append(
                    {
                        "term": item.get("term", ""),
                        "importance": min(base_importance * 2, 10),  # Cap at 10
                        "justification": item.get("justification", ""),
                        "source": "claims",
                    }
                )

    # Legacy fallback: old "essential_terms" key
    if not terms:
        raw = data.get("essential_terms", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "term" in item:
                    terms.append(
                        {
                            "term": item.get("term", ""),
                            "importance": item.get("importance", 5),
                            "justification": item.get("justification", ""),
                            "source": item.get("source", "description"),
                        }
                    )

    # Legacy fallback: old "terms" key
    if not terms:
        raw = data.get("terms", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "term" in item:
                    terms.append(
                        {
                            "term": item.get("term", ""),
                            "importance": item.get("importance", 5),
                            "justification": "",
                            "source": item.get("source", "description"),
                        }
                    )

    # Sort by importance descending, then alphabetically
    terms.sort(key=lambda x: (-x["importance"], x["term"]))
    return terms


class CPCExtractor:
    """
    Phase 1:
    Extract semantic understanding (terms, function, domains) from patent text.
    NO CPC class prediction - that is Phase 2's responsibility.
    """

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def extract(self, description: str, labeled_claims: str) -> Dict[str, Any]:
        # Truncate long texts to prevent LLM timeout
        # Keep claims intact (most important), truncate description
        MAX_DESC_CHARS = 3000  # ~750 tokens
        MAX_CLAIMS_CHARS = 2000  # ~500 tokens

        truncated_desc = (
            description[:MAX_DESC_CHARS]
            if len(description) > MAX_DESC_CHARS
            else description
        )
        truncated_claims = (
            labeled_claims[:MAX_CLAIMS_CHARS]
            if len(labeled_claims) > MAX_CLAIMS_CHARS
            else labeled_claims
        )

        if len(description) > MAX_DESC_CHARS:
            truncated_desc += "\n\n[...description truncated for length...]"

        prompt = phase1_prompt(truncated_claims, truncated_desc)

        response = self.llm.chat(
            system_prompt=prompt,
            user_message="Please analyze the patent and produce the structured JSON output.",
            temperature=0.1,
            max_tokens=4000,
        )

        if not response:
            raise RuntimeError(
                "Phase 1 extraction failed: LLM returned empty response. "
                "This usually means:\n"
                "1. Ollama is not running (start with: ollama serve)\n"
                "2. The model is not loaded (pull with: ollama pull phi4:latest)\n"
                "3. The model is too large for your GPU/CPU (try a smaller model)\n"
                "4. The request timed out (120B models need 5-10 minutes)"
            )

        # 1. Try parsing the whole thing first
        try:
            data = json.loads(response)
        except Exception:
            data = None

        # 2. Strip markdown fences and try again
        if data is None:
            cleaned = re.sub(
                r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                data = json.loads(cleaned)
            except Exception:
                data = None

        # 3. Fallback: greedy regex (legacy behaviour)
        if data is None:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None

        if data is None:
            return {"raw": response}

        # Normalize and sort terms
        data["essential_terms"] = _normalize_terms(data)

        # Backward compatibility: keep legacy keys if present
        if "terms" not in data and "essential_terms" in data:
            data["terms"] = data["essential_terms"]

        return data
