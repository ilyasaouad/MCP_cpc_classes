import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from search_core.ollama_client import OllamaClient
from .prompts import (
    phase1_prompt,
    phase1_reconciliation_prompt,
    score_phase1_completeness,
)

logger = logging.getLogger(__name__)


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

    def extract(
        self,
        description: str,
        labeled_claims: str,
        drawing_descriptions: str = "",
    ) -> Dict[str, Any]:
        """
        Phase 1: Multi-pass semantic extraction.

        Runs 4 independent passes (A/B/C/D), reconciles results, and scores
        completeness.  Uses the new multi-pass prompt architecture from prompts.py.
        """
        MAX_DESC_CHARS = 3000
        MAX_CLAIMS_CHARS = 2000
        MAX_DRAWINGS_CHARS = 2000

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
        truncated_drawings = (
            drawing_descriptions[:MAX_DRAWINGS_CHARS]
            if len(drawing_descriptions) > MAX_DRAWINGS_CHARS
            else drawing_descriptions
        )

        if len(description) > MAX_DESC_CHARS:
            truncated_desc += "\n\n[...description truncated for length...]"

        # Get all pass prompts
        prompts: Dict[str, Any] = phase1_prompt(
            truncated_claims, truncated_desc, truncated_drawings
        )

        pass_names = ["pass_a", "pass_b", "pass_c"]
        if prompts.get("pass_d"):
            pass_names.append("pass_d")

        pass_results: Dict[str, Dict[str, Any]] = {}

        # Run each pass independently
        for pname in pass_names:
            prompt_text = prompts[pname]
            try:
                response = self.llm.chat(
                    system_prompt=prompt_text,
                    user_message="Produce ONLY the JSON output as specified.",
                    temperature=0.1,
                    max_tokens=4000,
                )
                parsed = self._parse_json_response(response)
                if parsed:
                    pass_results[pname] = parsed
                    logger.info("Phase 1 %s: OK", pname)
                else:
                    logger.warning("Phase 1 %s: failed to parse response", pname)
            except Exception as e:
                logger.warning("Phase 1 %s: LLM error: %s", pname, e)

        if not pass_results:
            raise RuntimeError(
                "Phase 1 extraction failed: all passes returned empty. "
                "Check LLM availability and model."
            )

        # Reconcile passes into final output
        reconciliation_prompt = phase1_reconciliation_prompt(
            pass_results.get("pass_a", {}),
            pass_results.get("pass_b", {}),
            pass_results.get("pass_c", {}),
            pass_results.get("pass_d"),
            truncated_claims,
        )

        try:
            response = self.llm.chat(
                system_prompt=reconciliation_prompt,
                user_message="Produce ONLY the JSON output as specified.",
                temperature=0.1,
                max_tokens=6000,
            )
            data = self._parse_json_response(response)
        except Exception as e:
            logger.warning("Phase 1 reconciliation: LLM error: %s", e)
            data = None

        if data is None:
            # Fallback: return pass results directly
            data = pass_results.get("pass_b", pass_results.get("pass_a", {}))
            data["_reconciliation_failed"] = True

        # Score completeness
        completeness = score_phase1_completeness(data, truncated_claims)
        data["phase1_completeness"] = completeness
        logger.info(
            "Phase 1 completeness: %s (%d/100) — %s",
            completeness["status"],
            completeness["score"],
            "; ".join(completeness["issues"])
            if completeness["issues"]
            else "no issues",
        )

        # Normalize and sort terms
        data["essential_terms"] = _normalize_terms(data)

        # Backward compatibility: keep legacy keys
        if "terms" not in data and "essential_terms" in data:
            data["terms"] = data["essential_terms"]

        # Extract claims if not already parsed
        if "claims" not in data:
            data["claims"] = self._extract_claims_from_labeled(truncated_claims)

        return data

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response with fallbacks."""
        if not response:
            return None

        # 1. Try raw parse
        try:
            return json.loads(response)  # type: ignore[no-any-return]
        except Exception:
            pass

        # 2. Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except Exception:
            pass

        # 3. Greedy regex
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))  # type: ignore[no-any-return]
            except Exception:
                pass

        return None

    @staticmethod
    def _extract_claims_from_labeled(text: str) -> List[str]:
        """Extract individual claims from labeled text."""
        claims = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[INDEPENDENT]") or stripped.startswith(
                "[DEPENDENT"
            ):
                # Collect claim text lines until next label
                pass
            elif stripped and claims:
                # First non-label line after a claim starts
                continue
        # Simple version: collect lines after labels
        current = []
        for line in text.splitlines():
            if line.strip().startswith("[INDEPENDENT]") or line.strip().startswith(
                "[DEPENDENT"
            ):
                if current:
                    claims.append(" ".join(current))
                current = [line.strip()]
            elif line.strip() and current:
                current.append(line.strip())
        if current:
            claims.append(" ".join(current))
        return claims
