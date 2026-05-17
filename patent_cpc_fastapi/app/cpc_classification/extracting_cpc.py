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
                    source = item.get(
                        "source", item.get("source_section", "description")
                    )
                    # Claims-derived terms get 2x importance boost
                    importance = item.get("importance", 5)
                    if source == "claims" or (
                        source != "description" and "claim" in source.lower()
                    ):
                        importance = min(importance * 2, 10)
                    terms.append(
                        {
                            "term": item.get("term", ""),
                            "importance": importance,
                            "justification": item.get("justification", ""),
                            "source": source,
                        }
                    )

    # Sort by importance descending, then alphabetically
    terms.sort(key=lambda x: (-x["importance"], x["term"]))
    return terms


# Domains that are generic infrastructure — confidence capped when a stronger
# subject-matter domain (G10L, G06V, H04L, A61B, etc.) is co-present.
_GENERIC_INFRASTRUCTURE_DOMAINS = {"G06F"}

# Domains considered subject-matter anchors (not just methodology/infrastructure)
_SUBJECT_MATTER_DOMAINS = {
    "G10L", "G06V", "G06T", "H04L", "H04W", "A61B",
    "B60W", "B60L", "G05B", "E21B", "G01N", "B01D",
}


def _normalize_domain_signals(signals: list) -> list:
    """
    Normalize domain signals to ensure backward compatibility.
    Handles: 'label' (new), 'name' (old), 'domain' (Pass C format).
    Ensures every signal has label, cpc_family, confidence, role.
    Also caps G06F confidence when a stronger subject-matter domain is present.
    """
    normalized = []
    for s in signals:
        if not isinstance(s, dict):
            continue

        # Normalize label/name/domain — prefer 'label' (new format)
        label = s.get("label") or s.get("name") or s.get("domain") or ""
        cpc_family = s.get("cpc_family", "")
        confidence = s.get("confidence", 0.0)
        role = s.get("role", "secondary")

        # Skip signals with no meaningful content
        if not label and not cpc_family:
            logger.warning(
                "[Phase 1] Domain signal has no label, name, or cpc_family: %s", s
            )
            continue

        # If label is empty but cpc_family exists, derive label from family
        if not label and cpc_family:
            label = f"CPC {cpc_family}"

        normalized.append(
            {
                "label": label,
                "name": label,
                "cpc_family": cpc_family,
                "confidence": confidence,
                "role": role,
                "evidence": s.get("evidence", ""),
                "convergence_count": s.get("convergence_count", 0),
                "passes_supporting": s.get("passes_supporting", []),
            }
        )

    # Cap generic infrastructure domains (G06F) when a subject-matter domain is present
    subject_max_conf = max(
        (s["confidence"] for s in normalized if s["cpc_family"] in _SUBJECT_MATTER_DOMAINS),
        default=0.0,
    )
    if subject_max_conf >= 0.80:
        for s in normalized:
            if s["cpc_family"] in _GENERIC_INFRASTRUCTURE_DOMAINS and s["confidence"] > 0.65:
                logger.warning(
                    "[Phase 1] Capping %s confidence %.2f → 0.65 "
                    "(generic infrastructure, subject-matter domain at %.2f present)",
                    s["cpc_family"], s["confidence"], subject_max_conf,
                )
                s["confidence"] = 0.65

    return normalized


def _normalize_domain_roles(signals: list) -> list:
    """
    Enforce correct role assignments on domain signals.

    The reconciliation prompt requires primary / secondary / negative roles, but
    LLMs sometimes label everything 'secondary'.  This function:
      1. Promotes the highest-confidence non-negative signal to 'primary' if
         no primary exists.
      2. Prefers a subject-matter domain (G10L, G06V, …) over a methodology
         domain (G06N) for the primary slot.
    """
    positive = [s for s in signals if s.get("role") != "negative"]
    has_primary = any(s.get("role") == "primary" for s in positive)

    if not has_primary and positive:
        # Prefer a subject-matter anchor over G06N/G06F for the primary slot
        subject_candidates = [s for s in positive if s["cpc_family"] in _SUBJECT_MATTER_DOMAINS]
        candidate = (
            max(subject_candidates, key=lambda x: x.get("confidence", 0))
            if subject_candidates
            else max(positive, key=lambda x: x.get("confidence", 0))
        )
        candidate["role"] = "primary"
        logger.info(
            "[Phase 1] No primary role — promoted '%s' (%s, conf=%.2f) to primary",
            candidate.get("label", "?"),
            candidate.get("cpc_family", "?"),
            candidate.get("confidence", 0),
        )

    return signals


def _ensure_classification_strategy(strategy, domain_signals: list) -> dict:
    """
    Ensure classification_strategy is a valid structured object.
    If missing, a plain string, or malformed, reconstruct from domain signals.
    """
    if isinstance(strategy, str):
        logger.warning(
            "[Phase 1] classification_strategy is a plain string '%s' — "
            "forcing structured reconstruction from domain signals",
            strategy,
        )
        strategy = None
    if isinstance(strategy, dict) and all(
        k in strategy for k in ("strategy", "primary_family", "anchor_split")
    ):
        return strategy

    if not domain_signals:
        return {
            "strategy": "single_domain",
            "primary_family": "G06F",
            "secondary_family": None,
            "anchor_split": [1.0, 0.0],
            "reconstructed": True,
            "reason": "No domain signals available",
        }

    sorted_signals = sorted(
        domain_signals,
        key=lambda x: x.get("confidence", 0),
        reverse=True,
    )
    positive_signals = [s for s in sorted_signals if s.get("role") != "negative"]
    if not positive_signals:
        positive_signals = sorted_signals[:1]

    primary = positive_signals[0] if positive_signals else sorted_signals[0]
    secondary = positive_signals[1] if len(positive_signals) > 1 else None

    if secondary and secondary.get("confidence", 0) >= 0.70:
        conf_gap = primary.get("confidence", 0) - secondary.get("confidence", 0)
        split = [0.55, 0.45] if conf_gap <= 0.15 else [0.65, 0.35]
        strategy_type = "hybrid"
    else:
        split = [1.0, 0.0]
        strategy_type = "single_domain"

    return {
        "strategy": strategy_type,
        "primary_family": primary.get("cpc_family", "G06F"),
        "secondary_family": secondary.get("cpc_family") if secondary else None,
        "anchor_split": split,
        "reconstructed": True,
        "reason": "LLM did not emit structured strategy block",
    }


_PROMPT_MARKERS = [
    "[TARGET]",
    "[FACTORY]",
    "[EVIDENCE]",
    "[LEGAL GATE]",
    "[SEARCH]",
    "[CHART]",
]


def _strip_prompt_markers_from_text(text: str) -> str:
    """Strip prompt instruction markers from raw LLM response text BEFORE parsing."""
    for marker in _PROMPT_MARKERS:
        text = text.replace(marker, "")
    text = re.sub(r"^\s*[:—\-]+\s*", "", text)
    return text


def _strip_prompt_markers_from_dict(obj):
    """Recursively strip prompt instruction markers from all string values in parsed dict."""
    if isinstance(obj, dict):
        return {k: _strip_prompt_markers_from_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_prompt_markers_from_dict(item) for item in obj]
    elif isinstance(obj, str):
        for marker in _PROMPT_MARKERS:
            if obj.startswith(marker):
                obj = obj[len(marker) :].strip()
                if obj.startswith(":") or obj.startswith("—") or obj.startswith("-"):
                    obj = obj[1:].strip()
        return obj
    return obj


def _validate_evidence_citations(evidence_table: list) -> None:
    """
    Validate that high-weight evidence terms have citations.
    Logs warnings for missing citations on weight >= 7 terms.
    """
    for row in evidence_table:
        if not isinstance(row, dict):
            continue
        weight = row.get("weight", 0)
        # Accept both 'source_quote' (new) and 'citation' (legacy)
        quote = row.get("source_quote") or row.get("citation", "")
        term = row.get("term", "")
        if weight >= 7 and not quote:
            logger.warning(
                "[Phase 1] Evidence term '%s' has weight %d but no source quote.",
                term,
                weight,
            )


# ── Generic structural terms that should never score above 4 ──
STRUCTURAL_NOISE_TERMS = {
    "acquisition module",
    "processing module",
    "storage module",
    "output module",
    "input module",
    "control module",
    "interface module",
    "communication module",
    "memory",
    "processor",
    "controller",
    "device",
    "system",
    "apparatus",
    "component",
    "network",
}


def deduplicate_evidence_table(rows: list[dict]) -> list[dict]:
    """
    Remove duplicate evidence rows by normalising the term field.
    Keep the row with the highest weight when duplicates exist.
    Enforces max 10 rows sorted by weight descending.
    """
    if not rows:
        return []

    seen = {}
    for row in rows:
        if not isinstance(row, dict) or "term" not in row:
            continue
            
        # Robust normalization: alphanumeric only, collapse spaces
        term = str(row["term"]).lower()
        term = re.sub(r"[^a-z0-9 ]", " ", term)
        key = " ".join(term.split())
        
        if not key:
            continue

        weight = row.get("weight", 0)
        if key not in seen:
            seen[key] = row
            seen[key]["weight"] = weight
        else:
            if weight > seen[key].get("weight", 0):
                seen[key] = row
                seen[key]["weight"] = weight

    deduped = list(seen.values())
    deduped.sort(key=lambda x: x.get("weight", 0), reverse=True)
    deduped = deduped[:10]

    if len(deduped) < 3:
        logger.warning(
            "[Phase 1] Evidence table has only %d unique rows after dedup. "
            "Prompt may need richer extraction instructions.",
            len(deduped),
        )

    return deduped


def validate_evidence_weights(rows: list[dict]) -> list[dict]:
    """
    Enforce global weight cap at 10.
    Cap structural noise terms at weight 4.
    Cap weight >= 7 terms at 6 if citation is missing.
    """
    for row in rows:
        if not isinstance(row, dict):
            continue
            
        # 1. Global cap at 10
        if row.get("weight", 0) > 10:
            logger.warning(
                "[Phase 1] Weight capped to 10 for term '%s' (%d -> 10)",
                row.get("term", "unknown"), row["weight"]
            )
            row["weight"] = 10

        term = row.get("term", "")
        term_lower = term.lower().strip()
        
        # 2. Structural noise cap
        if term_lower in STRUCTURAL_NOISE_TERMS:
            if row.get("weight", 0) > 4:
                logger.warning(
                    "[Phase 1] Structural term '%s' weight capped: %d → 4",
                    term,
                    row["weight"],
                )
                row["weight"] = 4
                
        # 3. Missing citation cap
        quote = row.get("source_quote") or row.get("citation", "")
        if row.get("weight", 0) >= 7 and not quote.strip():
            logger.warning(
                "[Phase 1] EVIDENCE WARNING: '%s' weight=%d but source quote empty. "
                "Capping weight to 6 until quote provided.",
                term,
                row["weight"],
            )
            row["weight"] = 6
            
    return rows


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
        completeness. Uses the new multi-pass prompt architecture from prompts.py.

        IMPORTANT: Passes A/B/C read ONLY description (not claims).
        Claims are first read in Phase 1.2 (forensic audit).
        The reconciliation pass receives claims for final verification only.
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

        claim_independent = truncated_claims.count("[INDEPENDENT]")
        claim_dependent = truncated_claims.count("[DEPENDENT")
        logger.info(
            "Phase 1 extract: desc=%d chars, claims=%d chars (%d indep + %d dep)",
            len(truncated_desc),
            len(truncated_claims),
            claim_independent,
            claim_dependent,
        )

        # ═══════════════════════════════════════════════════════════════
        # FIX: Pass description ONLY to Phase 1 prompts (not claims)
        # Claims will be analyzed for the FIRST time in Phase 1.2
        # ═══════════════════════════════════════════════════════════════
        prompts: Dict[str, Any] = phase1_prompt(truncated_desc, truncated_drawings)

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

        # Log which passes produced valid output and their key fields
        for pname, presult in pass_results.items():
            keys = list(presult.keys()) if isinstance(presult, dict) else []
            logger.info(
                "Phase 1 %s: %d keys — %s",
                pname,
                len(keys),
                ", ".join(keys[:6]) if keys else "EMPTY",
            )

        # Reconcile passes into final output
        # Claims provided here for final VERIFICATION only (not extraction)
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
            if response:
                response = _strip_prompt_markers_from_text(response)
            data = self._parse_json_response(response)
            if data:
                data = _strip_prompt_markers_from_dict(data)
        except Exception as e:
            logger.warning("Phase 1 reconciliation: LLM error: %s", e)
            data = None

        if data is None:
            # Fallback: merge available pass data instead of losing passes
            data = self._merge_pass_fallback(pass_results)
            data["_reconciliation_failed"] = True
            logger.warning(
                "Phase 1 reconciliation failed — merged fallback from %d passes: %s",
                len(pass_results),
                list(pass_results.keys()),
            )

        # Score completeness (uses claims for validation, not extraction)
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

        # ── Normalize domain signals: handle both "label" and "name" keys ──
        data["domain_signals"] = _normalize_domain_signals(
            data.get("domain_signals", [])
        )
        # ── Enforce correct role assignments (primary / secondary / negative) ──
        data["domain_signals"] = _normalize_domain_roles(data["domain_signals"])

        # ── Reconstruct classification_strategy if missing or malformed ──
        data["classification_strategy"] = _ensure_classification_strategy(
            data.get("classification_strategy"),
            data.get("domain_signals", []),
        )

        # ── Ensure core_function_generalized and cpc_terms are present ──
        if "core_function_generalized" not in data or not data["core_function_generalized"]:
            data["core_function_generalized"] = self._generate_fallback_generalized_function(data)
            
        if "cpc_terms" not in data or not data["cpc_terms"]:
            data["cpc_terms"] = self._generate_fallback_cpc_terms(data)

        # ── Deduplicate, validate weights, then validate citations ──
        evidence_table = data.get("evidence_table", [])
        if evidence_table:
            evidence_table = deduplicate_evidence_table(evidence_table)
            evidence_table = validate_evidence_weights(evidence_table)
            data["evidence_table"] = evidence_table
        _validate_evidence_citations(data.get("evidence_table", []))

        # Backward compatibility: keep legacy keys
        if "terms" not in data and "essential_terms" in data:
            data["terms"] = data["essential_terms"]

        # Extract claims if not already parsed
        if "claims" not in data:
            data["claims"] = self._extract_claims_from_labeled(truncated_claims)

        return data

    @staticmethod
    def _merge_pass_fallback(pass_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "primary_domain": {},
            "domain_signals": [],
            "class_hypotheses": [],
            "terms": [],
            "disambiguated_terms": [],
            "essential_features": [],
            "optional_features": [],
            "component_glossary": [],
            "hardware_software_boundary": {},
            "negative_signals": [],
            "negative_domains": [],
        }

        pass_a = pass_results.get("pass_a", {})
        pass_b = pass_results.get("pass_b", {})
        pass_c = pass_results.get("pass_c", {})
        pass_d = pass_results.get("pass_d") or {}

        # Core identity: Pass B (structure) + Pass A (function)
        if pass_b.get("technical_object"):
            merged["technical_object"] = pass_b["technical_object"]
        elif pass_a.get("primary_action"):
            merged["technical_object"] = pass_a["primary_action"]

        if pass_a.get("primary_action"):
            merged["core_function"] = pass_a["primary_action"]
            merged["secondary_actions"] = pass_a.get("secondary_actions", [])
            merged["core_problem"] = pass_a.get("core_problem", "")

        # Critical: Pass C (problem-solution — essential features, domain roots)
        if pass_c.get("objective_technical_problem"):
            merged["problem_solved"] = pass_c["objective_technical_problem"]
        if pass_c.get("solution_summary"):
            merged["solution_summary"] = pass_c["solution_summary"]
        if pass_c.get("essential_features"):
            merged["essential_features"] = pass_c["essential_features"]
        if pass_c.get("optional_features"):
            merged["optional_features"] = pass_c["optional_features"]
        if pass_c.get("technical_effect"):
            merged["technical_effect"] = pass_c["technical_effect"]
        if pass_c.get("closest_prior_art"):
            merged["closest_prior_art"] = pass_c["closest_prior_art"]
        domain_signals_c = pass_c.get("domain_signals_from_problem", [])
        if domain_signals_c:
            merged["domain_signals"].extend(domain_signals_c)

        # Pass D: component glossary, data flow, hardware/software boundary
        if pass_d.get("component_glossary"):
            merged["component_glossary"] = pass_d["component_glossary"]
        if pass_d.get("data_flow"):
            merged["data_flow"] = pass_d["data_flow"]
        if pass_d.get("hardware_software_boundary"):
            merged["hardware_software_boundary"] = pass_d["hardware_software_boundary"]

        # Merge terms from all passes (deduped by term name)
        seen_terms: set = set()
        for pass_data in (pass_a, pass_b, pass_c, pass_d):
            for source_key in (
                "functional_terms",
                "structural_terms",
                "drawing_terms",
                "terms",
            ):
                for t in pass_data.get(source_key, []):
                    if isinstance(t, dict) and t.get("term"):
                        name = t["term"].strip().lower()
                        if name and name not in seen_terms:
                            seen_terms.add(name)
                            merged["terms"].append(t)

        # Deduplicate domain_signals by name
        seen_ds: set = set()
        unique_signals = []
        for ds in merged["domain_signals"]:
            name = ds.get("name", ds.get("domain", ""))
            if name and name not in seen_ds:
                seen_ds.add(name)
                unique_signals.append(ds)
        merged["domain_signals"] = unique_signals

        merged["classification_strategy"] = "hybrid"
        merged["strategy_reasoning"] = "Fallback merge: reconciliation LLM failed"

        # ── Generate fallback core_function_generalized and cpc_terms ──
        merged["core_function_generalized"] = CPCExtractor._generate_fallback_generalized_function(merged)
        merged["cpc_terms"] = CPCExtractor._generate_fallback_cpc_terms(merged)

        return merged

    @staticmethod
    def _generate_fallback_generalized_function(data: Dict[str, Any]) -> List[str]:
        """Generate fallback core_function_generalized from core_function."""
        core_fn = data.get("core_function", "").strip()
        if not core_fn:
            return [
                "technical process using machine learning",
                "automated data processing and analysis",
                "computational method for signal processing",
            ]
        # Basic generalizations
        return [
            core_fn,
            f"{core_fn} using machine learning",
            f"automated {core_fn}",
        ]

    @staticmethod
    def _generate_fallback_cpc_terms(data: Dict[str, Any]) -> List[str]:
        """Generate fallback cpc_terms from evidence_table + disambiguated_terms."""
        cpc_term_set: set = set()
        
        # From evidence table
        for row in data.get("evidence_table", []):
            if isinstance(row, dict):
                term = row.get("term", "")
                if term:
                    cpc_term_set.add(term)
                    
        # From disambiguated terms
        for dt in data.get("disambiguated_terms", []):
            if isinstance(dt, dict):
                term = dt.get("term", "")
                if term:
                    cpc_term_set.add(term)
                for domain in dt.get("avoid", []):
                    if domain and len(domain) >= 4:
                        cpc_term_set.add(f"{domain[:4]} avoidance")
                        
        # From domain signals
        for ds in data.get("domain_signals", []):
            if isinstance(ds, dict):
                label = ds.get("label", ds.get("name", ""))
                if label:
                    cpc_term_set.add(label.lower())

        # If empty, add generics
        if not cpc_term_set:
            cpc_term_set.update(["machine learning", "data processing", "pattern recognition"])
            
        return sorted(list(cpc_term_set))[:10]

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
