"""
phase1_2_runner.py — Phase 1.2 Mandatory Claim-to-Domain Forensic Audit.

Phase 1.2 is the SYSTEM OF RECORD for domain validation. Phase 2 MUST NOT start
until Phase 1.2 has produced a final_primary_anchor. This phase uses the full
claims text to provide legally-defensible justification for every domain pursued.

Key functions:
  run_phase1_2()          Main entry point — runs the audit LLM call
  apply_phase1_2_result()  Applies audit findings back to the Phase 1 data
"""

import logging
from difflib import SequenceMatcher

from ..prompts.prompt_phase1_2 import phase1_2_audit_prompt
from ..utils.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


def _fuzzy_match(term: str, claims_text: str, threshold: float = 0.72) -> tuple:
    """
    Returns (matched: bool, matched_phrase: str).
    Checks if term or any semantic variant appears in claims_text.
    threshold=0.72 allows for plurals, hyphenation variants,
    and minor rephrasing while blocking false positives.
    """
    term_lower = term.lower()
    claims_lower = claims_text.lower()

    # Pass 1: direct substring
    if term_lower in claims_lower:
        return True, term

    # Pass 2: sliding window fuzzy match over claim sentences
    sentences = claims_lower.split(".")
    for sentence in sentences:
        ratio = SequenceMatcher(None, term_lower, sentence).ratio()
        if ratio >= threshold:
            return True, sentence.strip()

        # Pass 3: token overlap (handles "phoneme-derived centers" vs
        #          "centers derived from phonemes")
        term_tokens = set(term_lower.replace("-", " ").split())
        sentence_tokens = set(sentence.split())
        if not term_tokens:
            continue
        overlap = len(term_tokens & sentence_tokens) / len(term_tokens)
        if overlap >= 0.75:
            return True, sentence.strip()

    return False, ""


def run_phase1_2(
    classifier,
    phase1: dict,
    full_labeled_claims: str,
) -> dict:
    """
    Mandatory Phase 1.2 audit: validate Phase 1 domain signals against claims.

    Returns the forensic audit result. Use apply_phase1_2_result() to
    integrate the findings back into the Phase 1 data before Phase 2.
    """
    if not phase1 or not full_labeled_claims:
        logger.warning("Phase 1.2 skipped: missing phase1 data or claims text")
        return {
            "audit_status": "SKIPPED",
            "reason": "Missing Phase 1 data or claims text",
            "final_primary_anchor": phase1.get("primary_domain", {}).get(
                "cpc_class", ""
            ),
            "secondary_anchors": [],
            "rejected_domains": [],
            "hybrid_detected": False,
        }

    prompt = phase1_2_audit_prompt(phase1, full_labeled_claims)

    try:
        logger.info(
            "Phase 1.2: running forensic audit on %d domain signals",
            len(phase1.get("domain_signals", [])),
        )
        response = classifier.llm.chat(
            system_prompt=prompt,
            user_message="Produce ONLY the JSON output as specified.",
            temperature=0.1,
            max_tokens=4000,
        )
        audit = parse_llm_json(response)

        if not audit:
            logger.warning("Phase 1.2: LLM returned invalid JSON — audit skipped")
            return {
                "audit_status": "ERROR",
                "reason": "LLM response could not be parsed as JSON",
                "final_primary_anchor": phase1.get("primary_domain", {}).get(
                    "cpc_class", ""
                ),
                "secondary_anchors": [],
                "rejected_domains": [],
                "hybrid_detected": False,
                "raw_response": response[:500] if response else "",
            }

        rejected = audit.get("rejected_domains", [])
        if rejected:
            logger.warning(
                "Phase 1.2: REJECTED %d hallucinated domain(s): %s",
                len(rejected),
                [r.get("domain", "?") for r in rejected],
            )
        else:
            logger.info("Phase 1.2: no hallucinated domains detected")

        anchor = audit.get("final_primary_anchor", "")
        if anchor:
            logger.info(
                "Phase 1.2: final primary anchor = %s (confidence: %.2f)",
                anchor,
                audit.get("audit_confidence", 0.0),
            )

        # Generate signal_validations from term_claim_mapping for the UI
        audit["signal_validations"] = _build_signal_validations(audit)

        return audit

    except Exception as e:
        logger.error("Phase 1.2 LLM call failed: %s — audit skipped", str(e))
        return {
            "audit_status": "ERROR",
            "reason": f"LLM call failed: {str(e)}",
            "final_primary_anchor": phase1.get("primary_domain", {}).get(
                "cpc_class", ""
            ),
            "secondary_anchors": [],
            "rejected_domains": [],
            "hybrid_detected": False,
        }


def apply_phase1_2_result(phase1: dict, audit: dict, claims_text: str = "") -> None:
    """
    Apply Phase 1.2 audit findings back to the Phase 1 data.
    Modifies phase1 in-place.

    If the audit found a different primary anchor, overrides phase1["primary_domain"].
    If the audit rejected domains, removes them from domain_signals and adds to
    negative signals.

    Uses fuzzy matching to validate domain signals against claims text.
    """
    if not audit or audit.get("audit_status") in ("SKIPPED", "ERROR"):
        return

    phase1["phase1_2_audit"] = audit

    # ── Fuzzy-match validation of domain signals against claims ──
    if claims_text:
        _validate_signals_with_fuzzy_match(phase1, claims_text)

    # Override primary domain with audit result if different
    anchor = audit.get("final_primary_anchor", "")
    current_anchor = phase1.get("primary_domain", {}).get("cpc_class", "")
    if anchor and anchor != current_anchor:
        logger.info(
            "Phase 1.2: overriding primary_domain %s → %s (audit correction)",
            current_anchor,
            anchor,
        )
        phase1["primary_domain"] = {
            "name": anchor,
            "cpc_class": anchor,
            "confidence": audit.get("audit_confidence", 0.85),
            "supporting_passes": ["1.2_forensic_audit"],
            "reasoning": audit.get("legal_justification", "Phase 1.2 audit override"),
        }

    # Move rejected domains from domain_signals to negative domains
    rejected = audit.get("rejected_domains", [])
    if rejected:
        rejected_families = {r.get("cpc_family", r.get("domain", "")) for r in rejected}
        rejected_families.discard("")

        domain_signals = phase1.get("domain_signals", [])
        kept_signals = []
        negative_additions = []

        for ds in domain_signals:
            ds_family = ds.get("cpc_family", "")
            if ds_family in rejected_families:
                negative_additions.append(
                    {
                        "domain": ds.get("name", ds_family),
                        "cpc_family": ds_family,
                        "confidence": 0.95,
                        "penalize_family": ds_family,
                        "rejection_source": "phase1_2_audit",
                    }
                )
                logger.info("Phase 1.2: moved %s to negative domains", ds_family)
            else:
                kept_signals.append(ds)

        if negative_additions:
            phase1["domain_signals"] = kept_signals
            existing_negatives = phase1.get("negative_domains", [])
            phase1["negative_domains"] = existing_negatives + negative_additions

    # Set hybrid flag if detected — preserve the structured strategy dict
    if audit.get("hybrid_detected"):
        existing_strategy = phase1.get("classification_strategy")
        if isinstance(existing_strategy, dict):
            existing_strategy["strategy"] = audit.get("hybrid_type", "hybrid")
        else:
            existing_strategy = {
                "strategy": audit.get("hybrid_type", "hybrid"),
                "primary_family": audit.get("final_primary_anchor", ""),
                "secondary_family": audit.get("secondary_anchors", [None])[0],
                "anchor_split": [0.60, 0.40],
                "reconstructed": True,
                "reason": "Phase 1.2 hybrid detection",
            }
        phase1["classification_strategy"] = existing_strategy
        bridge = audit.get("bridge_clause_found")
        bridge_str = str(bridge) if bridge is not None else ""
        phase1["strategy_reasoning"] = (
            "Phase 1.2 audit detected functionally integrated domains. " + bridge_str
        )

    # Add status note
    existing_note = phase1.get("phase1_status_note", "")
    if rejected:
        phase1["phase1_status_note"] = (
            "Phase 1.2 Forensic Audit: hallucinated domains detected and removed. "
            + existing_note
        )


def _build_signal_validations(audit: dict) -> list:
    """
    Convert term_claim_mapping into the signal_validations format expected by the UI.
    Derives status (validated / rejected / downgraded) from anchor_type and keyword_density.
    """
    mapping = audit.get("term_claim_mapping", [])
    rejected_families = {
        r.get("cpc_family", r.get("domain", ""))
        for r in audit.get("rejected_domains", [])
    }
    primary = audit.get("final_primary_anchor", "")
    validations = []

    for entry in mapping:
        if not isinstance(entry, dict):
            continue
        cpc = entry.get("cpc_family", "")
        domain = entry.get("domain", cpc)
        anchor = entry.get("anchor_type", "")
        density = entry.get("keyword_density", 0)
        claims = entry.get("supporting_claims", [])

        # Build short evidence string from first supporting claim
        evidence = ""
        if claims:
            first = claims[0]
            terms = first.get("evidence_terms", [])
            evidence = f"Claim {first.get('claim_number', '?')}: {', '.join(terms[:4])}"

        if cpc in rejected_families or anchor == "HALLUCINATION":
            status = "rejected"
            reason = f"Zero or insufficient claim support (density={density})"
        elif cpc == primary or anchor == "PRIMARY_ANCHOR":
            status = "validated"
            reason = f"Primary anchor — {density} claim keyword match(es)"
        elif anchor == "SECONDARY_ANCHOR":
            status = "validated"
            reason = f"Secondary anchor — {density} claim keyword match(es)"
        else:
            status = "downgraded"
            reason = f"Weak claim support (density={density})"

        validations.append(
            {
                "domain": domain,
                "cpc_family": cpc,
                "status": status,
                "reason": reason,
                "claims_evidence": evidence,
            }
        )

    return validations


def _validate_signals_with_fuzzy_match(phase1: dict, claims_text: str) -> None:
    """
    Fuzzy-match each domain signal's label/terms against the claims text.
    Adds a 'fuzzy_validated' flag to each domain signal.
    Demotes signals with no fuzzy match to confidence 0.3.
    """
    domain_signals = phase1.get("domain_signals", [])
    if not domain_signals or not claims_text:
        return

    for signal in domain_signals:
        if not isinstance(signal, dict):
            continue
        label = signal.get("label", signal.get("name", ""))
        cpc = signal.get("cpc_family", "")
        matched, phrase = _fuzzy_match(label, claims_text)
        if not matched and cpc:
            matched, phrase = _fuzzy_match(cpc, claims_text)
        signal["fuzzy_validated"] = matched
        if not matched and signal.get("role") not in ("negative",):
            old_conf = signal.get("confidence", 0)
            signal["confidence"] = min(old_conf, 0.35)
            logger.warning(
                "[Phase 1.2] Fuzzy match FAILED for '%s' (%s) — confidence capped at 0.35",
                label,
                cpc,
            )
