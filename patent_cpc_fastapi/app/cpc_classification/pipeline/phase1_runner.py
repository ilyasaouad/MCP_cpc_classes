import logging
import re
from typing import Dict, Any, List

from ..extracting_cpc import _ensure_classification_strategy
from ..prompts.prompt_phase1 import (
    score_phase1_completeness,
    phase1_1_emergency_anchor_prompt,
)
from ..utils.json_utils import parse_phase1_1_anchor

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

# ═══════════════════════════════════════════════════════════════════════════
# FIX: Removed generic/hallucination-prone keyword mappings
# "frame", "classification", "model", "signal" were mapping to wrong domains
# ═══════════════════════════════════════════════════════════════════════════

_KEYWORD_TO_CPC: Dict[str, str] = {
    # G10L — Speech / Audio
    "speech": "G10L",
    "audio": "G10L",
    "acoustic": "G10L",
    "phoneme": "G10L",
    "voice": "G10L",
    "utterance": "G10L",
    "speaker": "G10L",
    "mfcc": "G10L",
    "spectral": "G10L",
    "formant": "G10L",
    "prosody": "G10L",
    "pitch": "G10L",
    "speech frame": "G10L",
    "speech segmentation": "G10L",
    # G06N — AI / ML
    "neural network": "G06N",
    "training": "G06N",
    "inference": "G06N",
    "weights": "G06N",
    "layers": "G06N",
    "gradient": "G06N",
    "backpropagation": "G06N",
    "deep learning": "G06N",
    "activation": "G06N",
    "self-supervised": "G06N",
    "clustering": "G06N",
    "pre-training": "G06N",
    "pretraining": "G06N",
    "encoder": "G06N",
    "decoder": "G06N",
    "transformer": "G06N",
    "lstm": "G06N",
    "attention": "G06N",
    "embedding": "G06N",
    "classifier": "G06N",
    "loss": "G06N",
    "machine learning": "G06N",
    "neural network model": "G06N",
    # G06F — Computing
    "processor": "G06F",
    "memory": "G06F",
    "database": "G06F",
    "file": "G06F",
    "program": "G06F",
    "interface": "G06F",
    "computing": "G06F",
    "storage": "G06F",
    "cache": "G06F",
    "cpu": "G06F",
    "gpu": "G06F",
    # G06T / G06V — Image / Video (SPECIFIC mappings only)
    "image": "G06T",
    "pixel": "G06T",
    "camera": "G06T",
    "visual": "G06T",
    "picture": "G06T",
    "photograph": "G06T",
    "video": "G06V",
    "image frame": "G06V",
    "video frame": "G06V",
    "object detection": "G06V",
    "image classification": "G06V",
    "image recognition": "G06V",
    "object recognition": "G06V",
    # H04L — Networking
    "network": "H04L",
    "packet": "H04L",
    "transmission": "H04L",
    "protocol": "H04L",
    "routing": "H04L",
    "node": "H04L",
    # H04W — Wireless
    "wireless": "H04W",
    "radio": "H04W",
    "antenna": "H04W",
    "base station": "H04W",
    "wireless signal": "H04W",
    "radio signal": "H04W",
    "wireless channel": "H04W",
    # A61B — Medical
    "patient": "A61B",
    "diagnosis": "A61B",
    "medical": "A61B",
    "surgical": "A61B",
    "anatomical": "A61B",
    "clinical": "A61B",
    # B60W — Vehicle control
    "vehicle": "B60W",
    "driving": "B60W",
    "steering": "B60W",
    "brake": "B60W",
    "acceleration": "B60W",
    "cruise": "B60W",
    # B60L — Electric propulsion
    "electric vehicle": "B60L",
    "battery": "B60L",
    "electric motor": "B60L",
    "propulsion": "B60L",
    "charging": "B60L",
    # G05B — Control systems
    "control system": "G05B",
    "automation": "G05B",
    "actuator": "G05B",
    "feedback control": "G05B",
    # F16J/F16K — Mechanical
    "seal": "F16J",
    "gasket": "F16J",
    "piston": "F16J",
    "valve": "F16K",
    "fluid flow": "F16K",
    "fluid": "F16K",
    # E21B — Drilling
    "drilling": "E21B",
    "borehole": "E21B",
    "well": "E21B",
    "formation": "E21B",
    # B01D — Separation
    "separation": "B01D",
    "filtration": "B01D",
    "membrane": "B01D",
    "purification": "B01D",
    # G01N — Analysis
    "sample analysis": "G01N",
    "measurement": "G01N",
    "detection": "G01N",
}

# ── Domain families that are commonly co-occurring (not conflicts) ──
_RELATED_FAMILIES: Dict[str, set] = {
    "G10L": {"G06N", "G06F", "H04L"},
    "G06N": {"G10L", "G06F", "G06T", "G06V", "G06K", "G05B", "H04L", "A61B"},
    "G06T": {"G06V", "G06N", "G06F"},
    "G06V": {"G06T", "G06N", "G06F"},
    "G06F": {"G06N", "G10L", "G06T", "G06V", "G06K", "H04L", "H04W", "G05B", "A61B"},
    "H04L": {"H04W", "G06F", "G06N", "G10L"},
    "H04W": {"H04L", "G06F"},
    "A61B": {"G06N", "G06F", "G06T", "G06V"},
    "B60W": {"B60L", "G05B", "G06F", "G06N"},
    "B60L": {"B60W", "G06F"},
    "G05B": {"G06F", "G06N", "B60W", "F16K"},
    "E21B": {"G01N", "B01D", "F16J", "F16K"},
    "F16J": {"F16K", "E21B"},
    "F16K": {"F16J", "E21B", "G05B"},
    "B01D": {"G01N", "E21B"},
    "G01N": {"B01D", "E21B", "A61B"},
}


def _detect_domain_conflict(phase1: dict) -> dict | None:
    """Detect domain conflicts between high-confidence signals.
    
    FIX: Now collects ALL conflicts, not just the first one.
    Returns first conflict for resolution, but logs all.
    """
    domain_signals = phase1.get("domain_signals", [])
    if not domain_signals:
        return None

    positives = [
        ds
        for ds in domain_signals
        if ds.get("role") in ("primary", "secondary")
        and ds.get("confidence", 0) >= 0.70
    ]
    if len(positives) < 2:
        return None

    families = []
    for ds in positives:
        cpc = ds.get("cpc_family", "")
        if len(cpc) >= 4:
            families.append(cpc[:4])

    if len(families) < 2:
        return None

    unique_families = set(families)
    conflicts = []
    
    for fam_a in unique_families:
        for fam_b in unique_families:
            if fam_a >= fam_b:
                continue
            related = _RELATED_FAMILIES.get(fam_a, set())
            if fam_b not in related:
                conf_a = next(
                    ds["confidence"]
                    for ds in positives
                    if (ds.get("cpc_family", "")[:4] == fam_a)
                )
                conf_b = next(
                    ds["confidence"]
                    for ds in positives
                    if (ds.get("cpc_family", "")[:4] == fam_b)
                )
                logger.warning(
                    "Phase 1 domain conflict: %s (%.2f) vs %s (%.2f) — "
                    "unrelated domains. Component audit will resolve.",
                    fam_a,
                    conf_a,
                    fam_b,
                    conf_b,
                )
                conflicts.append({
                    "conflict_a": fam_a,
                    "conflict_b": fam_b,
                    "confidence_a": conf_a,
                    "confidence_b": conf_b,
                })

    if not conflicts:
        return None

    # Return first conflict for resolution, but include all for logging
    return {
        "conflict_a": conflicts[0]["conflict_a"],
        "conflict_b": conflicts[0]["conflict_b"],
        "signals": positives,
        "all_conflicts": conflicts,
    }


def _component_audit(phase1: dict) -> Dict[str, Any]:
    # FIX: Simplified term extraction (removed dead code)
    terms = phase1.get("terms", []) or phase1.get("essential_terms", [])
    
    if not terms:
        logger.warning("Component audit: no terms available")
        return {
            "anchor": None,
            "votes": {},
            "total_terms_checked": 0,
            "note": "No terms available for component audit",
        }

    votes: Dict[str, int] = {}
    total_high = 0

    sorted_terms = sorted(
        terms,
        key=lambda t: t.get("importance", 0) if isinstance(t, dict) else 0,
        reverse=True,
    )

    for t in sorted_terms[:15]:
        if not isinstance(t, dict):
            continue
        term_text = t.get("term", "").strip().lower()
        importance = t.get("importance", 5)
        if not term_text or len(term_text) < 3:
            continue

        weight = 2 if importance >= 9 else 1
        total_high += 1

        for keyword, cpc_family in _KEYWORD_TO_CPC.items():
            if keyword in term_text:
                votes[cpc_family] = votes.get(cpc_family, 0) + weight
                break

    if not votes:
        logger.warning("Component audit: no terms matched any CPC family keywords")
        return {
            "anchor": None,
            "votes": {},
            "total_terms_checked": total_high,
            "note": "No domain keywords found in high-importance terms",
        }

    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_votes[0][0]
    winner_count = sorted_votes[0][1]
    total_votes = sum(votes.values())

    confidence = min(round(winner_count / max(total_votes, 1), 2), 0.95)
    if len(sorted_votes) >= 2:
        margin = winner_count - sorted_votes[1][1]
        if margin >= 3:
            confidence = min(0.95, confidence + 0.1)

    logger.info(
        "Component audit: %d terms checked, votes=%s → anchor=%s (%.2f)",
        total_high,
        dict(votes),
        winner,
        confidence,
    )

    return {
        "anchor": winner,
        "confidence": confidence,
        "votes": dict(votes),
        "total_terms_checked": total_high,
        "note": f"Component voting selected {winner} based on {winner_count} keyword matches",
    }


def _ensure_system_context(phase1: dict) -> None:
    """
    Ensure system_context is never empty.
    If missing or empty, trigger recovery inference.
    """
    existing = phase1.get("system_context", "").strip()
    if existing and len(existing) > 5:
        return  # Already has meaningful system context

    recovery = _recover_system_context(phase1)
    phase1["system_context"] = recovery["value"]
    phase1["system_context_meta"] = {
        "type": recovery["type"],
        "components": recovery.get("components", []),
    }
    logger.info(
        "Phase 1 system context ensured: '%s' (type=%s)",
        recovery["value"][:80],
        recovery["type"],
    )


def _recover_system_context(phase1: dict) -> dict:
    """
    Mandatory system context recovery.

    If system_context is missing or empty, infer it from:
    - core_function (method steps)
    - technical_objects (what the invention IS)
    - domain_signals (industry/domain)
    - components/terms (training pipeline, device references)

    Returns a structured dict with 'type' and 'components'.
    """
    existing = phase1.get("system_context", "")
    if existing and len(existing.strip()) > 5:
        # Already has meaningful system context
        return {"type": "llm_extracted", "value": existing}

    core_function = phase1.get("core_function", "").strip()
    technical_object = phase1.get("technical_object", "").strip()
    domain_signals = phase1.get("domain_signals", [])
    terms = phase1.get("terms", phase1.get("essential_terms", []))

    # Step 1: Build component list from component_glossary (preferred — short module names)
    # or terms, but only keep short entries (<= 40 chars) to avoid claim-language bleed.
    components = []
    glossary = phase1.get("component_glossary", [])
    if glossary:
        for g in glossary[:8]:
            if isinstance(g, dict):
                name = g.get("name", "").strip()
                if name and len(name) <= 40:
                    components.append(name)
    if not components and isinstance(terms, list):
        for t in terms[:10]:
            if isinstance(t, dict):
                term = t.get("term", "").strip()
                importance = t.get("importance", 0)
                # Only include short noun phrases — skip verbose method steps
                if term and 3 < len(term) <= 40 and importance >= 6:
                    components.append(term)

    # Step 2: Infer system type from domain signals + core function
    domain_keywords = []
    for ds in domain_signals:
        if isinstance(ds, dict):
            name = ds.get("name", "")
            role = ds.get("role", "")
            conf = ds.get("confidence", 0)
            if role in ("primary", "secondary") and conf >= 0.5:
                domain_keywords.append(name)

    # Step 3: Derive system type
    inferred_type = _derive_system_type(
        core_function, technical_object, domain_keywords
    )

    # Step 4: Build system context string — use components only when they are short module
    # names, not when the list reads like claim language.
    if components:
        system_str = f"{inferred_type} for {core_function}" if core_function else inferred_type
    elif core_function:
        system_str = f"{inferred_type} for {core_function}"
    else:
        system_str = inferred_type

    logger.info(
        "Phase 1 system context recovery: inferred '%s' from function=%s, domains=%s",
        inferred_type,
        core_function[:50] if core_function else "(empty)",
        domain_keywords,
    )

    return {"type": "inferred", "value": system_str, "components": components[:8]}


def _derive_system_type(
    core_function: str, technical_object: str, domain_keywords: list
) -> str:
    """
    Derive system/application type from available signals.
    """
    text = f"{core_function} {technical_object}".lower()

    # Speech/Audio systems
    if any(k in text for k in ["speech", "audio", "voice", "acoustic", "phoneme"]):
        if any(k in text for k in ["train", "pretrain", "model", "learn"]):
            return "speech pretraining system"
        if any(k in text for k in ["recogn", "transcri", "convert"]):
            return "speech recognition system"
        return "speech processing system"

    # Neural network / ML systems
    if any(k in text for k in ["neural network", "deep learning", "transformer"]):
        if any(k in text for k in ["train", "pretrain", "fine-tune"]):
            return "neural network training system"
        return "machine learning system"

    # Image/Video systems
    if any(k in text for k in ["image", "video", "visual", "camera"]):
        return "image processing system"

    # Network/Communication systems
    if any(k in text for k in ["network", "protocol", "transmission", "packet"]):
        return "data communication system"

    # Control/Automation systems
    if any(k in text for k in ["control", "sensor", "actuator", "feedback"]):
        return "control system"

    # Vehicle systems
    if any(k in text for k in ["vehicle", "driving", "autonomous"]):
        return "vehicle control system"

    # Medical systems
    if any(k in text for k in ["medical", "diagnosis", "patient", "clinical"]):
        return "medical diagnostic system"

    # Fallback: use domain keywords
    if domain_keywords:
        return f"{domain_keywords[0]} system"

    # Ultimate fallback: derive from core function
    if core_function:
        words = core_function.split()[:4]
        return f"{' '.join(words)} system"

    return "technical system"


def _resolve_any_domain_conflict(phase1: dict) -> None:
    conflict = _detect_domain_conflict(phase1)
    if not conflict:
        return

    audit = _component_audit(phase1)
    if audit["anchor"]:
        phase1["primary_domain"] = {
            "name": audit["anchor"],
            "cpc_class": audit["anchor"],
            "confidence": audit["confidence"],
            "supporting_passes": ["1.1_component_audit"],
            "reasoning": audit["note"],
        }
        phase1["phase1_status_note"] = (
            "Phase 1.1 Conflict Resolution: domain hallucination detected. "
            "Component audit overrode primary domain."
        )
        phase1["phase1_emergency_anchor"] = audit
        phase1["phase1_conflict_resolution"] = {
            "conflict": conflict,
            "audit": audit,
        }
        logger.warning(
            "Phase 1.1 conflict resolution: component audit overrode primary domain → %s",
            audit["anchor"],
        )


def _enforce_disambiguation(phase1: dict) -> None:
    """
    Post-processing step: enforce disambiguated_term avoid tags on domain_signals.

    If the Phase 1 extraction explicitly says "avoid G06T" in disambiguated_terms,
    then any domain_signal for G06T must have its confidence set to 0.0 and be
    moved to negative_domains. This ensures the disambiguation decisions are
    respected before Phase 1.2 audit and Phase 2 search.
    """
    disambiguated = phase1.get("disambiguated_terms", [])
    if not disambiguated:
        return

    avoid_families: set = set()
    for dt in disambiguated:
        if isinstance(dt, dict):
            for domain in dt.get("avoid", []):
                if domain and len(domain) >= 4:
                    avoid_families.add(domain[:4])

    if not avoid_families:
        return

    # Never zero out methodology domains — they are legitimate secondary anchors
    _PROTECTED_DOMAINS = {"G06N", "G06F", "H04L"}
    avoided_for_enforcement = avoid_families - _PROTECTED_DOMAINS
    if avoid_families != avoided_for_enforcement:
        stripped = avoid_families - avoided_for_enforcement
        logger.info(
            "Disambiguation: G06N/G06F/H04L in avoid lists are protected — not zeroed: %s",
            stripped,
        )

    if not avoided_for_enforcement:
        return

    domain_signals = phase1.get("domain_signals", [])
    updated_signals = []
    moved_to_negative = []

    for ds in domain_signals:
        ds_family = ds.get("cpc_family", "")
        if not ds_family:
            updated_signals.append(ds)
            continue

        if ds_family in avoided_for_enforcement:
            logger.warning(
                "Disambiguation enforcement: %s avoided by disambiguated_terms → "
                "confidence zeroed, moved to negative_domains",
                ds_family,
            )
            moved_to_negative.append(
                {
                    "domain": ds.get("name", ds_family),
                    "cpc_family": ds_family,
                    "confidence": 0.95,
                    "penalize_family": ds_family,
                    "rejection_source": "disambiguated_terms_enforcement",
                }
            )
            # Zero out confidence on the original signal so Phase 1.2 sees it as rejected
            ds["confidence"] = 0.0
            ds["role"] = "negative"
            ds["_disambiguation_overridden"] = True
            updated_signals.append(ds)
        else:
            updated_signals.append(ds)

    if moved_to_negative:
        phase1["domain_signals"] = updated_signals
        existing_negatives = phase1.get("negative_domains", [])
        existing_families = {n.get("cpc_family", "") for n in existing_negatives}
        new_negatives = [
            n for n in moved_to_negative if n["cpc_family"] not in existing_families
        ]
        phase1["negative_domains"] = existing_negatives + new_negatives
        phase1["_disambiguation_enforced"] = True
        logger.info(
            "Disambiguation enforcement: %d domain(s) zeroed and moved to negative: %s",
            len(moved_to_negative),
            [n["cpc_family"] for n in moved_to_negative],
        )


def run_phase1(classifier, description, labeled_claims):
    best_phase1 = None
    best_score = -1

    claim_independent_count = (
        labeled_claims.count("[INDEPENDENT]") if labeled_claims else 0
    )
    claim_dependent_count = labeled_claims.count("[DEPENDENT") if labeled_claims else 0
    logger.info(
        "Phase 1 starting: desc=%d chars, claims=%d indep + %d dep (%d chars total)",
        len(description),
        claim_independent_count,
        claim_dependent_count,
        len(labeled_claims),
    )
    if not labeled_claims or claim_independent_count == 0:
        logger.warning(
            "Phase 1: NO labeled claims detected — terms will lack claims source weighting"
        )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            phase1 = classifier.extractor.extract(description, labeled_claims)

            if not isinstance(phase1, dict) or not phase1.get("technical_object"):
                logger.warning(
                    "Phase 1 attempt %d: no technical_object, retrying", attempt
                )
                continue

            # ── Force classification_strategy reconstruction ──
            strategy = phase1.get("classification_strategy")
            if strategy is None or isinstance(strategy, str):
                strategy = _ensure_classification_strategy(
                    strategy, phase1.get("domain_signals", [])
                )
                phase1["classification_strategy"] = strategy
                logger.info(
                    "[Phase 1] classification_strategy reconstructed from signals: %s",
                    strategy,
                )
            assert "anchor_split" in strategy and len(strategy["anchor_split"]) == 2, (
                "[Phase 1] FATAL: classification_strategy missing anchor_split"
            )
            assert abs(sum(strategy["anchor_split"]) - 1.0) < 0.01, (
                "[Phase 1] FATAL: anchor_split must sum to 1.0"
            )

            completeness = score_phase1_completeness(phase1, labeled_claims)
            phase1["phase1_completeness"] = completeness
            
            # ═══════════════════════════════════════════════════════════════
            # FIX: Add flat status/score keys for debug output
            # ═══════════════════════════════════════════════════════════════
            phase1["phase1_status"] = completeness["status"]  # "PASS" / "WARN" / "FAIL"
            phase1["phase1_score"] = completeness["score"]    # 0-100
            
            status = completeness["status"]
            score = completeness["score"]
            issues = "; ".join(completeness["issues"])

            if score > best_score:
                best_score = score
                best_phase1 = phase1

            if status == "PASS":
                logger.info("Phase 1 attempt %d: PASS (%d/100)", attempt, score)
                _resolve_any_domain_conflict(phase1)
                _enforce_disambiguation(phase1)
                _ensure_system_context(phase1)
                return phase1
            elif status == "WARN":
                logger.info(
                    "Phase 1 attempt %d: WARN (%d/100): %s", attempt, score, issues
                )
                _resolve_any_domain_conflict(phase1)
                _enforce_disambiguation(phase1)
                _ensure_system_context(phase1)
                return phase1
            else:
                logger.warning(
                    "Phase 1 attempt %d: FAIL (%d/100): %s — retrying",
                    attempt,
                    score,
                    issues,
                )

        except Exception as e:
            logger.error("Phase 1 attempt %d failed: %s", attempt, str(e))
            continue

    if best_phase1 is not None:
        logger.warning(
            "Phase 1 best score after %d attempts: %d/100", _MAX_RETRIES, best_score
        )
    else:
        best_phase1 = {}

    # ── Phase 1.1: Emergency Anchor (LLM-based recovery) ──
    phase1_1_triggered = True
    phase1_1_reason = (
        f"Phase 1 failed after {_MAX_RETRIES} attempts (best score: {best_score}/100)"
    )
    logger.info("[Phase 1.1: TRIGGERED — reason: %s]", phase1_1_reason)

    anchor_result = _run_phase1_1_emergency_anchor(
        classifier, best_phase1, labeled_claims
    )

    # ── Component audit as backup for LLM anchor ──
    audit = _component_audit(best_phase1)
    if audit["anchor"] and anchor_result.get("emergency_anchor") == "G06F":
        logger.warning(
            "Phase 1.1 LLM returned generic G06F — component audit overrides with %s",
            audit["anchor"],
        )
        anchor_result = {
            "emergency_anchor": audit["anchor"],
            "confidence": audit["confidence"],
            "reasoning": audit["note"],
            "anchor_basis": "component_audit",
        }

    best_phase1["primary_domain"] = {
        "name": anchor_result["emergency_anchor"],
        "cpc_class": anchor_result["emergency_anchor"],
        "confidence": anchor_result["confidence"],
        "supporting_passes": ["1.1_emergency"],
        "reasoning": anchor_result["reasoning"],
    }
    best_phase1["phase1_status_note"] = (
        "Phase 1.1 Auto-Recovery triggered. Confidence may be lower due to domain ambiguity."
    )
    best_phase1["phase1_emergency_anchor"] = anchor_result
    best_phase1["phase1_1_status"] = {
        "triggered": phase1_1_triggered,
        "reason": phase1_1_reason,
        "summary_quality_score": best_score / 100.0,
    }

    _enforce_disambiguation(best_phase1)
    _ensure_system_context(best_phase1)

    return best_phase1


def _run_phase1_1_emergency_anchor(
    classifier, phase1: dict, labeled_claims: str
) -> dict:
    solution_summary = phase1.get("solution_summary", "")
    problem_solved = phase1.get("problem_solved", "")
    technical_object = phase1.get("technical_object", "")
    core_function = phase1.get("core_function", "")
    independent_claim_1 = _extract_first_independent_claim(labeled_claims)

    prompt = phase1_1_emergency_anchor_prompt(
        solution_summary,
        problem_solved,
        technical_object,
        core_function,
        independent_claim_1,
    )

    try:
        response = classifier.llm.chat(
            system_prompt=prompt,
            user_message="Produce ONLY the JSON output as specified.",
            temperature=0.1,
            max_tokens=2000,
        )
        anchor = parse_phase1_1_anchor(response)
        logger.info(
            "Phase 1.1 emergency anchor: %s (confidence: %.2f, basis: %s)",
            anchor["emergency_anchor"],
            anchor["confidence"],
            anchor.get("anchor_basis", "unknown"),
        )
    except Exception as e:
        logger.error(
            "Phase 1.1 LLM call failed: %s — using hard-coded fallback", str(e)
        )
        anchor = parse_phase1_1_anchor(None)

    return anchor


def _extract_first_independent_claim(labeled_claims: str) -> str:
    if not labeled_claims:
        return ""
    lines = labeled_claims.splitlines()
    collecting = False
    claim_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[INDEPENDENT]":
            if collecting:
                break
            collecting = True
            continue
        if stripped.startswith("[DEPENDENT"):
            continue
        if collecting and stripped:
            claim_lines.append(stripped)
    return " ".join(claim_lines)


_ISSUE_TO_WRITING_DIAGNOSIS: Dict[str, str] = {
    "technical_object is missing or too vague": (
        "The patent does not clearly define what the invention is. "
        "The technical object should be stated explicitly in the description or abstract."
    ),
    "No confident domain signals found": (
        "The patent lacks clear technical domain vocabulary. "
        "The description may be too abstract or use non-standard terminology."
    ),
    "Primary domain confidence too low or missing": (
        "The technical domain of the invention could not be determined. "
        "The description may mix multiple unrelated technologies without clear focus."
    ),
    "Too few high-importance terms": (
        "The patent description contains insufficient technical detail. "
        "Key components and mechanisms are not described with enough specificity."
    ),
    "No essential features extracted": (
        "Independent claim 1 does not clearly define the essential technical features. "
        "Claims may be too broadly or vaguely drafted."
    ),
}


def _diagnose_writing_quality(completeness: dict) -> List[str]:
    issues = completeness.get("issues", [])
    diagnoses = []
    for issue in issues:
        diagnosis = _ISSUE_TO_WRITING_DIAGNOSIS.get(issue, issue)
        diagnoses.append(diagnosis)
    return diagnoses


def build_classification_quality(phase1: Any) -> Dict[str, Any]:
    if not isinstance(phase1, dict):
        return {
            "level": "unreliable",
            "phase1_status": "FAIL — No Extraction Data",
            "note": "Phase 1 did not produce valid extraction data.",
            "user_message": (
                "Classification is unreliable. The patent text could not be processed. "
                "Please verify the document format and resubmit."
            ),
        }

    if phase1.get("error"):
        return {
            "level": "unreliable",
            "phase1_status": "FAIL — Error During Extraction",
            "note": f"Phase 1 extraction error: {phase1.get('error', 'unknown')}",
            "user_message": (
                "Classification failed due to a processing error. "
                "The patent text may be malformed or too large. "
                "Please review the document and resubmit."
            ),
        }

    is_recovery = bool(phase1.get("phase1_status_note"))
    is_hardcoded_fallback = bool(
        phase1.get("phase1_emergency_anchor", {})
        .get("reasoning", "")
        .startswith("Hard-coded fallback")
    )

    if is_hardcoded_fallback:
        return {
            "level": "unreliable",
            "phase1_status": "FAIL — Hardcoded Fallback Applied",
            "note": "Emergency anchor also failed. Generic G06F fallback used.",
            "user_message": (
                "Classification is unreliable. The patent text could not be processed "
                "automatically. Please review the document format and resubmit, or perform "
                "manual classification."
            ),
        }

    completeness = phase1.get("phase1_completeness", {})
    status = completeness.get("status", "FAIL")
    score = completeness.get("score", 0)
    writing_issues = _diagnose_writing_quality(completeness)

    if is_recovery:
        user_msg = "Classification quality is low due to weak patent writing quality. "
        if writing_issues:
            user_msg += (
                "Specific issues detected: "
                + "; ".join(f"({i + 1}) {d}" for i, d in enumerate(writing_issues))
                + " "
            )
        user_msg += (
            "Recommendation: strengthen the technical description and ensure "
            "independent claim 1 explicitly recites all essential features of the invention."
        )
        if score < 40:
            user_msg += (
                " The patent text may also be in a non-English language, very short or "
                "incomplete, or describe a highly ambiguous cross-domain invention."
            )
        return {
            "level": "low",
            "phase1_status": "FAIL — Phase 1.1 Recovery Applied",
            "note": "Standard extraction failed after 3 attempts. Emergency anchor used.",
            "user_message": user_msg,
            "writing_quality_issues": writing_issues,
            "writing_quality_score": score,
        }

    if status == "WARN":
        return {
            "level": "medium",
            "phase1_status": "WARN",
            "note": "Extraction completed with minor gaps.",
            "user_message": (
                "Classification confidence is moderate. Some patent sections "
                "may have been unclear or missing."
            ),
        }

    return {
        "level": "high",
        "phase1_status": "PASS",
        "note": "Full multi-pass extraction succeeded.",
        "user_message": None,
    }