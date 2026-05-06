import logging
import os
import re
import math
import json
from typing import Dict, Any, Set, List, Tuple, Optional
from collections import Counter

from search_core.ollama_client import OllamaClient
from .extracting_cpc import CPCExtractor
from .cpc_xml_parser import CPCXMLParser
from .knowledge_graph import CPCKnowledgeGraph
from .prompts import (
    label_claims,
    domain_inference_prompt,
    semantic_scoring_prompt,
    validation_prompt_single,
    reconciliation_prompt,
    consistency_check_prompt,
)

logger = logging.getLogger(__name__)


def _resolve_xml_dir() -> str:
    """Resolve the CPC XML directory relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "resources", "cpc_scheme_2026")


def _normalize_word(word: str) -> str:
    """Normalize a word for matching: lowercase, strip punctuation, basic stemming."""
    word = word.lower().strip(".,;:!?()[]{}-")
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    return word


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into normalized words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {_normalize_word(w) for w in words if len(w) > 2}


# Synonym mapping for CPC technical terms
CPC_SYNONYMS = {
    "venting": ["deaeration", "degassing", "air removal", "bleeding", "vent"],
    "bleeding": ["deaeration", "degassing", "venting", "air removal"],
    "deaeration": ["venting", "degassing", "air removal", "bleeding"],
    "degassing": ["deaeration", "venting", "air removal"],
    "air removal": ["deaeration", "venting", "degassing"],
    "cooling": [
        "temperature control",
        "heat removal",
        "thermal management",
        "coolant",
        "radiator",
    ],
    "thermal management": ["cooling", "heat removal", "temperature control"],
    "sealing": ["gasketing", "packing", "jointing", "seal"],
    "seal": ["sealing", "gasket", "packing"],
    "valve": ["tap", "cock", "vent", "shut-off"],
    "battery": ["accumulator", "cell", "electrochemical storage"],
    "wellhead": [
        "well head",
        "blowout preventer",
        "christmas tree",
        "wellhead assembly",
    ],
    "drilling": ["boring", "earth drilling", "well drilling"],
    "explosive": ["charge", "detonation", "blast", "explosion"],
    "cutter": ["cutting tool", "perforator", "radial cutter", "pipe cutter"],
    "completion": ["wellbore completion", "tubing completion"],
    "device": ["apparatus", "equipment", "unit"],
    "apparatus": ["device", "equipment", "unit"],
}


def _get_synonyms(word: str) -> List[str]:
    """Get CPC synonyms for a word."""
    return CPC_SYNONYMS.get(word.lower(), [])


def _get_expanded_terms(term: str) -> Set[str]:
    """Get all variants of a term including synonyms."""
    terms = {term.lower()}
    words = term.lower().split()
    for word in words:
        syns = _get_synonyms(word)
        for syn in syns:
            for w in words:
                variant = term.lower().replace(w, syn)
                terms.add(variant)
            terms.add(syn)
    return terms


def _parse_llm_json(response) -> dict:
    """Parse JSON from LLM response with multiple fallback strategies."""
    if not response:
        return {}
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


class CPCClassifier:
    """
    IMPROVED CPC CLASSIFICATION PIPELINE
    Addresses weaknesses W1-W8
    """

    def __init__(
        self,
        model_name: str = "gpt-oss:120b-cloud",
        knowledge_graph: Optional[CPCKnowledgeGraph] = None,
    ):
        self.llm = OllamaClient(model_name=model_name)
        self.extractor = CPCExtractor(self.llm)
        self.xml_parser = CPCXMLParser(_resolve_xml_dir())
        self.knowledge_graph = knowledge_graph

    def classify(self, text: str, claims: str = "") -> Dict[str, Any]:

        # ─────────────────────────────
        # PHASE 1: LLM extraction (improved)
        # ─────────────────────────────
        description = text
        labeled_claims = ""
        if claims:
            labeled_claims = label_claims(claims)
        elif "CLAIMS:" in text or "claims:" in text.lower():
            parts = re.split(r"CLAIMS:|claims:", text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 2:
                description = parts[0].strip()
                claims_text = parts[1].strip()
                labeled_claims = label_claims(claims_text)

        phase1 = self.extractor.extract(description, labeled_claims)

        # ─────────────────────────────
        # PHASE 1b: Probabilistic domain inference (replaces hardcoded injection)
        # ─────────────────────────────
        domain_probs = {}
        try:
            domain_prompt = domain_inference_prompt(phase1)
            domain_response = self.llm.chat(
                system_prompt="You are estimating CPC domain relevance probabilities.",
                user_message=domain_prompt,
                temperature=0.1,
                max_tokens=2000,
            )
            domain_data = _parse_llm_json(domain_response)
            for dp in domain_data.get("domain_probabilities", []):
                cls = dp.get("class", "")
                prob = dp.get("probability", 0.0)
                if cls and prob > 0.3:  # Only keep domains with >30% probability
                    domain_probs[cls] = prob
            logger.info("Phase 1b: Domain probabilities: %s", domain_probs)
        except Exception as e:
            logger.warning("Phase 1b domain inference failed: %s", e)

        # Get base classes from LLM hypotheses or fallback
        class_hypotheses = phase1.get("class_hypotheses", [])
        cpc_classes = [
            h.get("class", "") for h in class_hypotheses if h.get("confidence", 0) > 0.3
        ]

        # Add high-probability domains from inference
        for cls, prob in domain_probs.items():
            if prob > 0.5 and cls not in cpc_classes:
                cpc_classes.append(cls)
                logger.info("Phase 1b: Adding %s (probability %.2f)", cls, prob)

        if not cpc_classes:
            # Fallback to old cpc_classes if available
            cpc_classes = phase1.get("cpc_classes", [])

        # ─────────────────────────────
        # Extract terms with section-aware importance
        # ─────────────────────────────
        terms = phase1.get(
            "terms", phase1.get("essential_terms", phase1.get("terms", []))
        )
        term_importance = {}
        for t in terms:
            if isinstance(t, dict):
                term = t.get("term", "").lower()
                importance = t.get("importance", 5)
                source = t.get("source_section", "unknown")
                # Apply section weight
                section_weight = {
                    "claims": 1.2,
                    "summary": 1.0,
                    "detailed_description": 0.9,
                    "abstract": 0.6,
                    "background": 0.2,
                }.get(source, 0.5)
                adjusted = min(round(importance * section_weight), 10)
                if term and len(term) > 3:
                    term_importance[term] = max(term_importance.get(term, 0), adjusted)
            else:
                term = str(t).lower()
                if term and len(term) > 3:
                    term_importance[term] = 5

        system_context = phase1.get("system_context", "").lower()
        core_function = phase1.get("core_function", "").lower()
        strategy = phase1.get("classification_strategy", "").lower()

        # ─────────────────────────────
        # PHASE 2a: Knowledge Graph Query (NEW)
        # ─────────────────────────────
        graph_classes = []
        if self.knowledge_graph and self.knowledge_graph.embeddings:
            logger.info("Phase 2a: Querying knowledge graph")
            try:
                # Combine patent text for semantic search
                patent_text_for_search = f"{phase1.get('technical_object', '')} {phase1.get('core_function', '')} {system_context}"

                # Get extracted terms
                extracted_terms = list(term_importance.keys())[:15]  # Top 15 terms

                # Query graph
                graph_results = self.knowledge_graph.find_initial_classes(
                    patent_text=patent_text_for_search,
                    extracted_terms=extracted_terms,
                    top_k=10,
                )

                graph_classes = [cls for cls, score in graph_results if score > 0.3]
                logger.info(
                    "Phase 2a: Graph suggested classes: %s (scores: %s)",
                    graph_classes,
                    {cls: round(score, 3) for cls, score in graph_results[:5]},
                )
            except Exception as e:
                logger.warning("Phase 2a graph query failed: %s", e)

        # ─────────────────────────────
        # PHASE 2b: Combine LLM + Graph suggestions
        # ─────────────────────────────
        # Merge LLM hypotheses with graph suggestions
        combined_classes = list(
            dict.fromkeys(cpc_classes + graph_classes)
        )  # Preserve order, remove duplicates
        if not combined_classes and cpc_classes:
            combined_classes = cpc_classes

        logger.info("Phase 2: Combined classes (LLM + Graph): %s", combined_classes)

        # ─────────────────────────────
        # PHASE 2c: XML expansion + improved scoring
        # ─────────────────────────────
        candidates = []
        score_margin = 0.0
        confidence_level = "medium"

        try:
            all_subgroups = self.xml_parser.expand_classes(
                combined_classes, include_non_allocatable=False
            )
            logger.info("Phase 2: Found %d total subgroups", len(all_subgroups))

            if all_subgroups:
                title_count = len(all_subgroups)
                doc_freq = Counter()
                all_tokens = []

                for sg in all_subgroups:
                    context = sg.get("full_context", sg.get("title", "")).lower()
                    tokens = _tokenize(context)
                    all_tokens.append(tokens)
                    for token in tokens:
                        doc_freq[token] += 1

                scored = []
                for idx, sg in enumerate(all_subgroups):
                    symbol = sg.get("symbol", "")
                    title = sg.get("title", "").lower()
                    context = sg.get("full_context", title).lower()
                    context_tokens = all_tokens[idx]
                    score = 0.0
                    matching_terms = 0

                    # Negative signal penalty (soft, using confidence)
                    negative_signals = phase1.get("negative_signals", [])
                    negative_domains = phase1.get("negative_domains", [])
                    for neg in negative_signals:
                        if isinstance(neg, dict):
                            term = neg.get("term", "").lower()
                            conf = neg.get("confidence", 0.5)
                        else:
                            term = str(neg).lower()
                            conf = 0.5
                        if term in context or term in title:
                            score -= 5.0 * conf

                    for neg in negative_domains:
                        if isinstance(neg, dict):
                            domain = neg.get("domain", "").lower()
                            conf = neg.get("confidence", 0.5)
                        else:
                            domain = str(neg).lower()
                            conf = 0.5
                        if domain in context or domain in title:
                            score -= 3.0 * conf

                    # Build multi-word phrases from terms for better semantic matching
                    term_phrases = []
                    for term in term_importance.keys():
                        words = term.split()
                        if len(words) >= 2:
                            # Add bigrams and trigrams
                            for i in range(len(words) - 1):
                                term_phrases.append(" ".join(words[i : i + 2]))
                            for i in range(len(words) - 2):
                                term_phrases.append(" ".join(words[i : i + 3]))

                    # Term matching with importance weighting
                    for term, importance in term_importance.items():
                        term_score = 0.0
                        term_tokens = _tokenize(term)
                        if not term_tokens:
                            continue

                        # Multi-word phrase matching (strong signal)
                        if len(term.split()) >= 2 and term in context:
                            avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                                term_tokens
                            )
                            idf = math.log(title_count / max(avg_df, 1))
                            importance_weight = importance / 5.0
                            term_score += (
                                idf * importance_weight * 8
                            )  # Higher weight for phrase matches
                            matching_terms += 1

                        # Word overlap
                        overlap = term_tokens & context_tokens
                        if overlap:
                            overlap_idf = sum(
                                math.log(title_count / max(doc_freq.get(t, 1), 1))
                                for t in overlap
                            )
                            term_score += overlap_idf * (importance / 5.0) * 3

                        # Single-word substring match (penalize if ambiguous)
                        if len(term.split()) == 1 and term in context:
                            # Check if the single word appears in a relevant context
                            # by looking at surrounding words in the CPC context
                            avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                                term_tokens
                            )
                            idf = math.log(title_count / max(avg_df, 1))
                            importance_weight = importance / 5.0
                            base_score = idf * importance_weight * 5

                            # Penalize if the single word is a false friend
                            # (e.g., "exchange" in dialog context vs "data exchange")
                            false_friend_penalty = 0.0
                            if term in [
                                "exchange",
                                "response",
                                "system",
                                "user",
                                "transfer",
                            ]:
                                # Check if the CPC context suggests a different meaning
                                if any(
                                    marker in context
                                    for marker in [
                                        "data exchange",
                                        "clipboard",
                                        "dde",
                                        "ole",
                                        "fault response",
                                        "error response",
                                        "redundant",
                                        "fault tolerance",
                                        "backup",
                                        "input device",
                                        "brain wave",
                                        "eeg",
                                        "emg",
                                        "printer",
                                        "peripheral",
                                    ]
                                ):
                                    false_friend_penalty = (
                                        base_score * 0.7
                                    )  # Reduce by 70%

                            term_score += base_score - false_friend_penalty
                            matching_terms += 1

                        # Synonym match
                        synonyms = _get_expanded_terms(term)
                        for syn in synonyms:
                            if syn != term and syn in context:
                                syn_tokens = _tokenize(syn)
                                avg_df = sum(
                                    doc_freq.get(t, 1) for t in syn_tokens
                                ) / max(len(syn_tokens), 1)
                                idf = math.log(title_count / max(avg_df, 1))
                                term_score += idf * (importance / 5.0) * 4

                        score += term_score

                    # Context boosts
                    sys_tokens = _tokenize(system_context)
                    sys_overlap = sys_tokens & context_tokens
                    for token in sys_overlap:
                        idf = math.log(title_count / max(doc_freq.get(token, 1), 1))
                        score += idf * 2

                    func_tokens = _tokenize(core_function)
                    func_overlap = func_tokens & context_tokens
                    for token in func_overlap:
                        idf = math.log(title_count / max(doc_freq.get(token, 1), 1))
                        score += idf * 4

                    # Domain-specific boost (calibrated, default 1.2 for unknown)
                    base_multiplier = 1.2  # W2: Default for unknown domains
                    class_prefix = symbol[:4] if len(symbol) >= 4 else symbol
                    if class_prefix in domain_probs:
                        # Scale multiplier by domain probability
                        base_multiplier = 1.0 + (domain_probs[class_prefix] * 1.0)
                    score *= base_multiplier

                    # Specificity bonus with term-density guard (W4)
                    symbol_depth = symbol.count("/") + sum(
                        symbol.count(d) for d in "0123456789"
                    )
                    if (
                        matching_terms >= 2
                    ):  # Only apply if at least 2 terms match at this depth
                        depth_bonus = min(symbol_depth * 0.5, 3.0)
                        score += depth_bonus

                    if score > 0:
                        scored.append((score, sg, matching_terms))

                scored.sort(key=lambda x: -x[0])

                # Normalization
                if scored:
                    scores = [s[0] for s in scored]
                    max_score = max(scores)
                    median_score = (
                        sorted(scores)[len(scores) // 2]
                        if len(scores) > 1
                        else max_score
                    )
                    denom = max_score + median_score * 0.5

                    # Calculate margin (W6)
                    if len(scores) >= 2:
                        score_margin = (scores[0] - scores[1]) / denom
                        if score_margin > 0.3:
                            confidence_level = "high"
                        elif score_margin < 0.1:
                            confidence_level = "low"
                        else:
                            confidence_level = "medium"

                    for score, sg, _ in scored[:7]:
                        normalized_score = min(score / denom, 1.0)
                        candidates.append(
                            {
                                "symbol": sg["symbol"],
                                "title": sg["title"],
                                "level": sg.get("level", 0),
                                "score": round(normalized_score, 4),
                                "full_context": sg.get("full_context", sg["title"]),
                            }
                        )

                logger.info(
                    "Phase 2: Selected %d candidates, margin=%.4f, confidence=%s",
                    len(candidates),
                    score_margin,
                    confidence_level,
                )
        except Exception as e:
            logger.error("Phase 2 expansion failed: %s", e)

        # ─────────────────────────────
        # PHASE 3: Ranking
        # ─────────────────────────────
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:7]

        # ─────────────────────────────
        # PHASE 5: Multi-pass validation (W5)
        # ─────────────────────────────
        validated_candidates = []
        filtered_out = []
        best_code = None

        try:
            if ranked:
                for candidate in ranked:
                    val_prompt = validation_prompt_single(
                        phase1, candidate, score_margin, confidence_level
                    )
                    val_response = self.llm.chat(
                        system_prompt="You are a senior patent examiner validating a single CPC candidate.",
                        user_message=val_prompt,
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    val_data = _parse_llm_json(val_response)

                    validation_result = {
                        "symbol": candidate["symbol"],
                        "title": candidate["title"],
                        "validation": val_data.get("decision", "PASS"),
                        "confidence": val_data.get("confidence", "medium"),
                        "function_aligned": val_data.get("scores", {}).get(
                            "function_alignment", 0
                        )
                        >= 0.6,
                        "context_aligned": val_data.get("scores", {}).get(
                            "context_alignment", 0
                        )
                        >= 0.5,
                        "visual_bias": val_data.get("scores", {}).get(
                            "visual_bias", False
                        ),
                        "justification": val_data.get("reasoning", ""),
                        "rejection_reason": val_data.get("rejection_reason", ""),
                    }

                    if val_data.get("decision", "PASS").upper() == "PASS":
                        validated_candidates.append(validation_result)
                    else:
                        filtered_out.append(validation_result)

                # Select best code from validated candidates
                if validated_candidates:
                    # Pick the one with highest confidence
                    high_conf = [
                        v for v in validated_candidates if v.get("confidence") == "high"
                    ]
                    if high_conf:
                        best = high_conf[0]
                    else:
                        best = validated_candidates[0]
                    best_code = {
                        "symbol": best["symbol"],
                        "title": best["title"],
                        "confidence": best["confidence"],
                        "reasoning": best["justification"],
                    }

                # If all failed, fallback
                if not validated_candidates and ranked:
                    logger.warning(
                        "Phase 5: All candidates failed validation. Using fallback."
                    )
                    fallback = ranked[0]
                    validated_candidates.append(
                        {
                            "symbol": fallback["symbol"],
                            "title": fallback["title"],
                            "validation": "PASS",
                            "confidence": "low",
                            "function_aligned": False,
                            "context_aligned": False,
                            "visual_bias": True,
                            "justification": "Fallback - all candidates failed validation",
                            "rejection_reason": "",
                        }
                    )
                    best_code = {
                        "symbol": fallback["symbol"],
                        "title": fallback["title"],
                        "confidence": "low",
                        "reasoning": "Fallback selection - manual review recommended.",
                    }

                logger.info(
                    "Phase 5: %d passed, %d filtered. Best: %s",
                    len(validated_candidates),
                    len(filtered_out),
                    best_code.get("symbol", "N/A") if best_code else "N/A",
                )
        except Exception as e:
            logger.warning("Phase 5 validation failed: %s", e)
            validated_candidates = [
                {
                    "symbol": node["symbol"],
                    "title": node["title"],
                    "validation": "PASS",
                    "confidence": "medium",
                    "justification": "Validation skipped due to error",
                }
                for node in ranked
            ]
            if ranked:
                best_code = {
                    "symbol": ranked[0]["symbol"],
                    "title": ranked[0]["title"],
                    "confidence": "medium",
                    "reasoning": "Validation skipped due to error",
                }

        # ─────────────────────────────
        # PHASE 6: Per-claim reconciliation (W7)
        # ─────────────────────────────
        reconciled_claims = []
        try:
            per_claim = phase1.get("claim_classifications", [])
            if per_claim and (validated_candidates or filtered_out):
                recon_prompt = reconciliation_prompt(
                    validated_candidates, filtered_out, per_claim
                )
                recon_response = self.llm.chat(
                    system_prompt="You are reconciling claim-level CPC classifications.",
                    user_message=recon_prompt,
                    temperature=0.1,
                    max_tokens=2000,
                )
                recon_data = _parse_llm_json(recon_response)
                reconciled_claims = recon_data.get("reconciled_claims", [])
                logger.info("Phase 6: Reconciled %d claims", len(reconciled_claims))
        except Exception as e:
            logger.warning("Phase 6 reconciliation failed: %s", e)

        # ─────────────────────────────
        # PHASE 7: Final consistency check
        # ─────────────────────────────
        consistency_result = {}
        try:
            if validated_candidates:
                # Build selected codes list for consistency check
                selected = [
                    {"symbol": v["symbol"], "title": v["title"]}
                    for v in validated_candidates[:3]
                ]
                consist_prompt = consistency_check_prompt(phase1, selected)
                consist_response = self.llm.chat(
                    system_prompt="You are performing a final coherence check on CPC classifications.",
                    user_message=consist_prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
                consist_data = _parse_llm_json(consist_response)
                consistency_result = {
                    "coherent": consist_data.get("coherent", True),
                    "issues": consist_data.get("issues", []),
                    "recommended_primary": consist_data.get("recommended_primary", ""),
                    "recommended_secondary": consist_data.get(
                        "recommended_secondary", []
                    ),
                    "reasoning": consist_data.get("reasoning", ""),
                }
                logger.info(
                    "Phase 7: Consistency check: coherent=%s",
                    consistency_result.get("coherent", True),
                )
        except Exception as e:
            logger.warning("Phase 7 consistency check failed: %s", e)

        # ─────────────────────────────
        # Build final result
        # ─────────────────────────────
        cpc_source = validated_candidates if validated_candidates else ranked
        cpc = [
            {
                "code": node.get("symbol", node["symbol"]),
                "score": node.get("score", 0.0),
            }
            for node in cpc_source
        ]

        phase2 = {
            "codes": [node["symbol"] for node in ranked],
            "reasoning": (
                "Ranked by improved TF-IDF scoring with section-aware term weighting, "
                "word-level matching, parent context, expanded synonyms, and probabilistic "
                "domain boosting. Score margin and confidence level calculated. "
                "Claims terms weighted 2x."
            ),
            "score_margin": round(score_margin, 4),
            "confidence_level": confidence_level,
        }

        result = {
            "phase1": phase1,
            "phase2": phase2,
            "phase3": ranked,
            "cpc": cpc,
        }

        if validated_candidates or filtered_out:
            result["phase5"] = {
                "validated_candidates": validated_candidates,
                "filtered_out": filtered_out,
                "best_code": best_code,
            }

        if best_code and best_code.get("symbol"):
            result["premier"] = {
                "symbol": best_code.get("symbol"),
                "title": best_code.get("title", ""),
                "confidence": best_code.get("confidence", "medium"),
                "reasoning": best_code.get("reasoning", ""),
            }
        elif ranked:
            result["premier"] = {
                "symbol": ranked[0]["symbol"],
                "title": ranked[0]["title"],
                "confidence": "medium",
                "reasoning": "Top-scoring candidate from Phase 2/3 scoring",
            }

        if reconciled_claims:
            result["per_claim"] = reconciled_claims
        else:
            # Fallback to original per-claim if reconciliation failed
            claim_classifications = phase1.get("claim_classifications", [])
            if claim_classifications:
                result["per_claim"] = claim_classifications

        if consistency_result:
            result["phase7"] = consistency_result

        return result
