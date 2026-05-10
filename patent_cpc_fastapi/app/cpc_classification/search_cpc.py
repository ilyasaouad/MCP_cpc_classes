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
from .cpc_family_router import CPCFamilyRouter
from .cpc_hypothesis_consolidation import CPCHypothesisConsolidator
from .cpc_hypothesis_resolver import CPCHypothesisResolver
from .cpc_decision_tree import CPCDecisionTreeConstraint
from .cpc_hierarchy_engine import UniversalCPCHierarchyEngine
from .prompts import (
    label_claims,
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
        self.family_router = CPCFamilyRouter(knowledge_graph, max_families=3)

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

        # ─────────────────────────────
        # PHASE 1: LLM extraction
        # ─────────────────────────────
        try:
            phase1 = self.extractor.extract(description, labeled_claims)
        except Exception as e:
            logger.error("Phase 1 extraction failed: %s", str(e))
            return {
                "error": str(e),
                "phase1": {},
                "phase2": {},
                "phase3": [],
                "phase4": {},
                "phase5": {},
                "cpc": [],
            }

        # Validate Phase 1 has minimum required fields
        if (
            not phase1
            or not phase1.get("technical_object")
            or not phase1.get("core_function")
        ):
            error_msg = (
                "Phase 1 extraction returned incomplete data. "
                "The LLM did not produce valid semantic extraction. "
                "Possible causes: model timeout, invalid JSON response, or model not loaded."
            )
            logger.error(error_msg)
            return {
                "error": error_msg,
                "phase1": phase1 if phase1 else {},
                "phase2": {},
                "phase3": [],
                "phase4": {},
                "phase5": {},
                "cpc": [],
            }

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
        # PHASE 2A: CPC Family Router (PRIMARY CLASSIFIER)
        # ─────────────────────────────
        # Phase 2A is now the SOLE source of CPC family classification.
        # Phase 1 only provides semantic understanding - NO CPC predictions.
        phase2a_result = self.family_router.route(phase1)
        top_cpc_families = phase2a_result.get("families", [])
        phase2a_reasoning = phase2a_result.get("reasoning", "")
        phase2a_source = phase2a_result.get("source", "unknown")

        logger.info(
            "Phase 2A: Families=%s (source=%s)", top_cpc_families, phase2a_source
        )

        # Fallback: if fewer than 2 families, use defaults
        if len(top_cpc_families) < 2:
            from .cpc_family_router import FALLBACK_FAMILIES

            for fb in FALLBACK_FAMILIES:
                if fb not in top_cpc_families:
                    top_cpc_families.append(fb)
                if len(top_cpc_families) >= 2:
                    break
            phase2a_reasoning += " Fallback families added."
            logger.info("Phase 2A: Fallback applied, families=%s", top_cpc_families)

        # ─────────────────────────────
        # PHASE 2B: Restricted XML Expansion (MODIFIED)
        # ─────────────────────────────
        # Use KG + semantic signals for class suggestions, restricted to Phase 2A families
        graph_classes = []
        if self.knowledge_graph and self.knowledge_graph.embeddings:
            logger.info(
                "Phase 2B: Querying knowledge graph within families %s",
                top_cpc_families,
            )
            try:
                patent_text_for_search = f"{phase1.get('technical_object', '')} {phase1.get('core_function', '')} {system_context}"

                # Initialize hybrid retrieval if not already done
                if not self.knowledge_graph.bm25_index:
                    logger.info("Initializing hybrid retrieval (BM25 + cross-encoder)")
                    self.knowledge_graph.init_hybrid_retrieval()

                # Use hybrid search (BM25 recall + semantic + cross-encoder)
                if self.knowledge_graph.bm25_index:
                    logger.info(
                        "Using hybrid search (BM25 + embedding + cross-encoder)"
                    )
                    graph_results = self.knowledge_graph.hybrid_search(
                        text=patent_text_for_search,
                        top_k=20,
                        bm25_k=200,
                        use_cross_encoder=True,
                    )
                else:
                    # Fallback to legacy semantic search
                    logger.info("Using legacy semantic search (no BM25)")
                    extracted_terms = list(term_importance.keys())[:15]
                    graph_results = self.knowledge_graph.find_initial_classes(
                        patent_text=patent_text_for_search,
                        extracted_terms=extracted_terms,
                        top_k=10,
                    )

                # Only keep graph results that match Phase 2A families
                graph_classes = [
                    cls
                    for cls, score in graph_results
                    if score > 0.3
                    and any(cls.startswith(fam) for fam in top_cpc_families)
                ]
                logger.info(
                    "Phase 2B: Graph suggested classes (filtered): %s",
                    graph_classes,
                )
            except Exception as e:
                logger.warning("Phase 2B graph query failed: %s", e)

        # Build combined classes from Phase 2A families + KG suggestions
        combined_classes = []
        seen = set()
        for cls in graph_classes:
            if cls not in seen:
                seen.add(cls)
                combined_classes.append(cls)

        if not combined_classes:
            # If no specific classes from KG, use the families themselves for expansion
            combined_classes = top_cpc_families

        logger.info("Phase 2B: Combined classes (filtered): %s", combined_classes)

        # ─────────────────────────────
        # PHASE 2C: XML Expansion + Scoring (RESTRICTED)
        # ─────────────────────────────
        candidates = []
        score_margin = 0.0
        confidence_level = "medium"
        phase2b_candidate_count = 0
        phase2c_final_count = 0

        try:
            # Expand with family prefix filtering
            all_subgroups = self.xml_parser.expand_classes(
                combined_classes,
                include_non_allocatable=False,
                allowed_roots=top_cpc_families,
            )
            phase2b_candidate_count = len(all_subgroups)
            logger.info(
                "Phase 2C: Found %d total subgroups (restricted to families %s)",
                phase2b_candidate_count,
                top_cpc_families,
            )

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

                    # ── Object-Aware Keyword Mapping (GUARDRAIL) ──
                    # Prevent cross-domain leakage from ambiguous terms
                    # e.g., "clipping" in G06T (image) vs G06N (weight clipping)
                    all_terms_text = " ".join(term_importance.keys())

                    # Check for NN subject matter signals
                    nn_subject_signals = [
                        "neural network",
                        "neural networks",
                        "deep learning",
                        "llm",
                        "large language model",
                        "transformer",
                        "model compression",
                        "quantization",
                        "weight quantization",
                        "weight clipping",
                        "model optimization",
                        "inference optimization",
                    ]
                    has_nn_subject = any(
                        sig in all_terms_text for sig in nn_subject_signals
                    )

                    # Check for image domain signals
                    image_signals = [
                        "image",
                        "pixel",
                        "camera",
                        "visual",
                        "graphics",
                        "rendering",
                        "picture",
                        "photograph",
                    ]
                    has_image_signal = any(
                        sig in all_terms_text for sig in image_signals
                    )

                    # Cross-domain leakage penalty
                    family3 = symbol[:3] if len(symbol) >= 3 else symbol
                    if has_nn_subject and not has_image_signal:
                        # NN subject matter without image signals
                        if family3 in ["G06T", "G10K"]:
                            # Penalize image/acoustics families
                            score *= 0.2
                            logger.debug(
                                "Cross-domain guardrail: NN subject penalizes %s",
                                family3,
                            )

                    # Model optimization boost
                    if has_nn_subject and family3 == "G06N":
                        # Boost G06N when NN is the subject
                        score *= 1.5
                        logger.debug(
                            "Model optimization boost: G06N boosted for NN subject"
                        )

                    # Object-context binding for ambiguous terms
                    if "clipping" in all_terms_text:
                        if "weight" in all_terms_text or "parameter" in all_terms_text:
                            # Weight clipping → G06N
                            if family3 == "G06N":
                                score *= 1.4
                            elif family3 == "G06T":
                                score *= 0.3
                        elif any(sig in all_terms_text for sig in image_signals):
                            # Image clipping → G06T
                            if family3 == "G06T":
                                score *= 1.3

                    # Domain-specific boost using Phase 2A family scores
                    base_multiplier = 1.2  # W2: Default for unknown domains
                    class_prefix = symbol[:4] if len(symbol) >= 4 else symbol
                    family3 = symbol[:3] if len(symbol) >= 3 else symbol
                    phase2a_scores = phase2a_result.get("scores", {})
                    # Check 4-char prefix first, then 3-char family
                    if class_prefix in phase2a_scores:
                        base_multiplier = 1.0 + (phase2a_scores[class_prefix] * 1.0)
                    elif family3 in phase2a_scores:
                        base_multiplier = 1.0 + (phase2a_scores[family3] * 1.0)
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

                    for score, sg, _ in scored[:10]:
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

                    phase2c_final_count = len(candidates)

                logger.info(
                    "Phase 2C: Selected %d final candidates, margin=%.4f, confidence=%s",
                    phase2c_final_count,
                    score_margin,
                    confidence_level,
                )
        except Exception as e:
            logger.error("Phase 2 expansion failed: %s", e)

        # ─────────────────────────────
        # PHASE 3: Ranking
        # ─────────────────────────────
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:10]

        # ─────────────────────────────
        # PHASE 3.5: Decision Tree Constraint Layer
        # ─────────────────────────────
        phase35_result = {}
        try:
            dt_constraint = CPCDecisionTreeConstraint()
            phase35_result = dt_constraint.apply_constraints(ranked, phase1)
            ranked = phase35_result.get("phase35_candidates", ranked)
            logger.info(
                "Phase 3.5: Applied %d constraint rules. Domain=%s (conf=%.2f).",
                phase35_result.get("phase35_adjustments", 0),
                phase35_result.get("phase35_domain", "unknown"),
                phase35_result.get("phase35_domain_confidence", 0),
            )
        except Exception as e:
            logger.warning("Phase 3.5 decision tree failed: %s", e)

        # ─────────────────────────────
        # PHASE 3.6: Universal CPC Hierarchy Selection Layer
        # ─────────────────────────────
        phase36_result = {}
        try:
            hierarchy_engine = UniversalCPCHierarchyEngine()
            phase36_result = hierarchy_engine.apply_hierarchy(
                ranked,
                phase1,
                primary_domain=phase35_result.get("phase35_domain"),
            )
            ranked = phase36_result.get("phase36_candidates", ranked)
            logger.info(
                "Phase 3.6: Contribution type=%s, adjustments=%d",
                phase36_result.get("phase36_primary_type", "unknown"),
                phase36_result.get("phase36_adjustments", 0),
            )
        except Exception as e:
            logger.warning("Phase 3.6 hierarchy engine failed: %s", e)

        # ─────────────────────────────
        # PHASE 4: CPC Hypothesis Consolidation
        # ─────────────────────────────
        phase4_result = {}
        try:
            consolidator = CPCHypothesisConsolidator(max_hypotheses=3)
            phase4_result = consolidator.consolidate(
                ranked_candidates=ranked,
                term_importance=term_importance,
            )
            logger.info(
                "Phase 4: Consolidated %d candidates into %d hypotheses. Primary=%s, confidence=%s",
                len(ranked),
                len(phase4_result.get("phase4_hypotheses", [])),
                phase4_result.get("phase4_primary_family", "N/A"),
                phase4_result.get("phase4_confidence", "N/A"),
            )
        except Exception as e:
            logger.warning("Phase 4 consolidation failed: %s", e)

        # ─────────────────────────────
        # PHASE 5: CPC Hypothesis Resolution (deterministic)
        # ─────────────────────────────
        phase5_result = {}
        validated_candidates = []
        filtered_out = []
        best_code = None

        try:
            resolver = CPCHypothesisResolver()
            phase5_result = resolver.resolve(phase4_result, phase1)

            # Build validated_candidates from Phase 5 for backward compat
            primary = phase5_result.get("primary", {})
            if primary and primary.get("family"):
                # Find matching candidates from ranked
                for candidate in ranked:
                    if candidate["symbol"].startswith(primary["family"]):
                        validated_candidates.append(
                            {
                                "symbol": candidate["symbol"],
                                "title": candidate["title"],
                                "validation": "PASS",
                                "confidence": primary.get("confidence", "high"),
                                "justification": primary.get("reasoning", ""),
                            }
                        )

                best_code = {
                    "symbol": validated_candidates[0]["symbol"]
                    if validated_candidates
                    else "",
                    "title": validated_candidates[0]["title"]
                    if validated_candidates
                    else "",
                    "confidence": primary.get("confidence", "high"),
                    "reasoning": primary.get("reasoning", ""),
                }

            logger.info(
                "Phase 5: Primary=%s (score=%.3f), Secondary=%s",
                primary.get("family", "N/A"),
                primary.get("final_score", 0),
                phase5_result.get("secondary", {}).get("family", "None"),
            )
        except Exception as e:
            logger.warning("Phase 5 resolution failed: %s", e)
            # Fallback to top ranked candidate
            if ranked:
                validated_candidates = [
                    {
                        "symbol": node["symbol"],
                        "title": node["title"],
                        "validation": "PASS",
                        "confidence": "medium",
                        "justification": "Resolution skipped due to error",
                    }
                    for node in ranked[:3]
                ]
                best_code = {
                    "symbol": ranked[0]["symbol"],
                    "title": ranked[0]["title"],
                    "confidence": "medium",
                    "reasoning": "Resolution skipped due to error",
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
                "Claims terms weighted 2x. "
                f"Phase 2A routed to families {top_cpc_families} via {phase2a_source}. "
                f"Primary={phase2a_result.get('primary', 'N/A')}, "
                f"modality={phase2a_result.get('modality', 'unknown')}. "
                f"Phase 2B expanded to {phase2b_candidate_count} candidates, "
                f"Phase 2C scored down to {phase2c_final_count}."
            ),
            "score_margin": round(score_margin, 4),
            "confidence_level": confidence_level,
            "phase2a_families": top_cpc_families,
            "phase2a_primary": phase2a_result.get("primary", ""),
            "phase2a_secondary": phase2a_result.get("secondary", []),
            "phase2a_modality": phase2a_result.get("modality", "unknown"),
            "phase2a_reasoning": phase2a_reasoning,
            "phase2a_source": phase2a_source,
            "phase2b_candidate_count": phase2b_candidate_count,
            "phase2c_final_count": phase2c_final_count,
        }

        result = {
            "phase1": phase1,
            "phase2": phase2,
            "phase3": ranked,
            "phase35": phase35_result,
            "phase36": phase36_result,
            "phase4": phase4_result,
            "cpc": cpc,
        }

        # Store Phase 5 result (new deterministic resolver)
        if phase5_result:
            result["phase5"] = phase5_result
        elif validated_candidates or filtered_out:
            # Fallback to old format
            result["phase5"] = {
                "validated_candidates": validated_candidates,
                "filtered_out": filtered_out,
                "best_code": best_code,
            }

        # ── Premier: use Phase 7 recommendation if available ──
        premier_symbol = None
        if consistency_result.get("coherent") and consistency_result.get(
            "recommended_primary"
        ):
            premier_symbol = consistency_result["recommended_primary"]
            # Try to find title from validated candidates or ranked
            premier_title = ""
            for node in validated_candidates or ranked:
                if node.get("symbol") == premier_symbol:
                    premier_title = node.get("title", "")
                    break
            result["premier"] = {
                "symbol": premier_symbol,
                "title": premier_title,
                "confidence": "high",
                "reasoning": f"Phase 7 consistency check selected this as the recommended primary code. {consistency_result.get('reasoning', '')}",
            }
        elif best_code and best_code.get("symbol"):
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

    def classify_from_phase1(self, phase1: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run classification pipeline from Phase 2 onwards using pre-computed Phase 1 data.

        This bypasses LLM extraction entirely - useful for testing or when LLM is unavailable.

        Args:
            phase1: Pre-computed Phase 1 output with keys:
                - technical_object
                - core_function
                - system_context
                - domain_signals
                - terms
                - classification_strategy

        Returns:
            Full classification result (same format as classify())
        """
        logger.info("Running pipeline from Phase 2 with manual Phase 1 data")

        # Ensure minimum fields exist
        if not phase1.get("terms"):
            phase1["terms"] = []
        if not phase1.get("domain_signals"):
            phase1["domain_signals"] = []
        if not phase1.get("classification_strategy"):
            phase1["classification_strategy"] = "function-first"

        # Build term importance dict
        term_importance = {}
        for t in phase1.get("terms", []):
            if isinstance(t, dict):
                term = t.get("term", "").lower()
                importance = t.get("importance", 5)
                if term and len(term) > 3:
                    term_importance[term] = importance
            elif isinstance(t, str):
                term = t.lower()
                if len(term) > 3:
                    term_importance[term] = 5

        system_context = phase1.get("system_context", "").lower()
        core_function = phase1.get("core_function", "").lower()
        strategy = phase1.get("classification_strategy", "").lower()

        # ─────────────────────────────
        # PHASE 2A: Family Router
        # ─────────────────────────────
        phase2a_result = self.family_router.route(phase1)
        top_cpc_families = phase2a_result.get("families", [])
        phase2a_reasoning = phase2a_result.get("reasoning", "")
        phase2a_source = phase2a_result.get("source", "unknown")

        logger.info("Phase 2A (manual): Families=%s", top_cpc_families)

        # Fallback
        if len(top_cpc_families) < 2:
            from .cpc_family_router import FALLBACK_FAMILIES

            for fb in FALLBACK_FAMILIES:
                if fb not in top_cpc_families:
                    top_cpc_families.append(fb)
                if len(top_cpc_families) >= 2:
                    break

        # ─────────────────────────────
        # PHASE 2B/C: XML Expansion + Scoring
        # ─────────────────────────────
        # Use families as combined classes for expansion
        combined_classes = top_cpc_families

        candidates = []
        score_margin = 0.0
        confidence_level = "medium"
        phase2b_candidate_count = 0
        phase2c_final_count = 0

        try:
            all_subgroups = self.xml_parser.expand_classes(
                combined_classes,
                include_non_allocatable=False,
                allowed_roots=top_cpc_families,
            )
            phase2b_candidate_count = len(all_subgroups)

            if all_subgroups:
                # Score candidates
                scored = self._score_candidates(
                    all_subgroups,
                    term_importance,
                    system_context,
                    core_function,
                    strategy,
                    phase2a_result,
                    phase1.get("negative_signals", []),
                )

                if scored:
                    denom = scored[0][0]
                    if denom == 0:
                        denom = 1.0

                    for score, sg, _ in scored[:10]:
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

                    phase2c_final_count = len(candidates)
        except Exception as e:
            logger.error("Phase 2B/C error: %s", e)

        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:10]

        # ─────────────────────────────
        # PHASE 4: Consolidation
        # ─────────────────────────────
        phase4_result = {}
        try:
            consolidator = CPCHypothesisConsolidator(max_hypotheses=2)
            phase4_result = consolidator.consolidate(
                ranked_candidates=ranked,
                term_importance=term_importance,
            )
        except Exception as e:
            logger.warning("Phase 4 error: %s", e)

        # ─────────────────────────────
        # PHASE 5: Resolution
        # ─────────────────────────────
        phase5_result = {}
        try:
            resolver = CPCHypothesisResolver()
            phase5_result = resolver.resolve(phase4_result, phase1)
        except Exception as e:
            logger.warning("Phase 5 error: %s", e)

        # Build result
        cpc = [
            {"code": node["symbol"], "score": node.get("score", 0.0)}
            for node in ranked[:5]
        ]

        return {
            "phase1": phase1,
            "phase2": {
                "codes": [node["symbol"] for node in ranked],
                "reasoning": f"Manual Phase 1 → Phase 2A routed to {top_cpc_families}",
                "score_margin": round(score_margin, 4),
                "confidence_level": confidence_level,
                "phase2a_families": top_cpc_families,
                "phase2a_primary": phase2a_result.get("primary", ""),
                "phase2a_modality": phase2a_result.get("modality", "unknown"),
                "phase2a_reasoning": phase2a_reasoning,
                "phase2a_source": phase2a_source,
                "phase2b_candidate_count": phase2b_candidate_count,
                "phase2c_final_count": phase2c_final_count,
            },
            "phase3": ranked,
            "phase4": phase4_result,
            "phase5": phase5_result,
            "cpc": cpc,
        }

    def _score_candidates(
        self,
        all_subgroups,
        term_importance,
        system_context,
        core_function,
        strategy,
        phase2a_result,
        negative_signals,
    ):
        """Score candidates using TF-IDF."""
        import math
        from collections import Counter

        scored = []

        # Build document frequency
        doc_freq = Counter()
        title_count = len(all_subgroups)

        for sg in all_subgroups:
            context = sg.get("full_context", sg["title"]).lower()
            tokens = set(context.split())
            for token in tokens:
                if len(token) > 2:
                    doc_freq[token] += 1

        # Score each subgroup
        for sg in all_subgroups:
            score = 0.0
            matching_terms = 0
            context = sg.get("full_context", sg["title"]).lower()
            context_tokens = set(context.split())
            title_lower = sg["title"].lower()
            symbol = sg["symbol"]

            # Negative signal penalties
            for neg in negative_signals:
                if isinstance(neg, dict):
                    term = neg.get("term", "").lower()
                    conf = neg.get("confidence", 0.5)
                    if term and (term in context or term in title_lower):
                        score -= 5.0 * conf

            # Term matching
            for term, importance in term_importance.items():
                term_score = 0.0
                importance_weight = importance / 10.0
                term_tokens = set(term.split())

                # Multi-word phrase match
                if len(term.split()) >= 2 and term in context:
                    avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                        term_tokens
                    )
                    idf = math.log(title_count / max(avg_df, 1))
                    term_score += idf * importance_weight * 8
                    matching_terms += 1

                # Word overlap
                overlap = term_tokens & context_tokens
                if overlap:
                    avg_df = sum(doc_freq.get(t, 1) for t in overlap) / len(overlap)
                    idf = math.log(title_count / max(avg_df, 1))
                    term_score += idf * (importance / 5.0) * 3

                # Single-word match
                if len(term.split()) == 1 and term in context:
                    idf = math.log(title_count / max(doc_freq.get(term, 1), 1))
                    base_score = idf * importance_weight * 5
                    term_score += base_score
                    matching_terms += 1

                score += term_score

            # Context boost
            sys_tokens = set(system_context.split())
            sys_overlap = sys_tokens & context_tokens
            for token in sys_overlap:
                if len(token) > 2:
                    idf = math.log(title_count / max(doc_freq.get(token, 1), 1))
                    score += idf * 2

            func_tokens = set(core_function.split())
            func_overlap = func_tokens & context_tokens
            for token in func_overlap:
                if len(token) > 2:
                    idf = math.log(title_count / max(doc_freq.get(token, 1), 1))
                    score += idf * 4

            # Domain boost
            base_multiplier = 1.2
            class_prefix = symbol[:4] if len(symbol) >= 4 else symbol
            family3 = symbol[:3] if len(symbol) >= 3 else symbol
            phase2a_scores = phase2a_result.get("scores", {})
            if class_prefix in phase2a_scores:
                base_multiplier = 1.0 + (phase2a_scores[class_prefix] * 1.0)
            elif family3 in phase2a_scores:
                base_multiplier = 1.0 + (phase2a_scores[family3] * 1.0)
            score *= base_multiplier

            # Specificity bonus
            symbol_depth = symbol.count("/") + sum(
                symbol.count(d) for d in "0123456789"
            )
            if matching_terms >= 2:
                depth_bonus = min(symbol_depth * 0.5, 3.0)
                score += depth_bonus

            scored.append((score, sg, matching_terms))

        scored.sort(key=lambda x: -x[0])
        return scored
