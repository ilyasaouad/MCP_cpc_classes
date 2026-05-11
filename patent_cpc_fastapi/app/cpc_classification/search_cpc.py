import logging
import os
import re
import math
import json
import numpy as np
from typing import Dict, Any, Set, List, Tuple, Optional
from collections import Counter

from search_core.ollama_client import OllamaClient
from .extracting_cpc import CPCExtractor
from .cpc_xml_parser import CPCXMLParser
from .knowledge_graph import CPCKnowledgeGraph
from .cpc_family_router import CPCFamilyRouter
from .cpc_layer_decomposer import CPCLayerDecomposer, merge_layers_to_family_list
from .cpc_hypothesis_consolidation import CPCHypothesisConsolidator
from .cpc_hypothesis_resolver import CPCHypothesisResolver
from .cpc_decision_tree import CPCDecisionTreeConstraint
from .cpc_cross_domain_validator import CrossDomainValidator
from .cpc_role_labeling import CPCRoleLabeling
from .cpc_role_classifier import CPCRoleClassifier, apply_role_scoring
from .cpc_phase2d_anchor import Phase2DSubclassAnchor
from .technical_weight_analyzer import (
    TechnicalWeightAnalyzer,
    apply_technical_weight_analysis,
)
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


def _tokenize_with_bigrams(text: str) -> Tuple[Set[str], Set[str]]:
    """Tokenize text into normalized unigrams AND bigrams.

    Bigrams capture technical phrases like "system_prompt", "user_prompt",
    "program_code" that anchor terms to specific technical domains.
    Returns (unigrams, bigrams) where bigrams use '_' as delimiter.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    normalized = [_normalize_word(w) for w in words if len(w) > 2]
    unigrams = set(normalized)
    bigrams = set()
    for i in range(len(normalized) - 1):
        big = f"{normalized[i]}_{normalized[i + 1]}"
        bigrams.add(big)
    return unigrams, bigrams


def _make_term_bigrams(term: str) -> List[str]:
    """Generate bigrams from a multi-word term (used for patent term bigram matching)."""
    words = [_normalize_word(w) for w in term.lower().split() if len(w) > 2]
    if len(words) < 2:
        return []
    return [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]


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
        self.layer_decomposer = CPCLayerDecomposer(knowledge_graph)

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
        # PHASE 1.5: Invention Role Classification
        # ─────────────────────────────
        phase15_result = {}
        try:
            role_classifier = CPCRoleClassifier(self.llm)
            phase15_result = role_classifier.classify_role(phase1)
            logger.info(
                "Phase 1.5: Role=%s (conf=%.2f)",
                phase15_result.get("role", "UNKNOWN"),
                phase15_result.get("confidence", 0),
            )
        except Exception as e:
            logger.warning("Phase 1.5 role classification failed: %s", e)
            phase15_result = {"role": "SYSTEM", "confidence": 0.5}

        # ─────────────────────────────
        # TECHNICAL WEIGHT ANALYSIS (NEW)
        # ─────────────────────────────
        tcr_result = {}
        try:
            tcr_analyzer = TechnicalWeightAnalyzer()
            tcr_result = tcr_analyzer.analyze(phase1)
            logger.info(
                "TCR Analysis: TCR=%.3f, force_flag=%s, comp_weight=%.2f, phys_weight=%.2f",
                tcr_result.get("tcr", 1.0),
                tcr_result.get("force_flag", "HYBRID_INVENTION"),
                tcr_result.get("computational_weight", 0),
                tcr_result.get("physical_weight", 0),
            )
        except Exception as e:
            logger.warning("Technical weight analysis failed: %s", e)
            tcr_result = {"tcr": 1.0, "force_flag": "HYBRID_INVENTION"}

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
        # PHASE 2A: CPC Layer Decomposition (REPLACES family routing)
        # ─────────────────────────────
        # Multi-layer decomposition: each technical layer maps to CPC independently.
        # NO collapsing into single family. NO cross-layer penalties.
        # Pass tcr_result to guide layer scoring (FORCE_SOFTWARE_CORE boosts pure_software layer)
        phase2a_result = self.layer_decomposer.decompose(
            phase1, phase15_result, tcr_result
        )

        # Extract layer decomposition results
        layer_result = phase2a_result
        layers = layer_result.get("layers", {})
        primary_layer = layer_result.get("primary_layer", "application")
        layer_scores = layer_result.get("layer_scores", {})
        layer_reasoning = layer_result.get("reasoning", "")

        # Get flat family list for backward compatibility with Phase 2B/2C
        # Pass tcr_result to filter application layer CPCs when TCR indicates pure software
        top_cpc_families = merge_layers_to_family_list(layer_result, tcr_result)
        phase2a_reasoning = layer_reasoning
        phase2a_source = "layer_decomposition"

        logger.info(
            "Phase 2A: Layer decomposition - primary=%s, layers=%s",
            primary_layer,
            list(layers.keys()),
        )
        logger.info(
            "Phase 2A: Layer scores - %s",
            {k: f"{v:.2f}" for k, v in layer_scores.items()},
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

        # Extract 4‑character subclass prefixes from Phase 2A TECHNICAL layers only
        # (excludes application layer). Used for XML expansion and as fallback
        # combined_classes. Phase 2D will filter precisely later.
        _tech_layers = {"pure_software", "data_reasoning", "interaction", "control"}
        _re_4char = re.compile(r"^([A-Z]\d{2}[A-Z])")
        allowed_roots = []
        root_seen = set()
        layers = layer_result.get("layers", {})
        for layer_name in _tech_layers:
            for cand in layers.get(layer_name, []):
                sym = cand.get("symbol", "")
                m = _re_4char.match(sym)
                if m:
                    prefix = m.group(1)
                    if prefix not in root_seen:
                        root_seen.add(prefix)
                        allowed_roots.append(prefix)
        logger.info(
            "Phase 2B: Allowed roots (4-char prefixes from tech layers): %s",
            allowed_roots,
        )

        if not combined_classes:
            # Use the technical-layer 4-char prefixes (NOT raw top_cpc_families
            # which includes application-layer noise like G06Q)
            combined_classes = allowed_roots

        # Pre-filter: only include classes that actually have XML files on disk.
        # This prevents wasted parse_file() calls for codes like G05D, G06K
        # that exist in layer definitions but have no corresponding CPC scheme file.
        xml_dir = _resolve_xml_dir()
        available_files = set(os.listdir(xml_dir))
        valid_combined = []
        phase2b_skipped = []
        for cls in combined_classes:
            xml_name = f"cpc-scheme-{cls}.xml"
            if xml_name in available_files:
                valid_combined.append(cls)
            else:
                # Try wildcard match (e.g., G05 → G05B, G05D files)
                pattern = f"cpc-scheme-{cls}"
                if any(f.startswith(pattern) for f in available_files):
                    valid_combined.append(cls)
                else:
                    phase2b_skipped.append(cls)
        if phase2b_skipped:
            logger.warning(
                "Phase 2B: Skipped %d classes with no XML files: %s",
                len(phase2b_skipped),
                phase2b_skipped,
            )
        combined_classes = valid_combined
        allowed_roots = [r for r in allowed_roots if r in valid_combined]

        logger.info("Phase 2B: Combined classes (filtered): %s", combined_classes)
        logger.info(
            "Phase 2B: Allowed roots (4-char prefixes from tech layers): %s",
            allowed_roots,
        )

        # ─────────────────────────────
        # PHASE 2C: XML Expansion + Scoring (RESTRICTED)
        # ─────────────────────────────
        candidates = []
        all_candidates = []
        score_margin = 0.0
        confidence_level = "medium"
        phase2b_candidate_count = 0
        phase2c_final_count = 0
        # Per-class expansion counts for Phase 2B display
        phase2b_expansion_counts = {}  # {prefix: subgroup_count}

        try:
            # Expand with family prefix filtering
            all_subgroups = self.xml_parser.expand_classes(
                combined_classes,
                include_non_allocatable=False,
                allowed_roots=allowed_roots,
            )
            phase2b_candidate_count = len(all_subgroups)

            # Collect per-class expansion counts
            for sg in all_subgroups:
                sym = sg.get("symbol", "")
                prefix = sym[:4] if len(sym) >= 4 else sym[:3]
                phase2b_expansion_counts[prefix] = (
                    phase2b_expansion_counts.get(prefix, 0) + 1
                )

            logger.info(
                "Phase 2B: Expanded %d total subgroups across %d families: %s",
                phase2b_candidate_count,
                len(phase2b_expansion_counts),
                {k: v for k, v in sorted(phase2b_expansion_counts.items())},
            )

            if all_subgroups:
                title_count = len(all_subgroups)
                doc_freq = Counter()
                all_tokens = []
                all_bigrams = []  # NEW: track bigrams per subgroup

                for sg in all_subgroups:
                    context = sg.get("full_context", sg.get("title", "")).lower()
                    unigrams, bigrams = _tokenize_with_bigrams(context)
                    all_tokens.append(unigrams)
                    all_bigrams.append(bigrams)
                    for token in unigrams:
                        doc_freq[token] += 1
                    for bigram in bigrams:
                        doc_freq[bigram] += 1

                scored = []
                for idx, sg in enumerate(all_subgroups):
                    symbol = sg.get("symbol", "")
                    title = sg.get("title", "").lower()
                    context = sg.get("full_context", title).lower()
                    context_tokens = all_tokens[idx]
                    context_bigrams = all_bigrams[
                        idx
                    ]  # NEW: bigrams for this candidate
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

                        # Bigram matching (NEW: captures technical phrases like "system_prompt")
                        term_bigrams = _make_term_bigrams(term)
                        if term_bigrams and context_bigrams:
                            for tb in term_bigrams:
                                if tb in context_bigrams:
                                    avg_df = doc_freq.get(tb, 1)
                                    idf = math.log(title_count / max(avg_df, 1))
                                    term_score += idf * (importance / 5.0) * 6
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

                # ═══════════════════════════════════════════════
                # HYBRID SCORING: TF-IDF + Semantic Similarity
                # Formula: Final = 0.4 × TF-IDF_norm + 0.6 × Sem_Sim
                # ═══════════════════════════════════════════════
                sem_scores = {}
                if self.knowledge_graph and self.knowledge_graph.embeddings:
                    patent_text = (
                        phase1.get("technical_object", "")
                        + " "
                        + phase1.get("core_function", "")
                    )
                    sem_scores = self._compute_semantic_scores(
                        [sg for _, sg, _ in scored], patent_text
                    )

                if sem_scores:
                    # Replace tuple scores with actual hybrid scores for proper normalization
                    max_tfidf = max(s[0] for s in scored) if scored else 1.0
                    hybrid_scored = []
                    for tfidf_score, sg, matching_terms in scored:
                        sym = sg.get("symbol", "")
                        tfidf_norm = tfidf_score / max_tfidf if max_tfidf > 0 else 0.0
                        sem = sem_scores.get(sym, 0.0)
                        hybrid = 0.4 * tfidf_norm + 0.6 * sem
                        hybrid_scored.append((round(hybrid, 6), sg, matching_terms))
                    scored = hybrid_scored
                    scored.sort(key=lambda x: -x[0])
                    logger.info(
                        "Hybrid scoring applied (0.4×TF-IDF + 0.6×Semantic) for %d candidates",
                        len(scored),
                    )

                # Normalization (common to both TF-IDF-only and hybrid)
                all_candidates = []
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
                        score_margin = round((scores[0] - scores[1]) / denom, 6)
                        if score_margin > 0.3:
                            confidence_level = "high"
                        elif score_margin < 0.1:
                            confidence_level = "low"
                        else:
                            confidence_level = "medium"

                    # Normalize ALL scored candidates (no truncation yet —
                    # Find-Until-Full below will determine the depth)
                    all_candidates = []
                    for score, sg, _ in scored:
                        normalized_score = min(score / denom, 1.0)
                        all_candidates.append(
                            {
                                "symbol": sg["symbol"],
                                "title": sg["title"],
                                "level": sg.get("level", 0),
                                "score": round(normalized_score, 6),
                                "full_context": sg.get("full_context", sg["title"]),
                            }
                        )

                    phase2c_final_count = len(all_candidates)

                logger.info(
                    "Phase 2C: Scored %d total candidates, margin=%.6f, confidence=%s",
                    phase2c_final_count,
                    score_margin,
                    confidence_level,
                )
        except Exception as e:
            logger.error("Phase 2 expansion failed: %s", e)

        # ─────────────────────────────
        # PHASE 2D: Subclass Structural Anchor + Find-Until-Full
        # ─────────────────────────────
        # Progressive expansion: start with top 500, deep-search to 1000,
        # then exhaust all candidates if still under the 20-survivor quota.
        phase2d_result = {}
        find_until_full_log = []  # [{depth, survivors, triggered_deep}, ...]
        candidates = []
        try:
            anchor_filter = Phase2DSubclassAnchor()
            search_depths = [500, 1000, len(all_candidates) if all_candidates else 0]
            search_depths = [d for d in search_depths if d > 0]

            for depth in search_depths:
                batch = all_candidates[:depth]
                phase2d_result = anchor_filter.filter(
                    candidates=batch,
                    layer_result=layer_result,
                    max_result=depth,
                )
                survivors = phase2d_result.get("purified_candidates", [])
                kept = len(survivors)

                entry = {
                    "depth": depth,
                    "survivors_found": kept,
                    "deep_search_triggered": kept < 20,
                }
                find_until_full_log.append(entry)

                if kept >= 20 or depth == search_depths[-1]:
                    if depth == 500 and kept >= 20:
                        logger.info(
                            "Find-Until-Full: Scanned %d to find %d valid technical anchors. ✓ Quota met.",
                            depth,
                            kept,
                        )
                    elif kept >= 20:
                        logger.info(
                            "Deep Search required. Scanned %d to find %d valid technical anchors. ✓ Quota met.",
                            depth,
                            kept,
                        )
                    else:
                        logger.warning(
                            "Find-Until-Full exhausted: Scanned %d to find only %d valid technical anchors. Using all survivors.",
                            depth,
                            kept,
                        )
                    candidates = survivors
                    break
                else:
                    logger.info(
                        "Deep Search required. Scanned %d to find %d valid technical anchors. Expanding to next depth.",
                        depth,
                        kept,
                    )

            logger.info(
                "Phase 2D: Final — kept=%d discarded=%d anchor_subclasses=%s",
                phase2d_result.get("kept_count", 0),
                phase2d_result.get("discarded_count", 0),
                ", ".join(phase2d_result.get("anchor_set", [])),
            )
        except Exception as e:
            logger.warning("Phase 2D anchor filter failed: %s", e)

        # ─────────────────────────────
        # PHASE 3: Ranking (Top 20 for Phase 8 completeness)
        # ─────────────────────────────
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

        # ─────────────────────────────
        # PHASE 3.5: Decision Tree Constraint Layer
        # ─────────────────────────────
        phase35_result = {}
        try:
            dt_constraint = CPCDecisionTreeConstraint()
            phase35_result = dt_constraint.apply_constraints(
                ranked, phase1, layer_result, tcr_result
            )
            ranked = phase35_result.get("phase35_candidates", ranked)
            logger.info(
                "Phase 3.5: Applied %d constraint rules. Domain=%s (conf=%.2f). Layer-mode=%s",
                phase35_result.get("phase35_adjustments", 0),
                phase35_result.get("phase35_domain", "unknown"),
                phase35_result.get("phase35_domain_confidence", 0),
                phase35_result.get("phase35_layer_mode", False),
            )
        except Exception as e:
            logger.warning("Phase 3.5 decision tree failed: %s", e)

        # ─────────────────────────────
        # PHASE 3.6: Cross-Domain Validation Layer
        # ─────────────────────────────
        phase36_result = {}
        try:
            validator = CrossDomainValidator()
            phase36_result = validator.validate(
                ranked,
                phase1,
                phase35_result,
            )
            ranked = phase36_result.get("phase36_candidates", ranked)
            logger.info(
                "Phase 3.6: Cross-domain validation complete. Verified=%s, adjustments=%d",
                phase36_result.get("phase36_domain_verified", False),
                phase36_result.get("phase36_adjustments", 0),
            )
        except Exception as e:
            logger.warning("Phase 3.6 cross-domain validation failed: %s", e)

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
            phase5_result = resolver.resolve(phase4_result, phase1, all_candidates)

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

        # ─────────────────────────────────────────────────────────────
        # FINAL CONSISTENCY CHECK + Feedback Loop
        # (Integrated into Phase 5/8 — not shown as separate step)
        # ─────────────────────────────────────────────────────────────
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
                    "Final Consistency: coherent=%s",
                    consistency_result.get("coherent", True),
                )
        except Exception as e:
            logger.warning("Final consistency check failed: %s", e)

        # ─────────────────────────────────────────────────────────────
        # FEEDBACK LOOP: If consistency detects issues, constrain scoring
        # ─────────────────────────────────────────────────────────────
        # If Phase 7 detects issues, use recommendations to constrain re-scoring
        # This gives the system a "second chance" to fix semantic drift
        feedback_applied = False
        feedback_constrained_families = []

        try:
            issues = consistency_result.get("issues", [])
            recommended_primary = consistency_result.get("recommended_primary", "")
            recommended_secondary = consistency_result.get("recommended_secondary", [])

            # Check if consistency detected problems
            has_warnings = (
                not consistency_result.get("coherent", True)
                or len(issues) > 0
                or recommended_primary  # LLM recommended a different primary
            )

            if has_warnings and recommended_primary:
                logger.info(
                    "Feedback loop triggered. Issues=%d, Recommended Primary=%s",
                    len(issues),
                    recommended_primary,
                )

                # Build constrained family set from consistency recommendations
                feedback_constrained_families = [
                    recommended_primary[:4]
                ]  # 4-char prefix
                for sec in recommended_secondary:
                    if sec and len(sec) >= 4:
                        feedback_constrained_families.append(sec[:4])

                # Also include any families mentioned in issues
                for issue in issues:
                    issue_str = str(issue).upper()
                    family_patterns = re.findall(r"[A-Z]\d{2}[A-Z]?", issue_str)
                    for fp in family_patterns:
                        if fp not in feedback_constrained_families:
                            feedback_constrained_families.append(fp)

                logger.info(
                    "Feedback loop: Constrained families: %s",
                    feedback_constrained_families,
                )

                # Filter validated_candidates to only include constrained families
                if validated_candidates:
                    original_count = len(validated_candidates)
                    validated_candidates = [
                        c
                        for c in validated_candidates
                        if any(
                            c.get("symbol", "").startswith(f)
                            for f in feedback_constrained_families
                        )
                    ]
                    filtered_count = len(validated_candidates)

                    if filtered_count < original_count:
                        logger.info(
                            "Feedback loop: Filtered candidates %d -> %d",
                            original_count,
                            filtered_count,
                        )

                    # If no candidates remain after filtering, keep top candidates
                    if not validated_candidates:
                        validated_candidates = [
                            c
                            for c in ranked[:5]
                            if not any(
                                c.get("symbol", "").startswith(h)
                                for h in ["H02", "B60Q"]
                            )
                        ]
                        logger.warning(
                            "Feedback loop: No candidates matched feedback filter. "
                            "Using top ranked (excluding hardware)"
                        )

                    feedback_applied = True

        except Exception as e:
            logger.warning("Phase 7.5 feedback loop failed: %s", e)

        # ─────────────────────────────
        # PHASE 8: CPC Role Labeling (3-Layer Explanation Model)
        # ─────────────────────────────
        role_labeling_result = {}
        try:
            # Build enhanced candidates list including Phase 7 recommendations
            enhanced_candidates = list(ranked)  # Start with ranked candidates

            # Add Phase 7 recommended codes if they exist and aren't already present
            recommended_primary = consistency_result.get("recommended_primary", "")
            recommended_secondary = consistency_result.get("recommended_secondary", [])

            # Add primary recommendation
            if recommended_primary:
                if isinstance(recommended_primary, dict):
                    sym = recommended_primary.get("symbol", "")
                else:
                    sym = recommended_primary

                if sym and not any(
                    c.get("symbol", "").startswith(sym[:6])
                    for c in enhanced_candidates[:10]
                ):
                    enhanced_candidates.insert(
                        0,
                        {
                            "symbol": sym,
                            "title": f"Recommended by Phase 7: {recommended_primary.get('title', 'Primary recommendation')}"
                            if isinstance(recommended_primary, dict)
                            else sym,
                            "score": 1.0,
                            "contribution_match": "primary",
                        },
                    )
                    logger.info("Phase 8: Added Phase 7 recommended primary: %s", sym)

            # Add secondary recommendations
            for sec in recommended_secondary:
                if isinstance(sec, dict):
                    sym = sec.get("symbol", "")
                    title = sec.get("title", sym)
                else:
                    sym = sec
                    title = sec

                if sym and not any(
                    c.get("symbol", "").startswith(sym[:6])
                    for c in enhanced_candidates[:10]
                ):
                    enhanced_candidates.insert(
                        1,
                        {
                            "symbol": sym,
                            "title": f"Recommended by Phase 7: {title}",
                            "score": 0.95,
                            "contribution_match": "secondary",
                        },
                    )
                    logger.info("Phase 8: Added Phase 7 recommended secondary: %s", sym)

            labeler = CPCRoleLabeling(llm=self.llm)
            role_labeling_result = labeler.label_roles(
                candidates=enhanced_candidates,
                phase1_data=phase1,
                phase36_result=phase36_result,
                tcr_result=tcr_result,
            )
            logger.info(
                "Phase 8: Role labeling complete. Core=%d, Support=%d, Context=%d, Coverage=%d",
                len(role_labeling_result.get("layer1_core", [])),
                len(role_labeling_result.get("layer2_support", [])),
                len(role_labeling_result.get("layer2_context", [])),
                len(role_labeling_result.get("layer3_coverage", [])),
            )
        except Exception as e:
            logger.warning("Phase 8 role labeling failed: %s", e)

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
                "Multi-layer CPC decomposition: each technical layer maps to CPC independently. "
                "No cross-layer penalties. No forced hierarchy. "
                f"Primary layer={primary_layer}. "
                f"Layer scores={ {k: round(v, 2) for k, v in layer_scores.items()} }. "
                f"Phase 2B expanded to {phase2b_candidate_count} candidates, "
                f"Phase 2C scored down to {phase2c_final_count}."
            ),
            "score_margin": round(score_margin, 6),
            "confidence_level": confidence_level,
            "phase2a_source": phase2a_source,
            "phase2a_reasoning": phase2a_reasoning,
            "phase2a_families": top_cpc_families,
            "phase2a_primary_layer": primary_layer,
            "phase2a_layer_scores": {k: round(v, 4) for k, v in layer_scores.items()},
            "phase2b_candidate_count": phase2b_candidate_count,
            "phase2b_expansion_counts": phase2b_expansion_counts,
            "phase2b_skipped_classes": phase2b_skipped,
            "phase2c_final_count": phase2c_final_count,
            "phase2d_anchor_set": phase2d_result.get("anchor_set", []),
            "phase2d_anchor_source": phase2d_result.get("anchor_source", []),
            "phase2d_kept_count": phase2d_result.get("kept_count", 0),
            "phase2d_discarded_count": phase2d_result.get("discarded_count", 0),
            "phase2d_discard_log": phase2d_result.get("discard_log", []),
            "phase2d_find_until_full": find_until_full_log,
            "phase2c_total_scored": phase2c_final_count,
        }

        # ── Premier: determine main recommendation BEFORE building result ──
        premier_data = None
        if consistency_result.get("coherent") and consistency_result.get(
            "recommended_primary"
        ):
            premier_symbol = consistency_result["recommended_primary"]
            premier_title = ""
            for node in validated_candidates or ranked:
                if node.get("symbol") == premier_symbol:
                    premier_title = node.get("title", "")
                    break
            premier_data = {
                "symbol": premier_symbol,
                "title": premier_title,
                "confidence": "high",
                "reasoning": f"Final consistency check selected this as the recommended primary code. {consistency_result.get('reasoning', '')}",
            }
        elif best_code and best_code.get("symbol"):
            premier_data = {
                "symbol": best_code.get("symbol"),
                "title": best_code.get("title", ""),
                "confidence": best_code.get("confidence", "medium"),
                "reasoning": best_code.get("reasoning", ""),
            }
        elif ranked:
            premier_data = {
                "symbol": ranked[0]["symbol"],
                "title": ranked[0]["title"],
                "confidence": "medium",
                "reasoning": "Top-scoring candidate from Phase 2/3 scoring",
            }

        result = {
            "phase1": phase1,
            "phase15": phase15_result,
            "tcr_analysis": tcr_result,
            "phase2": phase2,
            "phase2a_layers": layer_result,
            "phase3": ranked,
            "phase35": phase35_result,
            "phase36": phase36_result,
            "phase8_role_labeling": role_labeling_result,
            "phase4": phase4_result,
            "cpc": cpc,
            "formatted_report": self._build_formatted_report(
                phase1,
                role_labeling_result,
                consistency_result,
                tcr_result,
                pillars=phase5_result.get("pillars", {}),
                premier=premier_data,
            ),
        }
        if premier_data:
            result["premier"] = premier_data

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
        all_candidates = []
        score_margin = 0.0
        confidence_level = "medium"
        phase2b_candidate_count = 0
        phase2c_final_count = 0
        phase2b_expansion_counts = {}

        try:
            all_subgroups = self.xml_parser.expand_classes(
                combined_classes,
                include_non_allocatable=False,
                allowed_roots=top_cpc_families,
            )
            phase2b_candidate_count = len(all_subgroups)

            # Per-class expansion counts
            for sg in all_subgroups:
                sym = sg.get("symbol", "")
                prefix = sym[:4] if len(sym) >= 4 else sym[:3]
                phase2b_expansion_counts[prefix] = (
                    phase2b_expansion_counts.get(prefix, 0) + 1
                )

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
                    # ═══════════════════════════════════════════════
                    # HYBRID SCORING: TF-IDF + Semantic Similarity
                    # ═══════════════════════════════════════════════
                    sem_scores = {}
                    if self.knowledge_graph and self.knowledge_graph.embeddings:
                        patent_text = (
                            phase1.get("technical_object", "")
                            + " "
                            + phase1.get("core_function", "")
                        )
                        sem_scores = self._compute_semantic_scores(
                            [sg for _, sg, _ in scored], patent_text
                        )

                    if sem_scores:
                        max_tfidf = max(s[0] for s in scored) if scored else 1.0
                        hybrid_scored = []
                        for tfidf_score, sg, matching_terms in scored:
                            sym = sg.get("symbol", "")
                            tfidf_norm = (
                                tfidf_score / max_tfidf if max_tfidf > 0 else 0.0
                            )
                            sem = sem_scores.get(sym, 0.0)
                            hybrid = 0.4 * tfidf_norm + 0.6 * sem
                            hybrid_scored.append((round(hybrid, 6), sg, matching_terms))
                        scored = hybrid_scored
                        scored.sort(key=lambda x: -x[0])

                    denom = scored[0][0]
                    if denom == 0:
                        denom = 1.0

                    # Normalize ALL scored candidates (Find-Until-Full below)
                    all_candidates = []
                    for score, sg, _ in scored:
                        normalized_score = min(score / denom, 1.0)
                        all_candidates.append(
                            {
                                "symbol": sg["symbol"],
                                "title": sg["title"],
                                "level": sg.get("level", 0),
                                "score": round(normalized_score, 6),
                                "full_context": sg.get("full_context", sg["title"]),
                            }
                        )

                    phase2c_final_count = len(all_candidates)
        except Exception as e:
            logger.error("Phase 2B/C error: %s", e)

        # ─────────────────────────────
        # PHASE 2D: Subclass Structural Anchor + Find-Until-Full
        # ─────────────────────────────
        phase2d_result = {}
        find_until_full_log = []
        try:
            anchor_filter = Phase2DSubclassAnchor()
            search_depths = [500, 1000, len(all_candidates) if all_candidates else 0]
            search_depths = [d for d in search_depths if d > 0]

            for depth in search_depths:
                batch = all_candidates[:depth]
                phase2d_result = anchor_filter.filter(
                    candidates=batch,
                    layer_result=phase2a_result,
                    max_result=depth,
                )
                survivors = phase2d_result.get("purified_candidates", [])
                kept = len(survivors)

                entry = {
                    "depth": depth,
                    "survivors_found": kept,
                    "deep_search_triggered": kept < 20,
                }
                find_until_full_log.append(entry)

                if kept >= 20 or depth == search_depths[-1]:
                    if depth == 500 and kept >= 20:
                        logger.info(
                            "Find-Until-Full: Scanned %d to find %d valid technical anchors. ✓ Quota met.",
                            depth,
                            kept,
                        )
                    elif kept >= 20:
                        logger.info(
                            "Deep Search required. Scanned %d to find %d valid technical anchors. ✓ Quota met.",
                            depth,
                            kept,
                        )
                    else:
                        logger.warning(
                            "Find-Until-Full exhausted: Scanned %d to find only %d valid technical anchors. Using all survivors.",
                            depth,
                            kept,
                        )
                    candidates = survivors
                    break
                else:
                    logger.info(
                        "Deep Search required. Scanned %d to find %d valid technical anchors. Expanding to next depth.",
                        depth,
                        kept,
                    )
        except Exception as e:
            logger.warning("Phase 2D anchor filter failed: %s", e)

        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:20]

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
            phase5_result = resolver.resolve(phase4_result, phase1, all_candidates)
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
                "phase2b_expansion_counts": phase2b_expansion_counts,
                "phase2c_final_count": phase2c_final_count,
                "phase2c_total_scored": phase2c_final_count,
                "phase2d_anchor_set": phase2d_result.get("anchor_set", []),
                "phase2d_kept_count": phase2d_result.get("kept_count", 0),
                "phase2d_discarded_count": phase2d_result.get("discarded_count", 0),
                "phase2d_find_until_full": find_until_full_log,
            },
            "phase3": ranked,
            "phase4": phase4_result,
            "phase5": phase5_result,
            "cpc": cpc,
        }

    def _compute_semantic_scores(
        self, candidates: List[Dict], patent_text: str
    ) -> Dict[str, float]:
        """
        Compute semantic similarity scores for candidate CPCs using the KG embeddings.

        Uses all-mpnet-base-v2 embedding model to compute cosine similarity
        between the patent's technical text and each candidate CPC title.
        Returns dict mapping candidate symbol -> normalized similarity [0, 1].
        Falls back to empty dict if KG is unavailable.
        """
        sem_scores: Dict[str, float] = {}
        if not self.knowledge_graph or not self.knowledge_graph.embeddings:
            return sem_scores

        try:
            model = self.knowledge_graph._get_model()
            query_emb = model.encode(
                [patent_text], show_progress_bar=False, convert_to_numpy=True
            )[0]
            query_norm = np.linalg.norm(query_emb)

            if query_norm == 0:
                return sem_scores

            for c in candidates:
                symbol = c.get("symbol", "")
                if symbol not in self.knowledge_graph.embeddings:
                    continue
                cand_emb = self.knowledge_graph.embeddings[symbol]
                cand_norm = np.linalg.norm(cand_emb)
                if cand_norm == 0:
                    continue
                sim = float(np.dot(query_emb, cand_emb) / (query_norm * cand_norm))
                sem_scores[symbol] = max(0.0, sim)

            # Normalize to [0, 1] scale
            if sem_scores:
                max_val = max(sem_scores.values())
                if max_val > 0:
                    for sym in sem_scores:
                        sem_scores[sym] /= max_val

            logger.info(
                "Semantic scores computed for %d/%d candidates (KG=%s)",
                len(sem_scores),
                len(candidates),
                "loaded" if self.knowledge_graph else "unavailable",
            )
        except Exception as e:
            logger.warning("Semantic scoring failed, falling back to TF-IDF: %s", e)

        return sem_scores

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
        """Score candidates using TF-IDF with bigram support + hybrid semantic blending."""
        import math
        from collections import Counter

        scored = []

        # Build document frequency (unigrams + bigrams)
        doc_freq = Counter()
        title_count = len(all_subgroups)
        all_context_bigrams = []

        for sg in all_subgroups:
            context = sg.get("full_context", sg["title"]).lower()
            unigrams, bigrams = _tokenize_with_bigrams(context)
            all_context_bigrams.append(bigrams)
            for token in unigrams:
                doc_freq[token] += 1
            for bigram in bigrams:
                doc_freq[bigram] += 1

        # Score each subgroup
        for idx, sg in enumerate(all_subgroups):
            score = 0.0
            matching_terms = 0
            context = sg.get("full_context", sg["title"]).lower()
            context_tokens = set(re.findall(r"[a-zA-Z]+", context))
            context_tokens = {_normalize_word(w) for w in context_tokens if len(w) > 2}
            context_bigrams = all_context_bigrams[idx]
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
                term_tokens = {
                    _normalize_word(t) for t in term.lower().split() if len(t) > 2
                }

                # Multi-word phrase match
                if len(term.split()) >= 2 and term in context:
                    avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                        term_tokens
                    )
                    idf = math.log(title_count / max(avg_df, 1))
                    term_score += idf * importance_weight * 8
                    matching_terms += 1

                # Bigram matching (NEW: captures technical phrases)
                term_bigrams = _make_term_bigrams(term)
                if term_bigrams and context_bigrams:
                    for tb in term_bigrams:
                        if tb in context_bigrams:
                            avg_df = doc_freq.get(tb, 1)
                            idf = math.log(title_count / max(avg_df, 1))
                            term_score += idf * (importance / 5.0) * 6
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

    def _build_formatted_report(
        self,
        phase1: Dict[str, Any],
        role_labeling_result: Dict[str, Any],
        consistency_result: Dict[str, Any],
        tcr_result: Optional[Dict[str, Any]] = None,
        pillars: Optional[Dict[str, Any]] = None,
        premier: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build an Executive Patent Classification Report in Markdown.

        Format:
        # Executive Patent Classification Report
        ## Main Recommendation (Premier Code + Confidence)
        ## Technical Breakdown (Facets table)
        ## Professional Justification
        ## Suggested Indexing Codes
        ## Supporting Details (Core / Support / Context / Coverage)
        """
        lines = []
        coherent = consistency_result.get("coherent", True)
        premier_symbol = premier.get("symbol", "") if premier else ""
        premier_title = premier.get("title", "") if premier else ""

        # ═══════════════════════════════════════════════════════════
        # HEADER & HERO
        # ═══════════════════════════════════════════════════════════
        lines.append("# 📄 Executive Patent Classification Report")
        lines.append("")
        if premier_symbol:
            status = (
                "✅ Validated via Cross-Domain Consistency Check"
                if coherent
                else "⚠️ Requires Review"
            )
            lines.append(
                f"## Main Recommendation: `{premier_symbol}` — {premier_title}"
            )
            lines.append("")
            lines.append(f"**Status:** {status}")
        else:
            lines.append("## Main Recommendation: Not Available")
        lines.append("")

        # ═══════════════════════════════════════════════════════════
        # TECH STACK TABLE
        # ═══════════════════════════════════════════════════════════
        goal = {}
        method = {}
        context = {}
        if pillars:
            lines.append("## 🛠 Technical Breakdown")
            lines.append("")
            lines.append("| Role | CPC Code | Description |")
            lines.append("|------|----------|-------------|")
            goal = pillars.get("pillar1_goal", {})
            method = pillars.get("pillar2_method", {})
            context = pillars.get("pillar3_context", {})
            if goal and goal.get("symbol"):
                lines.append(
                    f"| **Primary Goal** | `{goal['symbol']}` | {self._shorten(goal.get('title', ''), 80)} |"
                )
            if method and method.get("symbol"):
                lines.append(
                    f"| **AI Methodology** | `{method['symbol']}` | {self._shorten(method.get('title', ''), 80)} |"
                )
            if context and context.get("symbol"):
                lines.append(
                    f"| **Domain Context** | `{context['symbol']}` | {self._shorten(context.get('title', ''), 80)} |"
                )
            lines.append("")

        # ═══════════════════════════════════════════════════════════
        # PROFESSIONAL JUSTIFICATION
        # ═══════════════════════════════════════════════════════════
        lines.append("## 💡 Professional Justification")
        lines.append("")
        llm_summary = role_labeling_result.get("phase85_executive_summary", "")
        if llm_summary:
            lines.append(llm_summary)
        else:
            # Fallback justification from facets
            parts = []
            if premier_symbol:
                parts.append(
                    f"The invention is primarily classified in `{premier_symbol}` as it represents the core technical contribution."
                )
            if method and method.get("symbol"):
                parts.append(
                    f"The use of `{method['symbol']}` reflects the AI/ML implementation strategy."
                )
            if context and context.get("symbol"):
                parts.append(
                    f"The inclusion of `{context['symbol']}` ensures the patent is protected within its specific industrial application domain."
                )
            if parts:
                lines.append(" ".join(parts))
            else:
                lines.append(
                    "The classification reflects the primary technical contribution of the disclosed invention "
                    "based on semantic analysis of the patent text, technical weight analysis, and cross-domain validation."
                )
        lines.append("")

        # ═══════════════════════════════════════════════════════════
        # SUGGESTED INDEXING CODES
        # ═══════════════════════════════════════════════════════════
        core = role_labeling_result.get("layer1_core", [])
        support = role_labeling_result.get("layer2_support", [])
        context_codes = role_labeling_result.get("layer2_context", [])
        coverage = role_labeling_result.get("layer3_coverage", [])

        all_listed = set()
        for c in [core, support, context_codes, coverage]:
            for item in c:
                all_listed.add(item.get("symbol", ""))

        # Collect pillars symbols already shown
        pillar_symbols = set()
        if pillars:
            for k, v in pillars.items():
                if v.get("symbol"):
                    pillar_symbols.add(v["symbol"])

        suggested = []
        for c in [core, support, context_codes, coverage]:
            for item in c:
                sym = item.get("symbol", "")
                if sym not in pillar_symbols and sym != premier_symbol:
                    title = item.get("title", "")
                    if title:
                        suggested.append(f"`{sym}` — {self._shorten(title, 80)}")

        if suggested:
            lines.append("## 📋 Suggested Indexing Codes")
            lines.append("")
            for s in suggested[:10]:
                lines.append(f"- {s}")
            lines.append("")
            lines.append(
                "_Copy these codes into your patent application as supplementary indexing references._"
            )
        lines.append("")

        # ═══════════════════════════════════════════════════════════
        # SUPPORTING DETAILS (condensed)
        # ═══════════════════════════════════════════════════════════
        if core or support or context_codes:
            lines.append("## 📊 Supporting Classification Details")
            lines.append("")
            if core:
                lines.append("**Core Invention:**")
                for c in core:
                    lines.append(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
                lines.append("")
            if support:
                lines.append("**Enabling Technology:**")
                for c in support:
                    lines.append(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
                lines.append("")
            if context_codes:
                lines.append("**Application Context:**")
                for c in context_codes:
                    lines.append(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _shorten(text: str, max_len: int = 80) -> str:
        """Truncate text to max_len chars, adding ellipsis if needed."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
