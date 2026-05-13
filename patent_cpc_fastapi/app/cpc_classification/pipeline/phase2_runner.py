import logging
import os
import re
import math
from collections import Counter

from ..scoring.tfidf_scorer import tokenize_with_bigrams, make_term_bigrams
from ..scoring.semantic_scorer import compute_semantic_scores
from ..scoring.domain_booster import get_synonyms, get_expanded_terms
from ..utils.xml_utils import resolve_xml_dir
from ..utils.text_utils import normalize_word
from ..cpc_phase2d_anchor import Phase2DSubclassAnchor
from ..cpc_layer_decomposer import merge_layers_to_family_list

logger = logging.getLogger(__name__)


def _tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {normalize_word(w) for w in words if len(w) > 2}


def run_phase2(classifier, phase1, phase15_result=None, tcr_result=None):
    if phase15_result is None:
        phase15_result = {}
    if tcr_result is None:
        tcr_result = {}

    terms = phase1.get("terms", phase1.get("essential_terms", phase1.get("terms", [])))
    term_importance = {}
    for t in terms:
        if isinstance(t, dict):
            term = t.get("term", "").lower()
            importance = t.get("importance", 5)
            source = t.get("source_section", "unknown")
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
    # PHASE 2A: CPC Layer Decomposition
    # ─────────────────────────────
    phase2a_result = classifier.layer_decomposer.decompose(
        phase1, phase15_result, tcr_result
    )

    layer_result = phase2a_result
    layers = layer_result.get("layers", {})
    primary_layer = layer_result.get("primary_layer", "application")
    layer_scores = layer_result.get("layer_scores", {})
    layer_reasoning = layer_result.get("reasoning", "")

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

    if len(top_cpc_families) < 2:
        from ..cpc_family_router import FALLBACK_FAMILIES

        for fb in FALLBACK_FAMILIES:
            if fb not in top_cpc_families:
                top_cpc_families.append(fb)
            if len(top_cpc_families) >= 2:
                break
        phase2a_reasoning += " Fallback families added."
        logger.info("Phase 2A: Fallback applied, families=%s", top_cpc_families)

    # ─────────────────────────────
    # PHASE 2B: Restricted XML Expansion
    # ─────────────────────────────
    graph_classes = []
    if classifier.knowledge_graph and classifier.knowledge_graph.embeddings:
        logger.info(
            "Phase 2B: Querying knowledge graph within families %s",
            top_cpc_families,
        )
        try:
            patent_text_for_search = f"{phase1.get('technical_object', '')} {phase1.get('core_function', '')} {system_context}"

            if not classifier.knowledge_graph.bm25_index:
                logger.info("Initializing hybrid retrieval (BM25 + cross-encoder)")
                classifier.knowledge_graph.init_hybrid_retrieval()

            if classifier.knowledge_graph.bm25_index:
                logger.info("Using hybrid search (BM25 + embedding + cross-encoder)")
                graph_results = classifier.knowledge_graph.hybrid_search(
                    text=patent_text_for_search,
                    top_k=20,
                    bm25_k=200,
                    use_cross_encoder=True,
                )
            else:
                logger.info("Using legacy semantic search (no BM25)")
                extracted_terms = list(term_importance.keys())[:15]
                graph_results = classifier.knowledge_graph.find_initial_classes(
                    patent_text=patent_text_for_search,
                    extracted_terms=extracted_terms,
                    top_k=10,
                )

            graph_classes = [
                cls
                for cls, score in graph_results
                if score > 0.3 and any(cls.startswith(fam) for fam in top_cpc_families)
            ]
            logger.info(
                "Phase 2B: Graph suggested classes (filtered): %s",
                graph_classes,
            )
        except Exception as e:
            logger.warning("Phase 2B graph query failed: %s", e)

    combined_classes = []
    seen = set()
    for cls in graph_classes:
        if cls not in seen:
            seen.add(cls)
            combined_classes.append(cls)

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
        combined_classes = allowed_roots

    xml_dir = resolve_xml_dir()
    available_files = set(os.listdir(xml_dir))
    valid_combined = []
    phase2b_skipped = []
    for cls in combined_classes:
        xml_name = f"cpc-scheme-{cls}.xml"
        if xml_name in available_files:
            valid_combined.append(cls)
        else:
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
    # PHASE 2C: XML Expansion + Scoring
    # ─────────────────────────────
    candidates = []
    all_candidates = []
    score_margin = 0.0
    confidence_level = "medium"
    phase2b_candidate_count = 0
    phase2c_final_count = 0
    phase2b_expansion_counts = {}

    try:
        all_subgroups = classifier.xml_parser.expand_classes(
            combined_classes,
            include_non_allocatable=False,
            allowed_roots=allowed_roots,
        )
        phase2b_candidate_count = len(all_subgroups)

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
            all_bigrams = []

            for sg in all_subgroups:
                context = sg.get("full_context", sg.get("title", "")).lower()
                unigrams, bigrams = tokenize_with_bigrams(context)
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
                context_bigrams = all_bigrams[idx]
                score = 0.0
                matching_terms = 0

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

                term_phrases = []
                for term in term_importance.keys():
                    words = term.split()
                    if len(words) >= 2:
                        for i in range(len(words) - 1):
                            term_phrases.append(" ".join(words[i : i + 2]))
                        for i in range(len(words) - 2):
                            term_phrases.append(" ".join(words[i : i + 3]))

                for term, importance in term_importance.items():
                    term_score = 0.0
                    term_tokens = _tokenize(term)
                    if not term_tokens:
                        continue

                    if len(term.split()) >= 2 and term in context:
                        avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                            term_tokens
                        )
                        idf = math.log(title_count / max(avg_df, 1))
                        importance_weight = importance / 5.0
                        term_score += idf * importance_weight * 8
                        matching_terms += 1

                    term_bigrams = make_term_bigrams(term)
                    if term_bigrams and context_bigrams:
                        for tb in term_bigrams:
                            if tb in context_bigrams:
                                avg_df = doc_freq.get(tb, 1)
                                idf = math.log(title_count / max(avg_df, 1))
                                term_score += idf * (importance / 5.0) * 6
                                matching_terms += 1

                    overlap = term_tokens & context_tokens
                    if overlap:
                        overlap_idf = sum(
                            math.log(title_count / max(doc_freq.get(t, 1), 1))
                            for t in overlap
                        )
                        term_score += overlap_idf * (importance / 5.0) * 3

                    if len(term.split()) == 1 and term in context:
                        avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(
                            term_tokens
                        )
                        idf = math.log(title_count / max(avg_df, 1))
                        importance_weight = importance / 5.0
                        base_score = idf * importance_weight * 5

                        false_friend_penalty = 0.0
                        if term in [
                            "exchange",
                            "response",
                            "system",
                            "user",
                            "transfer",
                        ]:
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
                                false_friend_penalty = base_score * 0.7

                        term_score += base_score - false_friend_penalty
                        matching_terms += 1

                    synonyms = get_expanded_terms(term)
                    for syn in synonyms:
                        if syn != term and syn in context:
                            syn_tokens = _tokenize(syn)
                            avg_df = sum(doc_freq.get(t, 1) for t in syn_tokens) / max(
                                len(syn_tokens), 1
                            )
                            idf = math.log(title_count / max(avg_df, 1))
                            term_score += idf * (importance / 5.0) * 4

                    score += term_score

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

                all_terms_text = " ".join(term_importance.keys())

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
                has_image_signal = any(sig in all_terms_text for sig in image_signals)

                family3 = symbol[:3] if len(symbol) >= 3 else symbol
                if has_nn_subject and not has_image_signal:
                    if family3 in ["G06T", "G10K"]:
                        score *= 0.2
                        logger.debug(
                            "Cross-domain guardrail: NN subject penalizes %s",
                            family3,
                        )

                if has_nn_subject and family3 == "G06N":
                    score *= 1.5
                    logger.debug(
                        "Model optimization boost: G06N boosted for NN subject"
                    )

                if "clipping" in all_terms_text:
                    if "weight" in all_terms_text or "parameter" in all_terms_text:
                        if family3 == "G06N":
                            score *= 1.4
                        elif family3 == "G06T":
                            score *= 0.3
                    elif any(sig in all_terms_text for sig in image_signals):
                        if family3 == "G06T":
                            score *= 1.3

                base_multiplier = 1.2
                class_prefix = symbol[:4] if len(symbol) >= 4 else symbol
                family3 = symbol[:3] if len(symbol) >= 3 else symbol
                phase2a_scores = phase2a_result.get("scores", {})
                if class_prefix in phase2a_scores:
                    base_multiplier = 1.0 + (phase2a_scores[class_prefix] * 1.0)
                elif family3 in phase2a_scores:
                    base_multiplier = 1.0 + (phase2a_scores[family3] * 1.0)
                score *= base_multiplier

                symbol_depth = symbol.count("/") + sum(
                    symbol.count(d) for d in "0123456789"
                )
                if matching_terms >= 2:
                    depth_bonus = min(symbol_depth * 0.5, 3.0)
                    score += depth_bonus

                if score > 0:
                    scored.append((score, sg, matching_terms))

            scored.sort(key=lambda x: -x[0])

            # Hybrid scoring: TF-IDF + Semantic Similarity
            sem_scores = {}
            if classifier.knowledge_graph and classifier.knowledge_graph.embeddings:
                patent_text = (
                    phase1.get("technical_object", "")
                    + " "
                    + phase1.get("core_function", "")
                )
                sem_scores = compute_semantic_scores(
                    [sg for _, sg, _ in scored], patent_text, classifier.knowledge_graph
                )

            if sem_scores:
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

            all_candidates = []
            if scored:
                scores = [s[0] for s in scored]
                max_score = max(scores)
                median_score = (
                    sorted(scores)[len(scores) // 2] if len(scores) > 1 else max_score
                )
                denom = max_score + median_score * 0.5

                if len(scores) >= 2:
                    score_margin = round((scores[0] - scores[1]) / denom, 6)
                    if score_margin > 0.3:
                        confidence_level = "high"
                    elif score_margin < 0.1:
                        confidence_level = "low"
                    else:
                        confidence_level = "medium"

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
    phase2d_result = {}
    find_until_full_log = []
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

    phase2_dict = {
        "codes": [node["symbol"] for node in candidates] if candidates else [],
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

    return candidates, all_candidates, phase2_dict, phase2d_result, find_until_full_log
