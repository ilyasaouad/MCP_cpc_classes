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
from .prompts.shared import label_claims
from .prompts.prompt_phases_5_6 import (
    validation_prompt_single,
    reconciliation_prompt,
)
from .prompts.prompt_phases_35_4_7 import (
    phase35_tiebreaker_prompt,
    phase4_sanity_check_prompt,
    phase7_consistency_prompt as consistency_check_prompt,
)
from .prompts.prompt_phase1 import score_phase1_completeness

# ── Pipeline runners ──
from .pipeline.phase1_runner import run_phase1
from .pipeline.phase15_tcr_runner import run_phase15_tcr
from .pipeline.phase2_runner import run_phase2
from .pipeline.phase3_runner import run_phase3
from .pipeline.phase4_5_runner import run_phase4_5
from .pipeline.phase7_runner import run_phase7
from .pipeline.phase8_runner import run_phase8

# ── Term importance builder ──
from .scoring.tfidf_scorer import (
    tokenize_with_bigrams,
    make_term_bigrams,
    score_candidates,
)
from .scoring.domain_booster import (
    get_synonyms,
    get_expanded_terms as get_synonym_expansions,
)
from .utils.text_utils import normalize_word
from .utils.xml_utils import resolve_xml_dir as _resolve_xml_dir_static

logger = logging.getLogger(__name__)


def _build_term_importance(phase1: Dict[str, Any]) -> Dict[str, int]:
    """Build weighted term importance dict from Phase 1 output."""
    term_importance: Dict[str, int] = {}
    terms = phase1.get("terms", phase1.get("essential_terms", []))
    for t in terms:
        if isinstance(t, dict) and "term" in t:
            term_importance[t["term"]] = t.get("importance", 5)
    if not term_importance:
        for t in phase1.get("essential_terms", []):
            if isinstance(t, dict) and "term" in t:
                term_importance[t["term"]] = t.get("importance", 5)
    return term_importance


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
        # Parse input
        # ─────────────────────────────
        description, labeled_claims = self._parse_input(text, claims)

        # ─────────────────────────────
        # PHASE 1: Extraction
        # ─────────────────────────────
        phase1 = run_phase1(self, description, labeled_claims)
        if "error" in phase1:
            return phase1

        # ─────────────────────────────
        # PHASE 1.5 + TCR
        # ─────────────────────────────
        phase15_result, tcr_result = run_phase15_tcr(self, phase1)

        # ─────────────────────────────
        # PHASE 2: Layer decomposition → XML expansion → Scoring → Anchor filter
        # ─────────────────────────────
        candidates, all_candidates, phase2_dict, phase2d_result, find_until_full_log = (
            run_phase2(self, phase1, phase15_result, tcr_result)
        )

        # ─────────────────────────────
        # PHASE 3: Ranking + 3.5 Decision Tree + 3.6 Cross-Domain Validation
        # ─────────────────────────────
        layer_result = phase2_dict.get("phase2a_layers", {})
        ranked, phase35_result, phase36_result = run_phase3(
            self, candidates, phase1, layer_result, tcr_result
        )

        # ─────────────────────────────
        # PHASE 4 + 5: Consolidation + Resolution + Tri-Pillar
        # ─────────────────────────────
        term_importance = _build_term_importance(phase1)
        phase4_result, phase5_result = run_phase4_5(
            self, ranked, phase1, all_candidates, term_importance
        )

        # ─────────────────────────────
        # PHASE 7: Consistency check + Feedback loop + Premier
        # ─────────────────────────────
        (
            consistency_result,
            validated_candidates,
            filtered_out,
            best_code,
            premier_data,
            feedback_data,
        ) = run_phase7(
            self,
            phase1,
            phase5_result,
            ranked,
            candidates,
            phase4_result,
            tcr_result,
            term_importance,
        )

        # ─────────────────────────────
        # PHASE 8: Role Labeling + Report
        # ─────────────────────────────
        role_labeling_result, formatted_report = run_phase8(
            self,
            phase1,
            phase5_result,
            ranked,
            validated_candidates,
            consistency_result,
            tcr_result,
            phase36_result,
            premier_data,
        )

        # ─────────────────────────────
        # Assemble result
        # ─────────────────────────────
        result = {
            "phase1": phase1,
            "phase15": phase15_result,
            "tcr_analysis": tcr_result,
            "phase2": phase2_dict,
            "phase2a_layers": layer_result,
            "phase3": ranked,
            "phase35": phase35_result,
            "phase36": phase36_result,
            "phase8_role_labeling": role_labeling_result,
            "phase4": phase4_result,
            "formatted_report": formatted_report,
            "cpc": ranked,
        }
        if phase5_result:
            result["phase5"] = phase5_result
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
    def _resolve_premier(
        consistency_result: Dict[str, Any],
        best_code: Optional[Dict[str, Any]],
        ranked: List[Dict[str, Any]],
        validated_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve the premier (best) CPC code from available sources.

        Priority:
          1. Consistency check recommendation (if coherent)
          2. Phase 5 best_code (new deterministic resolver)
          3. Top-ranked candidate from Phase 2/3 scoring
        """
        if consistency_result.get("coherent") and consistency_result.get(
            "recommended_primary"
        ):
            symbol = consistency_result["recommended_primary"]
            title = ""
            for node in (validated_candidates or []) + (ranked or []):
                if node.get("symbol") == symbol:
                    title = node.get("title", "")
                    break
            return {
                "symbol": symbol,
                "title": title,
                "confidence": "high",
                "reasoning": (
                    "Final consistency check selected this as the recommended primary code. "
                    + consistency_result.get("reasoning", "")
                ),
            }
        if best_code and best_code.get("symbol"):
            return {
                "symbol": best_code.get("symbol"),
                "title": best_code.get("title", ""),
                "confidence": best_code.get("confidence", "medium"),
                "reasoning": best_code.get("reasoning", ""),
            }
        if ranked:
            return {
                "symbol": ranked[0]["symbol"],
                "title": ranked[0]["title"],
                "confidence": "medium",
                "reasoning": "Top-scoring candidate from Phase 2/3 scoring",
            }
        return None

    @staticmethod
    def _shorten(text: str, max_len: int = 80) -> str:
        """Truncate text to max_len chars, adding ellipsis if needed."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
