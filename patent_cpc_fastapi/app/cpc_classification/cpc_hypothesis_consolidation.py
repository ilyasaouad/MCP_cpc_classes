"""
cpc_hypothesis_consolidation.py - Phase 4: CPC Hypothesis Consolidation

Transforms Phase 3 ranked candidates into structured CPC hypotheses.

KEY DESIGN DECISIONS:
- Only PRIMARY and SECONDARY roles (no tertiary to avoid confusion)
- Scores are normalized (mean-based, not raw sum)
- Weak clusters below threshold are discarded
- Max 2 clusters output
- Distinguishes "core invention" from "supporting infrastructure"

Constraints:
- MUST NOT generate new CPC codes
- MUST only reorganize Phase 3 input
- MUST identify exactly one PRIMARY family
"""

import logging
import math
from typing import Dict, List, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


def normalize_to_family(symbol: str) -> str:
    """Normalize CPC code to family level (first 4 characters)."""
    return symbol[:4] if len(symbol) >= 4 else symbol


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def lexical_similarity(title1: str, title2: str) -> float:
    """Compute lexical similarity between two CPC titles."""
    t1 = title1.lower().replace(",", " ").replace(";", " ").replace("/", " ")
    t2 = title2.lower().replace(",", " ").replace(";", " ").replace("/", " ")
    return jaccard_similarity(t1, t2)


class CPCCluster:
    """Represents a cluster of semantically related CPC candidates."""

    def __init__(self, family: str):
        self.family = family
        self.candidates: List[Dict[str, Any]] = []
        self.raw_score: float = 0.0
        self.normalized_score: float = 0.0
        self.coherence: float = 1.0
        self.term_density: float = 1.0

    def add_candidate(self, candidate: Dict[str, Any]):
        self.candidates.append(candidate)

    def compute_metrics(self, term_importance: Dict[str, float]) -> None:
        """Compute all cluster metrics."""
        self.coherence = self._compute_coherence()
        self.term_density = self._compute_term_density(term_importance)

        # Compute scores
        scores = [c.get("score", 0) for c in self.candidates]
        mean_score = sum(scores) / len(scores) if scores else 0
        size_factor = math.log(len(self.candidates) + 1)  # log(size) bonus

        # Raw sum (for reference)
        self.raw_score = sum(scores)

        # Normalized score: mean × coherence × log(size)
        self.normalized_score = mean_score * self.coherence * size_factor

    def _compute_coherence(self) -> float:
        """Compute intra-cluster coherence (0.5 - 1.2 range)."""
        if len(self.candidates) <= 1:
            return 1.0

        similarities = []
        for i in range(len(self.candidates)):
            for j in range(i + 1, len(self.candidates)):
                sim = lexical_similarity(
                    self.candidates[i].get("title", ""),
                    self.candidates[j].get("title", ""),
                )
                similarities.append(sim)

        if not similarities:
            return 1.0

        avg_sim = sum(similarities) / len(similarities)
        # Map 0-1 similarity to 0.5-1.2 coherence
        return 0.5 + (avg_sim * 0.7)

    def _compute_term_density(self, term_importance: Dict[str, float]) -> float:
        """Compute term density bonus (1.0 - 2.0 range)."""
        if not term_importance or not self.candidates:
            return 1.0

        matched_terms = set()
        for candidate in self.candidates:
            title = candidate.get("title", "").lower()
            for term in term_importance.keys():
                if term.lower() in title:
                    matched_terms.add(term.lower())

        # 1.0 + 0.08 per matched term, capped at 2.0
        bonus = 1.0 + (len(matched_terms) * 0.08)
        return min(bonus, 2.0)

    def get_supporting_codes(self) -> List[str]:
        """Get supporting CPC codes sorted by score descending."""
        sorted_candidates = sorted(
            self.candidates, key=lambda x: x.get("score", 0), reverse=True
        )
        return [c["symbol"] for c in sorted_candidates]

    def get_mean_score(self) -> float:
        """Get mean candidate score."""
        scores = [c.get("score", 0) for c in self.candidates]
        return sum(scores) / len(scores) if scores else 0

    def get_reasoning(self, is_primary: bool = False) -> str:
        """Generate cluster reasoning with proper interpretation."""
        count = len(self.candidates)
        mean = self.get_mean_score()

        # Extract keywords
        all_words = []
        for c in self.candidates:
            title = c.get("title", "").lower()
            all_words.extend(title.split())

        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "for",
            "and",
            "or",
            "in",
            "on",
            "to",
            "with",
            "by",
            "from",
            "as",
            "is",
            "are",
            "be",
            "being",
            "been",
            "methods",
            "systems",
            "apparatus",
            "using",
            "based",
            "method",
            "device",
        }
        keywords = [w for w in all_words if len(w) > 3 and w not in stopwords]
        top_keywords = sorted(
            set(keywords), key=lambda x: keywords.count(x), reverse=True
        )[:4]

        if count == 1:
            return f"Single candidate: {self.family} ({mean:.2f} avg)."

        # Determine if this is core invention or supporting infrastructure
        if is_primary:
            role_desc = "Core invention domain"
        else:
            role_desc = "Supporting technical infrastructure"

        return (
            f"{role_desc}: {count} candidates in {self.family} "
            f"(mean={mean:.2f}, coherence={self.coherence:.2f}). "
            f"Themes: {', '.join(top_keywords)}."
        )


class CPCHypothesisConsolidator:
    """
    Phase 4: Consolidates Phase 3 candidates into max 2 hypotheses.

    Rules:
    - Only PRIMARY and SECONDARY roles (no tertiary)
    - Max 2 hypotheses output
    - Weak clusters (score < threshold) are discarded
    - Scores are normalized (mean-based)
    """

    def __init__(
        self,
        max_hypotheses: int = 2,
        max_candidates: int = 20,
        discard_threshold: float = 0.3,
    ):
        self.max_hypotheses = max_hypotheses
        self.max_candidates = max_candidates
        self.discard_threshold = discard_threshold

    def consolidate(
        self,
        ranked_candidates: List[Dict[str, Any]],
        term_importance: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Consolidate Phase 3 candidates into Phase 4 hypotheses.

        Returns:
            Dict with hypotheses, primary family, confidence, centroid info
        """
        candidates = ranked_candidates[: self.max_candidates]
        logger.info("Phase 4: Consolidating %d candidates", len(candidates))

        if not candidates:
            return self._empty_result()

        # Cluster by family
        clusters = self._cluster_candidates(candidates)
        logger.info("Phase 4: Formed %d raw clusters", len(clusters))

        if not clusters:
            return self._empty_result()

        # Compute metrics
        for cluster in clusters:
            cluster.compute_metrics(term_importance)

        # Rank by normalized score
        ranked_clusters = sorted(
            clusters, key=lambda c: c.normalized_score, reverse=True
        )

        # Discard weak clusters
        strong_clusters = [
            c for c in ranked_clusters if c.normalized_score >= self.discard_threshold
        ]
        discarded = len(ranked_clusters) - len(strong_clusters)

        if discarded > 0:
            logger.info("Phase 4: Discarded %d weak clusters", discarded)

        if not strong_clusters:
            return self._empty_result()

        # Build hypotheses (max 2)
        hypotheses = self._build_hypotheses(strong_clusters)

        # Compute centroid and confidence
        primary = hypotheses[0] if hypotheses else None
        centroid_family = primary["family"] if primary else ""

        # Support weight = primary normalized score / sum of all normalized scores
        total_score = sum(c.normalized_score for c in strong_clusters)
        support_weight = (
            primary["normalized_score"] / total_score
            if total_score > 0 and primary
            else 0
        )

        confidence = self._compute_confidence(strong_clusters)

        result = {
            "phase4_hypotheses": hypotheses,
            "phase4_primary_family": centroid_family,
            "phase4_centroid_family": centroid_family,
            "phase4_support_weight": round(support_weight, 4),
            "phase4_confidence": confidence,
            "phase4_cluster_count": len(clusters),
            "phase4_strong_clusters": len(strong_clusters),
            "phase4_discarded_clusters": discarded,
            "phase4_reasoning": self._build_overall_reasoning(
                hypotheses, strong_clusters, support_weight
            ),
            "phase4_interpretation": self._generate_interpretation(
                hypotheses, support_weight, strong_clusters
            ),
        }

        logger.info(
            "Phase 4: Primary=%s, support_weight=%.2f, confidence=%s, "
            "hypotheses=%d, discarded=%d",
            centroid_family,
            support_weight,
            confidence,
            len(hypotheses),
            discarded,
        )

        return result

    def _generate_interpretation(
        self,
        hypotheses: List[Dict[str, Any]],
        support_weight: float,
        strong_clusters: List[CPCCluster],
    ) -> Dict[str, Any]:
        """
        Generate a human-readable interpretation of Phase 4 metrics.

        Analyzes Support Weight and Coherence to produce:
        - A status label (clean / messy / weak)
        - Descriptive insight text for each metric
        - Actionable advice for proceeding or debugging
        """
        primary = hypotheses[0] if hypotheses else {}
        primary_coherence = primary.get("coherence", 0.0)
        primary_family = primary.get("family", "unknown")

        # ── Support Weight interpretation ──
        if support_weight > 0.5:
            support_status = "clean"
            support_text = (
                f"**High Support Weight detected ({support_weight:.1%}).** "
                "This indicates a 'Clean' patent. The system is confident that "
                f"the invention resides in a single core technical family ({primary_family}), "
                "successfully filtering out unrelated noise like Business Methods (G06Q) "
                "or generic applications."
            )
        elif support_weight < 0.15:
            support_status = "messy"
            support_text = (
                f"**Low Support Weight / High Dispersion ({support_weight:.1%}).** "
                "The patent appears 'Messy' or 'Hybrid.' The search results are "
                "scattered across multiple unrelated domains, which may suggest the "
                "technical core is not yet clearly defined or is heavily polluted by "
                "application-layer keywords."
            )
        else:
            support_status = "moderate"
            support_text = (
                f"**Moderate Support Weight ({support_weight:.1%}).** "
                f"The primary family ({primary_family}) has a reasonable share of "
                "the candidate pool, but meaningful contributions exist in secondary "
                "families. The patent may span multiple technical domains."
            )

        # ── Coherence interpretation ──
        # Coherence range is 0.5–1.2 (mapped from avg lexical similarity)
        if primary_coherence > 0.8:
            coherence_status = "high"
            coherence_text = (
                f"**High Coherence ({primary_coherence:.2f}).** "
                "The candidates in this cluster are 'Family Neighbors.' They don't just "
                "share keywords; they reinforce each other's technical context, proving "
                "that the system has found a stable technical 'niche'."
            )
        elif primary_coherence < 0.6:
            coherence_status = "low"
            if support_weight > 0.8:
                # High support + low coherence = multi-aspect patent (normal, not a problem).
                # E.g. G10L15 (recognition) and G10L13 (synthesis) share the family but
                # use different sub-branch vocabulary, naturally lowering Jaccard similarity.
                coherence_text = (
                    f"**Sub-Branch Spread ({primary_coherence:.2f}).** "
                    "The candidates span multiple technical aspects within the same family "
                    f"(e.g. different sub-branches of {primary_family}). "
                    "This is expected for patents covering both the method and its application. "
                    "Support weight is strong — classification is reliable."
                )
            else:
                coherence_text = (
                    f"**Low Coherence ({primary_coherence:.2f}).** "
                    "While these codes are in the same family, they are functionally distant. "
                    "The candidate pool may be mixing unrelated technical aspects — "
                    "consider reviewing Phase 2B expansion breadth."
                )
        else:
            coherence_status = "moderate"
            coherence_text = (
                f"**Moderate Coherence ({primary_coherence:.2f}).** "
                "Candidates share some technical overlap but are not tightly clustered. "
                "The classification is plausible but could benefit from refinement."
            )

        # ── Actionable Advice ──
        if support_status == "clean" and coherence_status == "high":
            actionable = (
                "**Recommendation:** The classification is stable. "
                "Proceed to Phase 5 (Technical Justification)."
            )
        elif support_status == "messy" and coherence_status == "low":
            actionable = (
                "**Recommendation:** Increase Phase 2C retrieval depth "
                "or refine the technical anchors in Phase 2D. "
                "Consider providing more specific patent claims text."
            )
        elif support_status == "clean" and coherence_status != "high":
            actionable = (
                "**Recommendation:** Support weight is strong but coherence is "
                f"{coherence_status}. This is normal when the patent covers multiple "
                "technical aspects within the same family (e.g. recognition + synthesis "
                "both in G10L). Proceed to Phase 5 — classification is reliable."
            )
        elif support_status != "clean" and coherence_status == "high":
            actionable = (
                "**Recommendation:** Coherence is high but support is spread thin. "
                "The primary family is well-defined but doesn't dominate the pool. "
                "Refine Phase 2D anchors to reduce secondary-family noise."
            )
        else:
            actionable = (
                "**Recommendation:** Both metrics are moderate. "
                "The classification is usable but not definitive. "
                "Proceed to Phase 5 with caution."
            )

        return {
            "support_status": support_status,
            "support_text": support_text,
            "coherence_status": coherence_status,
            "coherence_text": coherence_text,
            "actionable_advice": actionable,
            "primary_coherence": round(primary_coherence, 4),
            "primary_family": primary_family,
        }

    def _cluster_candidates(self, candidates: List[Dict[str, Any]]) -> List[CPCCluster]:
        """Group candidates by family."""
        family_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            family = normalize_to_family(symbol)
            family_groups[family].append(candidate)

        clusters = []
        for family, group in family_groups.items():
            cluster = CPCCluster(family)
            for c in group:
                cluster.add_candidate(c)
            clusters.append(cluster)

        # Merge very similar clusters in same section
        clusters = self._merge_similar_clusters(clusters)
        return clusters

    def _merge_similar_clusters(self, clusters: List[CPCCluster]) -> List[CPCCluster]:
        """Merge clusters with very similar titles in same section."""
        if len(clusters) <= 1:
            return clusters

        merged = []
        used = set()

        for i, cluster_i in enumerate(clusters):
            if i in used:
                continue

            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue

                cluster_j = clusters[j]

                # Only merge same section
                if cluster_i.family[0] != cluster_j.family[0]:
                    continue

                # Check title similarity
                max_sim = 0.0
                for ci in cluster_i.candidates:
                    for cj in cluster_j.candidates:
                        sim = lexical_similarity(
                            ci.get("title", ""), cj.get("title", "")
                        )
                        max_sim = max(max_sim, sim)

                if max_sim > 0.6:
                    for c in cluster_j.candidates:
                        cluster_i.add_candidate(c)
                    used.add(j)

            merged.append(cluster_i)
            used.add(i)

        return merged

    def _build_hypotheses(
        self, ranked_clusters: List[CPCCluster]
    ) -> List[Dict[str, Any]]:
        """Build max 2 hypotheses (PRIMARY + optional SECONDARY)."""
        hypotheses = []

        for idx, cluster in enumerate(ranked_clusters[: self.max_hypotheses]):
            if idx == 0:
                role = "primary"
                is_primary = True
            elif idx == 1:
                # Only secondary if gap is reasonable (< 50%)
                primary_score = ranked_clusters[0].normalized_score
                if primary_score > 0:
                    gap_ratio = (
                        primary_score - cluster.normalized_score
                    ) / primary_score
                    if gap_ratio < 0.5:
                        role = "secondary"
                        is_primary = False
                    else:
                        # Gap too large, discard
                        logger.info(
                            "Phase 4: Discarding %s (gap=%.2f too large)",
                            cluster.family,
                            gap_ratio,
                        )
                        break
                else:
                    role = "secondary"
                    is_primary = False
            else:
                break  # Max 2 hypotheses

            hypotheses.append(
                {
                    "family": cluster.family,
                    "role": role,
                    "score": round(cluster.raw_score, 4),
                    "normalized_score": round(cluster.normalized_score, 4),
                    "mean_score": round(cluster.get_mean_score(), 4),
                    "candidate_count": len(cluster.candidates),
                    "supporting_codes": cluster.get_supporting_codes(),
                    "coherence": round(cluster.coherence, 4),
                    "reasoning": cluster.get_reasoning(is_primary=is_primary),
                }
            )

        return hypotheses

    def _compute_confidence(self, strong_clusters: List[CPCCluster]) -> str:
        """Compute overall confidence."""
        if not strong_clusters:
            return "low"

        if len(strong_clusters) == 1:
            return "high"

        primary = strong_clusters[0]
        secondary = strong_clusters[1]

        if primary.normalized_score > 0 and secondary.normalized_score > 0:
            gap_ratio = (
                primary.normalized_score - secondary.normalized_score
            ) / primary.normalized_score

            if gap_ratio > 0.5:
                return "high"
            elif gap_ratio > 0.3:
                return "medium"
            else:
                return "low"

        return "medium"

    def _build_overall_reasoning(
        self,
        hypotheses: List[Dict[str, Any]],
        strong_clusters: List[CPCCluster],
        support_weight: float,
    ) -> str:
        """Build overall reasoning with interpretation."""
        if not hypotheses:
            return "No hypotheses formed."

        primary = hypotheses[0]
        lines = [
            f"Consolidated {len(strong_clusters)} clusters into {len(hypotheses)} hypothesis{'es' if len(hypotheses) > 1 else ''}."
        ]

        # Interpretation
        if support_weight > 0.75:
            interpretation = "Strong single-domain invention"
        elif support_weight > 0.5:
            interpretation = "Primary domain with secondary support"
        else:
            interpretation = "Multi-domain invention"

        lines.append(
            f"{interpretation}: {primary['family']} is the core technical domain "
            f"(support_weight={support_weight:.2f})."
        )

        if len(hypotheses) > 1:
            secondary = hypotheses[1]
            lines.append(
                f"Secondary: {secondary['family']} provides supporting infrastructure."
            )

        return " ".join(lines)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "phase4_hypotheses": [],
            "phase4_primary_family": "",
            "phase4_centroid_family": "",
            "phase4_support_weight": 0.0,
            "phase4_confidence": "low",
            "phase4_cluster_count": 0,
            "phase4_strong_clusters": 0,
            "phase4_discarded_clusters": 0,
            "phase4_reasoning": "No candidates available for consolidation.",
        }


def consolidate_cpc_hypotheses(
    ranked_candidates: List[Dict[str, Any]],
    term_importance: Dict[str, float],
    max_hypotheses: int = 2,
) -> Dict[str, Any]:
    """Quick function to consolidate Phase 3 candidates."""
    consolidator = CPCHypothesisConsolidator(max_hypotheses=max_hypotheses)
    return consolidator.consolidate(ranked_candidates, term_importance)
