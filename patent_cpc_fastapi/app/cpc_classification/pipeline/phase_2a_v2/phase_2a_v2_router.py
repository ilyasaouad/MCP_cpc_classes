"""
phase_2a_v2_router.py — Phase 2A v2 Fusion Scoring Engine.

Replaces heuristic keyword routing with multi-source fusion:
  final_score = 0.45 × embedding_score
              + 0.35 × kg_score
              + 0.20 × anchor_score

All signals are aligned to CPC FAMILY level (4-char, e.g. G10L)
before fusion. Class-level scores (3-char, e.g. G10) are expanded
to their child families using the KG hierarchy.

All components are deterministic or semantic — no LLM dependency.
Outputs top 3–5 CPC families with evidence traces.
"""

import logging
from typing import Dict, List, Optional

from .cpc_kg_client import CPCKGClient
from .embedding_router import EmbeddingRouter
from .anchor_matcher import extract_anchors, match_anchors_to_cpc

logger = logging.getLogger(__name__)

# ── Fusion weights ──
W_EMBEDDING = 0.45
W_KG = 0.35
W_ANCHOR = 0.20

# ── Confidence thresholds ──
MIN_SCORE = 0.05  # Minimum final fused score to keep
MIN_SIGNAL = 0.1  # Minimum anchor score (if anchor is the only signal)
TOP_K = 3  # Max families returned
ANCHOR_ONLY_PENALTY = 0.5  # Penalty for families with only anchor signal

# ── Canonical CPC families (4-char) ──
_CPC_FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "G10L": "Speech and voice analysis; speech recognition; speech synthesis; "
    "audio signal processing; acoustic processing; phoneme modeling; "
    "speaker recognition; speech coding; speech enhancement",
    "G06N": "Computing arrangements based on biological models; neural networks; "
    "machine learning; deep learning; training algorithms; inference; "
    "model optimization; self-supervised learning; clustering; transformers",
    "G06F": "Digital data processing; program control; information retrieval; "
    "database structures; file management; computing infrastructure",
    "G06T": "Image data processing; image analysis; image enhancement; "
    "image segmentation; computer vision; graphics rendering",
    "G06V": "Video understanding; scene analysis; object detection; motion analysis",
    "H04L": "Data communication networks; transmission of digital information; "
    "protocols; routing; switching; network architectures",
    "H04W": "Wireless communication; radio networks; base station control; "
    "channel allocation; mobile networks",
    "G05B": "Control systems; industrial automation; process control; robotics",
    "A61B": "Medical diagnosis; surgery; clinical instruments; patient monitoring",
    "B60W": "Vehicle control; autonomous driving; steering; braking; cruise control",
    "B60L": "Electric vehicle propulsion; battery management; charging",
    "B60R": "Vehicle fittings; signal systems; safety equipment",
    "F16J": "Sealing; pistons; cylinders; mechanical joints",
    "F16K": "Valves; taps; fluid flow control; pressure regulation",
    "E21B": "Earth drilling; borehole; well completion; oil and gas extraction",
    "G01N": "Chemical/physical analysis; measurement; testing; sensing; detection",
    "B01D": "Separation; filtration; membrane processes; purification",
}


class Phase2AV2Router:
    """
    Phase 2A v2: Multi-source fusion CPC family router.

    Routes patent text to top 3–5 CPC families using:
      1. Embedding similarity (45%) — semantic match against CPC descriptions
      2. KG graph structure (35%) — neighbor coherence in CPC hierarchy
      3. Anchor keyword match (20%) — deterministic claim-term mapping
    """

    def __init__(self, knowledge_graph=None):
        self.kg_client = CPCKGClient(knowledge_graph)
        self.embedding_router = EmbeddingRouter(self.kg_client)

    @staticmethod
    def _align_to_family_level(
        scores: Dict[str, float],
        canonical_families: set,
        kg_client: Optional[CPCKGClient] = None,
    ) -> Dict[str, float]:
        """
        Ensure all score keys are at CPC family level (4-char).

        Steps:
        1. Normalize keys to 4-char (e.g. "G10" → expand to G10L, G10K, …)
        2. Filter to canonical family set
        3. Fill missing families with 0.0

        Returns a dict with exactly the canonical family keys.
        """
        aligned: Dict[str, float] = {}

        for key, value in scores.items():
            # Already 4-char family
            if len(key) >= 4:
                fam = key[:4]
                aligned[fam] = max(aligned.get(fam, 0.0), value)
            # 3-char class — expand to families via KG
            elif len(key) == 3 and kg_client:
                families = kg_client.get_families_for_class(key)
                if families:
                    for fam in families:
                        if fam in canonical_families:
                            aligned[fam] = max(aligned.get(fam, 0.0), value)
                else:
                    # No KG data — keep the 3-char key as-is for later
                    aligned[key] = value
            else:
                aligned[key] = value

        # Ensure all canonical families are present (fill missing with 0)
        for fam in canonical_families:
            if fam not in aligned:
                aligned[fam] = 0.0

        return aligned

    def route(
        self,
        phase1: dict,
        phase15_result: Optional[dict] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Route patent to top CPC families via multi-source fusion.

        Returns:
            {
                "families": [{"family": "G10L", "score": 0.91, "evidence": [...]}],
                "source": "phase_2a_v2_fusion",
                "debug": {...}
            }
        """
        technical_object = phase1.get("technical_object", "")
        core_function = phase1.get("core_function", "")
        cpc_terms = phase1.get("cpc_terms", [])
        core_function_generalized = phase1.get("core_function_generalized", [])

        # ── Identify validated families from Phase 1.2 forensic audit ──
        validated_families = set()
        audit = phase1.get("phase1_2_audit", {})
        if audit:
            # Primary anchor
            pa = audit.get("final_primary_anchor", "")
            if pa:
                validated_families.add(pa[:4])
            # Secondary anchors
            for sec in audit.get("secondary_anchors", []):
                if isinstance(sec, str):
                    validated_families.add(sec[:4])
                elif isinstance(sec, dict):
                    fam = sec.get("cpc_family", sec.get("family", ""))
                    if fam:
                        validated_families.add(fam[:4])

        # Build enriched query text from all available Phase 1 layers
        query_parts = [technical_object, core_function]
        if cpc_terms:
            query_parts.append(" ".join(cpc_terms))
        if core_function_generalized:
            query_parts.append(" ".join(core_function_generalized))
        enriched_query = " ".join(q for q in query_parts if q)

        # ── 1. Extract anchors from Phase 1 ──
        anchors = extract_anchors(phase1)
        anchor_scores = match_anchors_to_cpc(anchors)

        # ── 2. Embedding similarity scoring (ALWAYS computed) ──
        embedding_scores = self.embedding_router.score_families(
            enriched_query,
            "",
            candidate_families=list(_CPC_FAMILY_DESCRIPTIONS.keys()),
            top_k=10,
        )
        embedding_available = bool(embedding_scores)

        # ── 3. KG graph-aware scoring (ALWAYS computed) ──
        kg_scores = self._compute_kg_scores(
            enriched_query, core_function, embedding_scores, anchor_scores
        )
        kg_available = bool(kg_scores)

        # ── 4. ALIGN all signals to family level (4-char) ──
        canonical_families = set(_CPC_FAMILY_DESCRIPTIONS.keys())

        embedding_scores = self._align_to_family_level(
            embedding_scores, canonical_families, self.kg_client
        )
        kg_scores = self._align_to_family_level(
            kg_scores, canonical_families, self.kg_client
        )
        anchor_scores = self._align_to_family_level(
            anchor_scores, canonical_families, self.kg_client
        )

        # ── Determine fusion mode ──
        embedding_available = any(v > 0 for v in embedding_scores.values())
        kg_available = any(v > 0 for v in kg_scores.values())
        if embedding_available and kg_available:
            fusion_mode = "full"
        elif embedding_available or kg_available:
            fusion_mode = "partial"
        else:
            fusion_mode = "anchor_only"
            logger.warning(
                "Phase 2A v2: fusion_mode=anchor_only — embedding and KG both unavailable"
            )

        # ── 5. Fuse scores (all three signals at same family level) ──
        fused: Dict[str, dict] = {}
        anchor_only_penalized: List[str] = []

        for family in canonical_families:
            emb = embedding_scores.get(family, 0.0)
            kg = kg_scores.get(family, 0.0)
            anc = anchor_scores.get(family, 0.0)

            # Apply anchor-only penalty (unless validated by Phase 1.2 audit)
            if emb == 0 and kg == 0 and anc > 0 and family not in validated_families:
                anc = round(anc * ANCHOR_ONLY_PENALTY, 4)
                anchor_only_penalized.append(family)

            final = round(W_EMBEDDING * emb + W_KG * kg + W_ANCHOR * anc, 4)
            evidence = []
            if emb > 0:
                evidence.append(f"embedding_similarity={emb:.3f}")
            if kg > 0:
                evidence.append(f"kg_structure={kg:.3f}")
            if anc > 0:
                evidence.append(f"anchor_match={anc:.3f}")
            fused[family] = {
                "family": family,
                "score": final,
                "evidence": evidence,
                "embedding_score": emb,
                "kg_score": kg,
                "anchor_score": anc,
                "anchor_only_penalized": family in anchor_only_penalized,
            }

        # ── Collect Phase 1.2 rejected families — must never appear in output ──
        rejected_families: set = set()
        audit = phase1.get("phase1_2_audit", {})
        for sv in audit.get("signal_validations", []):
            if sv.get("status") == "rejected":
                fam = sv.get("cpc_family", "")
                if fam:
                    rejected_families.add(fam[:4])
        # Also pick up any domain_signals downgraded to confidence < 0.15
        for sig in phase1.get("domain_signals", []):
            if sig.get("role") == "negative" or sig.get("confidence", 1.0) < 0.15:
                fam = sig.get("cpc_family", "")
                if fam:
                    rejected_families.add(fam[:4])
        if rejected_families:
            logger.info(
                "Phase 2A v2: Honouring Phase 1.2 rejections — excluding %s",
                sorted(rejected_families),
            )

        # Sort and rank all families
        all_ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        # ── 6. FILTER low-confidence families ──
        filtered_out = []
        ranked = []

        for entry in all_ranked:
            fam = entry["family"]
            score = entry["score"]
            emb = entry["embedding_score"]
            kg = entry["kg_score"]
            anc = entry["anchor_score"]

            # Rule 0: Phase 1.2 explicit rejection — hard block regardless of score
            if fam in rejected_families:
                filtered_out.append(fam)
                logger.info("Phase 2A v2: %s blocked — rejected by Phase 1.2 audit", fam)
                continue

            # Rule 1: final score below threshold
            if score < MIN_SCORE:
                filtered_out.append(fam)
                continue

            # Rule 2: must have at least one meaningful signal
            has_embedding = emb > 0
            has_kg = kg > 0
            has_anchor = anc >= MIN_SIGNAL

            if not (has_embedding or has_kg or has_anchor):
                filtered_out.append(fam)
                continue

            ranked.append(entry)

        # Rule 3: limit to TOP_K
        ranked = ranked[:TOP_K]

        logger.info(
            "Phase 2A v2: %s → %d families (filtered %d, mode=%s, emb=%.3f, kg=%.3f, anc=%.0f)",
            core_function[:60] or technical_object[:60],
            len(ranked),
            len(filtered_out),
            fusion_mode,
            max(embedding_scores.values()) if embedding_scores else 0.0,
            max(kg_scores.values()) if kg_scores else 0.0,
            max(anchor_scores.values()) if anchor_scores else 0.0,
        )

        debug = {
            "alignment_level": "family",
            "anchors_found": {k: len(v) for k, v in anchors.items()},
            "anchor_scores": {k: v for k, v in anchor_scores.items() if v > 0},
            "embedding_raw_top3": dict(
                sorted(
                    [item for item in embedding_scores.items() if item[1] > 0],
                    key=lambda x: -x[1],
                )[:3]
            ),
            "kg_raw_top3": dict(
                sorted(
                    [item for item in kg_scores.items() if item[1] > 0],
                    key=lambda x: -x[1],
                )[:3]
            ),
            "fusion_weights": {
                "embedding": W_EMBEDDING,
                "kg": W_KG,
                "anchor": W_ANCHOR,
            },
            "fusion_mode": fusion_mode,
            "filtered_out": filtered_out,
            "anchor_only_penalty_applied": anchor_only_penalized,
        }

        return {
            "families": ranked,
            "family_names": [f["family"] for f in ranked],
            "primary": ranked[0]["family"] if ranked else "",
            "secondary": [f["family"] for f in ranked[1:3]] if len(ranked) > 1 else [],
            "source": "phase_2a_v2_fusion",
            "debug": debug,
        }

    def _compute_kg_scores(
        self,
        technical_object: str,
        core_function: str,
        embedding_scores: Dict[str, float],
        anchor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        KG-aware scoring: always computed, uses fallback when KG unavailable.

        Primary: graph neighbor coherence from KG.
        Fallback: text-based relevance scoring against CPC definitions.
        """
        kg_scores: Dict[str, float] = {}

        # ── Primary: KG graph traversal ──
        if self.kg_client and self.kg_client.is_available():
            try:
                families_to_check = set(embedding_scores) | set(anchor_scores)
                for family in families_to_check:
                    neighbors = self.kg_client.get_cpc_neighbors(family, depth=1)
                    if neighbors and family in embedding_scores:
                        neighbor_score_sum = sum(
                            embedding_scores.get(n, 0.0) for n in neighbors[:10]
                        )
                        coherence = min(
                            neighbor_score_sum / max(len(neighbors[:10]), 1), 1.0
                        )
                        kg_scores[family] = round(
                            embedding_scores[family] * 0.6 + coherence * 0.4, 3
                        )
                    elif family in embedding_scores:
                        kg_scores[family] = round(embedding_scores[family] * 0.6, 3)

                # Partial text fallback: cover families with anchor signal but no
                # embedding score (they were skipped by the primary KG loop above).
                uncovered = families_to_check - set(kg_scores)
                if uncovered:
                    text = f"{technical_object} {core_function}".strip().lower()
                    text_words = set(text.split()) if text else set()
                    for family in uncovered:
                        desc = _CPC_FAMILY_DESCRIPTIONS.get(family, "")
                        if desc and text_words:
                            desc_words = set(desc.lower().split())
                            overlap = text_words & desc_words
                            if overlap:
                                raw = len(overlap) / max(len(desc_words), 1)
                                kg_scores[family] = round(min(raw * 2, 1.0), 3)
                            else:
                                kg_scores[family] = 0.0
                    logger.info(
                        "KG partial text fallback: scored %d uncovered families %s",
                        len(uncovered),
                        sorted(uncovered),
                    )

                if kg_scores:
                    return kg_scores

            except Exception as e:
                logger.warning("KG graph scoring failed: %s, using fallback", e)

        # ── Fallback: text-based relevance against CPC definitions ──
        logger.info("KG scoring: using fallback text-relevance scoring")
        text = f"{technical_object} {core_function}".strip().lower()
        if not text:
            # Return zero scores for all known families
            return {f: 0.0 for f in _CPC_FAMILY_DESCRIPTIONS}

        text_words = set(text.split())
        for family, desc in _CPC_FAMILY_DESCRIPTIONS.items():
            desc_lower = desc.lower()
            desc_words = set(desc_lower.split())
            overlap = text_words & desc_words
            if overlap:
                # Simple word overlap ratio
                score = len(overlap) / max(len(desc_words), 1)
                kg_scores[family] = round(min(score * 2, 1.0), 3)
            else:
                kg_scores[family] = 0.0

        # Normalize to [0, 1]
        max_kg = max(kg_scores.values()) if kg_scores else 1.0
        if max_kg > 0:
            for f in kg_scores:
                kg_scores[f] = round(kg_scores[f] / max_kg, 3)

        return kg_scores
