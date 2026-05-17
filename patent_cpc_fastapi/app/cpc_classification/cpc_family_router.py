"""
cpc_family_router.py - Phase 2A: CPC Family Router

Routes patent text to top CPC families (3-character codes) using semantic analysis.
Uses modality detection to identify PRIMARY technical purpose vs secondary tools.

Usage:
    router = CPCFamilyRouter(knowledge_graph)
    families = router.route(phase1_data)
    # Returns: {"families": ["G06V", "G06N"], "reasoning": "...", "primary": "G06V"}
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# Fallback families when routing fails
FALLBACK_FAMILIES = ["G06", "H04"]


class CPCDomainTaxonomy:
    """
    Taxonomy mapping technical domains to CPC families with weights.

    CRITICAL: Distinguishes PURPOSE domains (what the invention does)
    from TOOL domains (technologies used to do it).

    Purpose domains get higher weight because CPC classification is based
    on technical purpose, not the tools used.
    """

    # PURPOSE domains: what the invention actually does (subject matter)
    # These dominate classification
    PURPOSE_DOMAINS = {
        # Computer Vision & Image Analysis
        "computer vision": ("G06V", 1.2),
        "image analysis": ("G06T", 1.2),
        "image processing": ("G06T", 1.2),
        "image segmentation": ("G06V", 1.3),
        "segmentation": ("G06V", 1.1),
        "object detection": ("G06V", 1.2),
        "defect detection": ("G06T", 1.3),
        "quality inspection": ("G06T", 1.2),
        "optical inspection": ("G01N", 1.2),
        "visual inspection": ("G06T", 1.1),
        "pattern recognition": ("G06V", 1.1),
        "facial recognition": ("G06V", 1.1),
        # Neural Networks & AI (when they ARE the invention, not just tools)
        "neural network": ("G06N", 1.3),  # NN IS the subject matter
        "neural networks": ("G06N", 1.3),
        "deep learning": ("G06N", 1.2),
        "artificial intelligence": ("G06N", 1.0),  # Only when AI is the focus
        "machine learning": ("G06N", 1.2),
        "large language model": ("G06N", 1.2),  # LLM IS the invention subject
        "llm": ("G06N", 1.2),
        "transformer": ("G06N", 1.2),
        "model compression": ("G06N", 1.3),
        "quantization": ("G06N", 1.3),  # NN quantization = G06N subject
        "weight quantization": ("G06N", 1.4),
        "model optimization": ("G06N", 1.3),
        "inference optimization": ("G06N", 1.3),
        "training acceleration": ("G06N", 1.2),
        "reinforcement learning": ("G06N", 1.1),
        "generative ai": ("G06N", 1.1),
        # Manufacturing & Mechanical
        "manufacturing": ("B23", 1.1),
        "machining": ("B23", 1.1),
        "production": ("B23", 1.0),
        "assembly": ("B23", 1.0),
        "mechanical": ("F16", 1.0),
        "vehicle": ("B60", 1.1),
        "automotive": ("B60", 1.1),
        # Medical
        "medical": ("A61", 1.1),
        "healthcare": ("A61", 1.1),
        "diagnosis": ("A61", 1.1),
        # Communication
        "wireless": ("H04", 1.0),
        "communication": ("H04", 1.0),
        "network": ("H04", 1.0),
        "telecommunication": ("H04", 1.0),
        # Control & Automation
        "control system": ("G05", 1.0),
        "automation": ("G05", 1.0),
        "robot": ("B25J", 1.1),
        "robotics": ("B25J", 1.1),
        # Measurement & Sensors
        "sensor": ("G01", 1.0),
        "measurement": ("G01", 1.0),
        "optical measurement": ("G01", 1.1),
        # Energy & Power
        "energy": ("H01", 1.0),
        "battery": ("H01", 1.0),
        "power": ("H01", 1.0),
        "semiconductor": ("H01", 1.1),
        "circuit": ("H01", 1.0),
        # Materials & Chemistry
        "material": ("C01", 1.0),
        "chemical": ("C01", 1.0),
        # Data Processing (as purpose, not tool)
        "database": ("G06F", 0.9),
        "information retrieval": ("G06F", 0.9),
        "cloud computing": ("G06F", 0.9),
        "server": ("G06F", 0.8),
    }

    # TOOL domains: technologies used to achieve the purpose
    # These get lower weight because they're means, not ends
    # NOTE: NN/AI terms removed from here - they are in PURPOSE_DOMAINS when they ARE the invention
    TOOL_DOMAINS = {
        # AI/ML as tools (only when used for something else, e.g., "AI for image processing")
        "artificial intelligence": ("G06N", 0.4),  # Generic AI mention
        "chatbot": ("G06N", 0.3),  # Chatbot is usually an application
        "fine-tuning": ("G06N", 0.3),  # Fine-tuning can be tool or subject
        # NLP (tool for language processing)
        "natural language processing": ("G06F", 0.5),
        "text processing": ("G06F", 0.5),
        "sentiment analysis": ("G06F", 0.5),
        # Computing infrastructure (tools)
        "computing": ("G06F", 0.3),
        "data processing": ("G06F", 0.3),
        "processor": ("G06F", 0.3),
        "algorithm": ("G06F", 0.3),
        "embedded system": ("G06F", 0.4),
        "software": ("G06F", 0.3),
        "program": ("G06F", 0.3),
        "gpu": ("G06F", 0.3),  # GPU is hardware tool
        "tpu": ("G06F", 0.3),
        # Signal processing (tool)
        "signal processing": ("G06F", 0.4),
        "audio processing": ("G10L", 0.5),
        "speech recognition": ("G10L", 0.5),
        "voice": ("G10L", 0.5),
        # Specific AI techniques (lower weight - these are methods, not subjects)
        "regression": ("G06N", 0.3),
        "classification": ("G06N", 0.3),
        "clustering": ("G06N", 0.3),
    }

    # FUNCTION domains: neither purpose nor tool, but functional mechanisms
    # Medium weight
    FUNCTION_DOMAINS = {
        "rule-based": ("G06N", 0.7),
        "expert system": ("G06N", 0.6),
        "inference": ("G06N", 0.6),
        "logical reasoning": ("G06N", 0.6),
        "decision support": ("G06Q", 0.7),
        "decision tree": ("G06N", 0.6),
        "knowledge graph": ("G06N", 0.5),
        "ontology": ("G06N", 0.5),
    }

    # CO-OCCURRENCE rules: when domains appear together
    COOCCURRENCE_RULES = [
        # Vision + AI → Vision is primary, AI is tool
        {
            "conditions": {
                "computer vision",
                "image",
                "segmentation",
                "defect",
                "inspection",
            },
            "required_any": {
                "artificial intelligence",
                "machine learning",
                "neural network",
                "llm",
                "deep learning",
            },
            "primary": "G06V",
            "secondary": "G06N",
            "boost_primary": 0.4,
            "penalize_secondary": 0.5,
        },
        # NLP + AI → NLP is primary (function), AI is tool
        {
            "conditions": {
                "natural language processing",
                "text",
                "language",
                "sentiment",
                "chatbot",
            },
            "required_any": {
                "artificial intelligence",
                "machine learning",
                "neural network",
                "llm",
            },
            "primary": "G06F",
            "secondary": "G06N",
            "boost_primary": 0.3,
            "penalize_secondary": 0.5,
        },
        # Speech + AI → Speech is primary
        {
            "conditions": {"speech", "voice", "audio"},
            "required_any": {
                "artificial intelligence",
                "machine learning",
                "neural network",
            },
            "primary": "G10L",
            "secondary": "G06N",
            "boost_primary": 0.3,
            "penalize_secondary": 0.5,
        },
        # Manufacturing + Vision → Both important
        {
            "conditions": {"manufacturing", "machining", "production"},
            "required_any": {"computer vision", "image", "inspection", "defect"},
            "primary": "B23",
            "secondary": "G06T",
            "boost_primary": 0.2,
            "penalize_secondary": 0.0,
        },
        # Medical + AI → Medical is primary
        {
            "conditions": {"medical", "healthcare", "diagnosis"},
            "required_any": {"artificial intelligence", "machine learning"},
            "primary": "A61",
            "secondary": "G06N",
            "boost_primary": 0.3,
            "penalize_secondary": 0.5,
        },
        # Vehicle + AI → Vehicle is primary
        {
            "conditions": {"vehicle", "automotive", "car"},
            "required_any": {
                "artificial intelligence",
                "machine learning",
                "neural network",
            },
            "primary": "B60",
            "secondary": "G06N",
            "boost_primary": 0.3,
            "penalize_secondary": 0.5,
        },
    ]


class CPCFamilyRouter:
    """
    Phase 2A: Routes patent to top CPC families using semantic analysis.

    Uses modality detection to identify PRIMARY technical purpose.
    Correctly handles AI-as-tool vs AI-as-purpose distinction.
    """

    def __init__(
        self, knowledge_graph, max_families: int = 3, min_score_threshold: float = 0.15
    ):
        self.knowledge_graph = knowledge_graph
        self.max_families = max_families
        self.min_score_threshold = min_score_threshold
        self.taxonomy = CPCDomainTaxonomy()

    def route(
        self, phase1_data: Dict[str, Any], phase15_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route patent to top CPC families.

        Args:
            phase1_data: Output from Phase 1
            phase15_data: Optional output from Phase 1.5 (invention role)

        Returns dict with:
            - families: List of 3-char CPC families
            - primary: Primary family (the technical purpose)
            - secondary: Secondary families
            - reasoning: Explanation
            - scores: Dict of family -> score
            - source: Routing method used
            - modality: Detected primary modality
        """
        if not self.knowledge_graph or not self.knowledge_graph.embeddings:
            logger.warning(
                "Knowledge graph not available, using domain signal fallback"
            )
            return self._route_from_domain_analysis(phase1_data, phase15_data)

        try:
            return self._route_from_embeddings(phase1_data, phase15_data)
        except Exception as e:
            logger.warning("Embedding routing failed: %s, using fallback", e)
            return self._route_from_domain_analysis(phase1_data, phase15_data)

    def _route_from_embeddings(
        self, phase1_data: Dict[str, Any], phase15_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route using knowledge graph embeddings with modality-aware scoring."""
        technical_object = phase1_data.get("technical_object", "")
        core_function = phase1_data.get("core_function", "")
        system_context = phase1_data.get("system_context", "")

        patent_text = f"{technical_object} {core_function} {system_context}".strip()
        top_terms = self._get_top_terms(phase1_data, n=15)
        term_text = " ".join(top_terms)

        # Get base scores from embeddings
        semantic_results = self.knowledge_graph.find_classes_by_text(
            patent_text, top_k=20
        )
        term_results = (
            self.knowledge_graph.find_classes_by_text(term_text, top_k=20)
            if term_text
            else []
        )

        combined_scores = {}
        for family, score in semantic_results:
            if score >= self.min_score_threshold:
                combined_scores[family] = score * 0.6
        for family, score in term_results:
            if score >= self.min_score_threshold:
                combined_scores[family] = combined_scores.get(family, 0) + score * 0.4

        # Apply role-based scoring from Phase 1.5
        if phase15_data and phase15_data.get("role"):
            from .cpc_role_classifier import apply_role_scoring

            combined_scores = apply_role_scoring(
                combined_scores, phase15_data["role"], phase1_data
            )
            logger.info(
                "Phase 2A: Applied role-based scoring for role=%s", phase15_data["role"]
            )

        # Apply hard domain constraints (GUARDRAILS)
        combined_scores = self._apply_hard_constraints(phase1_data, combined_scores)

        # Apply modality correction
        combined_scores = self._apply_modality_correction(phase1_data, combined_scores)

        return self._select_families(combined_scores, phase1_data, "embedding")

    def _route_from_domain_analysis(
        self, phase1_data: Dict[str, Any], phase15_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route using domain signal analysis with purpose/tool distinction.

        This is the primary fallback when KG is unavailable.
        Uses the new taxonomy to correctly weight purpose vs tool domains.
        """
        domain_signals = phase1_data.get("domain_signals", [])
        strategy = phase1_data.get("classification_strategy", "")
        system_context = phase1_data.get("system_context", "")
        core_function = phase1_data.get("core_function", "")
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))

        # Collect all domain mentions
        all_domains = []

        # From explicit domain signals
        for signal in domain_signals:
            if isinstance(signal, dict):
                name = signal.get("name", "").lower()
                conf = signal.get("confidence", 0.5)
                all_domains.append((name, conf, "signal"))

        # From extracted terms
        for term_obj in terms:
            if isinstance(term_obj, dict):
                term = term_obj.get("term", "").lower()
                importance = term_obj.get("importance", 5) / 10.0  # Normalize to 0-1
                all_domains.append((term, importance, "term"))

        # From system context and core function
        text_to_check = f"{system_context} {core_function}".lower()
        all_domains.append((text_to_check, 0.3, "context"))

        # Score families using taxonomy
        family_scores = {}
        detected_domains = set()

        # ── Apply primary_domain boost from Phase 1 (CRITICAL) ──
        primary_domain = phase1_data.get("primary_domain", {})
        if primary_domain and isinstance(primary_domain, dict):
            pd_name = primary_domain.get("name", "").lower()
            pd_cpc = primary_domain.get("cpc_class", "")
            pd_conf = primary_domain.get("confidence", 0.5)

            # Boost the primary domain family
            if pd_cpc:
                family_scores[pd_cpc] = family_scores.get(pd_cpc, 0) + (pd_conf * 3.0)
                logger.info(
                    "PRIMARY DOMAIN BOOST: %s → %s (score=%.3f)",
                    pd_name,
                    pd_cpc,
                    pd_conf * 3.0,
                )

            # Also check if primary domain maps to a family in taxonomy
            for keyword, (family, weight) in self.taxonomy.PURPOSE_DOMAINS.items():
                if keyword in pd_name or keyword in text_to_check:
                    score = pd_conf * weight * 2.0  # Double boost for primary domain
                    family_scores[family] = family_scores.get(family, 0) + score
                    detected_domains.add(keyword)
                    logger.info(
                        "PRIMARY DOMAIN PURPOSE: %s → %s (score=%.3f)",
                        keyword,
                        family,
                        score,
                    )

        for domain_text, confidence, source in all_domains:
            # Check PURPOSE domains (highest weight)
            for keyword, (family, weight) in self.taxonomy.PURPOSE_DOMAINS.items():
                if keyword in domain_text:
                    score = confidence * weight
                    family_scores[family] = family_scores.get(family, 0) + score
                    detected_domains.add(keyword)
                    logger.debug(
                        "PURPOSE: %s → %s (score=%.3f)", keyword, family, score
                    )

            # Check TOOL domains (lower weight)
            for keyword, (family, weight) in self.taxonomy.TOOL_DOMAINS.items():
                if keyword in domain_text:
                    score = confidence * weight
                    family_scores[family] = family_scores.get(family, 0) + score
                    detected_domains.add(keyword)
                    logger.debug("TOOL: %s → %s (score=%.3f)", keyword, family, score)

            # Check FUNCTION domains
            for keyword, (family, weight) in self.taxonomy.FUNCTION_DOMAINS.items():
                if keyword in domain_text:
                    score = confidence * weight
                    family_scores[family] = family_scores.get(family, 0) + score
                    detected_domains.add(keyword)
                    logger.debug(
                        "FUNCTION: %s → %s (score=%.3f)", keyword, family, score
                    )

        # Apply co-occurrence rules
        family_scores = self._apply_cooccurrence_rules(detected_domains, family_scores)

        # Apply role-based scoring from Phase 1.5
        if phase15_data and phase15_data.get("role"):
            from .cpc_role_classifier import apply_role_scoring

            family_scores = apply_role_scoring(
                family_scores, phase15_data["role"], phase1_data
            )
            logger.info(
                "Phase 2A (domain): Applied role=%s for scoring", phase15_data["role"]
            )

        # Apply hard domain constraints (GUARDRAILS)
        family_scores = self._apply_hard_constraints(phase1_data, family_scores)

        # Apply modality-based boost/penalty
        family_scores = self._apply_modality_correction(phase1_data, family_scores)

        return self._select_families(family_scores, phase1_data, "domain_analysis")

    def _apply_cooccurrence_rules(
        self, detected_domains: set, scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply co-occurrence rules to adjust scores."""
        scores = dict(scores)  # Copy

        for rule in self.taxonomy.COOCCURRENCE_RULES:
            # Check if conditions are met
            conditions_met = bool(detected_domains & rule["conditions"])
            required_met = bool(detected_domains & rule["required_any"])

            if conditions_met and required_met:
                primary = rule["primary"]
                secondary = rule["secondary"]

                # Boost primary (the purpose)
                if primary in scores:
                    scores[primary] += rule["boost_primary"]
                else:
                    scores[primary] = rule["boost_primary"]

                # Penalize secondary (the tool)
                if secondary in scores:
                    scores[secondary] *= rule["penalize_secondary"]

                logger.info(
                    "Co-occurrence rule triggered: %s (primary) + %s (tool). "
                    "Boosted %s, penalized %s",
                    primary,
                    secondary,
                    primary,
                    secondary,
                )

        return scores

    def _apply_hard_constraints(
        self, phase1_data: Dict[str, Any], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply hard domain constraints based on semantic signals.

        These are GUARDRAILS that prevent cross-domain misclassification.
        Uses primary_domain from Phase 1 as the main anchor.
        """
        scores = dict(scores)

        # Get all text for analysis
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms_text = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )
        system_context = phase1_data.get("system_context", "").lower()
        core_function = phase1_data.get("core_function", "").lower()
        all_text = f"{all_terms_text} {system_context} {core_function}"

        # Get negative signals
        negative_domains = phase1_data.get("negative_domains", [])
        neg_domains = set()
        for nd in negative_domains:
            if isinstance(nd, dict):
                neg_domains.add(nd.get("domain", "").lower())
            else:
                neg_domains.add(str(nd).lower())

        # ── Constraint 1: Primary Domain Enforcement ──
        # If Phase 1 detected a primary domain, boost it and penalize unrelated families
        primary_domain = phase1_data.get("primary_domain", {})
        if primary_domain and isinstance(primary_domain, dict):
            pd_cpc = primary_domain.get("cpc_class", "")
            pd_conf = primary_domain.get("confidence", 0.5)
            pd_name = primary_domain.get("name", "").lower()

            if pd_cpc and pd_conf >= 0.6:
                # Strong primary domain detected
                scores[pd_cpc] = scores.get(pd_cpc, 0) + (pd_conf * 2.0)
                logger.info(
                    "HARD CONSTRAINT: Primary domain %s detected (conf=%.2f). "
                    "Boosting %s.",
                    pd_name,
                    pd_conf,
                    pd_cpc,
                )

                # Penalize unrelated families
                all_families = set(scores.keys())
                unrelated = all_families - {pd_cpc}
                for family in unrelated:
                    # Check if this family has any domain signals
                    has_signal = any(
                        sig.get("cpc_family", "") == family
                        and sig.get("role", "") != "negative"
                        for sig in phase1_data.get("domain_signals", [])
                        if isinstance(sig, dict)
                    )
                    if not has_signal:
                        scores[family] *= 0.4  # Penalize unrelated
                        logger.info(
                            "HARD CONSTRAINT: Penalizing unrelated %s (no domain signals)",
                            family,
                        )

        # ── Constraint 2: Domain-Specific Subject Matter ──
        # Apply domain-specific boosts based on detected subject matter
        domain_constraints = {
            "G06N": {
                "signals": [
                    "neural network",
                    "neural networks",
                    "deep learning",
                    "llm",
                    "large language model",
                    "transformer",
                    "model compression",
                    "quantization",
                    "weight quantization",
                    "inference optimization",
                    "model optimization",
                    "training acceleration",
                    "reinforcement learning",
                    "generative ai",
                    "parameter optimization",
                    "pruning",
                    "distillation",
                ],
                "internals": [
                    "weight",
                    "parameter",
                    "layer",
                    "tensor",
                    "activation",
                    "gradient",
                ],
                "penalize_unrelated": ["G06T", "G10K", "G06V"],
                "related_keywords": {
                    "G06T": [
                        "image",
                        "pixel",
                        "camera",
                        "visual",
                        "graphics",
                        "rendering",
                    ],
                    "G10K": ["audio", "sound", "acoustic", "speaker", "microphone"],
                    "G06V": [
                        "vision",
                        "object detection",
                        "image recognition",
                        "facial",
                    ],
                },
            },
            "G06T": {
                "signals": [
                    "image processing",
                    "image analysis",
                    "computer graphics",
                    "rendering",
                    "image compression",
                    "image filtering",
                    "image enhancement",
                    "video processing",
                    "image segmentation",
                    "texture",
                    "pixel",
                    "frame",
                    "graphics",
                ],
                "internals": [
                    "pixel",
                    "frame",
                    "texture",
                    "vertex",
                    "shader",
                    "render",
                ],
                "penalize_unrelated": ["G06N", "G10K", "H04L"],
                "related_keywords": {
                    "G06N": [
                        "neural network",
                        "machine learning",
                        "training",
                        "inference",
                    ],
                    "G10K": ["audio", "sound", "acoustic"],
                    "H04L": ["protocol", "transmission", "packet", "network"],
                },
            },
            "H04L": {
                "signals": [
                    "telecommunication",
                    "data transmission",
                    "network protocol",
                    "wireless",
                    "communication system",
                    "error correction",
                    "channel coding",
                    "modulation",
                    "data link",
                    "packet switching",
                    "routing",
                    "bandwidth",
                ],
                "internals": [
                    "protocol",
                    "packet",
                    "channel",
                    "modulation",
                    "bandwidth",
                    "transmission",
                ],
                "penalize_unrelated": ["G06T", "G10K", "A61"],
                "related_keywords": {
                    "G06T": ["image", "pixel", "graphics"],
                    "G10K": ["audio", "sound", "acoustic"],
                    "A61": ["medical", "diagnosis", "surgical"],
                },
            },
            "F16": {
                "signals": [
                    "mechanical",
                    "sealing",
                    "valve",
                    "pump",
                    "engine",
                    "turbine",
                    "bearing",
                    "gear",
                    "structural",
                    "fastener",
                    "coupling",
                    "housing",
                ],
                "internals": ["seal", "valve", "bearing", "gear", "shaft", "housing"],
                "penalize_unrelated": ["G06N", "G06T", "H04L"],
                "related_keywords": {
                    "G06N": ["neural network", "machine learning"],
                    "G06T": ["image", "pixel", "graphics"],
                    "H04L": ["protocol", "transmission", "network"],
                },
            },
            "A61": {
                "signals": [
                    "medical",
                    "healthcare",
                    "diagnosis",
                    "surgical",
                    "therapy",
                    "patient",
                    "clinical",
                    "biomedical",
                    "imaging",
                    "radiology",
                ],
                "internals": [
                    "patient",
                    "surgical",
                    "diagnosis",
                    "therapy",
                    "clinical",
                ],
                "penalize_unrelated": ["G06T", "H04L", "F16"],
                "related_keywords": {
                    "G06T": ["image", "pixel", "graphics"],
                    "H04L": ["protocol", "transmission", "network"],
                    "F16": ["mechanical", "seal", "valve"],
                },
            },
        }

        for family, config in domain_constraints.items():
            has_subject = any(sig in all_text for sig in config["signals"])
            has_internal = any(sig in all_text for sig in config["internals"])

            if has_subject or has_internal:
                # Boost this family
                scores[family] = scores.get(family, 0) + 2.0
                logger.info(
                    "HARD CONSTRAINT: %s subject matter detected. Boosting %s.",
                    family,
                    family,
                )

                # Penalize unrelated families
                for unrelated_family in config["penalize_unrelated"]:
                    if unrelated_family in scores:
                        # Check if there are domain signals for the unrelated family
                        related_keywords = config["related_keywords"].get(
                            unrelated_family, []
                        )
                        has_unrelated_signal = any(
                            kw in all_text for kw in related_keywords
                        )

                        if not has_unrelated_signal:
                            scores[unrelated_family] *= 0.3
                            logger.info(
                                "HARD CONSTRAINT: Penalizing unrelated %s - no %s domain signals",
                                unrelated_family,
                                unrelated_family,
                            )

        # ── Constraint 3: Negative Domain Enforcement ──
        # If user explicitly said invention is NOT about X, penalize X families
        domain_to_family = {
            "image processing": "G06T",
            "computer vision": "G06V",
            "acoustics": "G10K",
            "audio processing": "G10L",
            "medical": "A61",
            "vehicle": "B60",
        }
        for neg_domain, family in domain_to_family.items():
            if neg_domain in neg_domains and family in scores:
                scores[family] *= 0.4  # Strong penalty for negative domains
                logger.info(
                    "HARD CONSTRAINT: Negative domain '%s' → penalizing %s",
                    neg_domain,
                    family,
                )

        # ── Constraint 3: Object-Context Binding ──
        # "weight clipping" → G06N (not G06T image clipping)
        # "image clipping" → G06T (not G06N)
        object_bindings = [
            ("weight clipping", "G06N", 1.5),
            ("weight quantization", "G06N", 1.5),
            ("model compression", "G06N", 1.5),
            ("inference latency", "G06N", 1.3),
            ("image clipping", "G06T", 1.3),
            ("video compression", "G06T", 1.3),
        ]
        for phrase, family, boost in object_bindings:
            if phrase in all_text and family in scores:
                scores[family] *= boost
                logger.info(
                    "HARD CONSTRAINT: Object binding '%s' → boosting %s by %.1fx",
                    phrase,
                    family,
                    boost,
                )

        return scores

    def _apply_modality_correction(
        self, phase1_data: Dict[str, Any], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Detect primary modality and adjust scores.

        If vision terms dominate AI terms, boost vision families.
        If AI terms dominate without clear purpose, keep AI but note ambiguity.
        """
        scores = dict(scores)  # Copy

        # Get all terms text
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms_text = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )

        # Count vision-related terms
        vision_keywords = [
            "image",
            "vision",
            "segmentation",
            "pixel",
            "optical",
            "camera",
            "visual",
            "defect",
            "inspection",
            "mask",
            "boundary",
            "contour",
            "texture",
            "pattern",
        ]
        vision_count = sum(1 for kw in vision_keywords if kw in all_terms_text)

        # Count AI-related terms (expanded)
        ai_keywords = [
            "llm",
            "neural network",
            "neural networks",
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "model",
            "training",
            "inference",
            "classifier",
            "quantization",
            "transformer",
            "weight",
            "parameter",
            "tensor",
        ]
        ai_count = sum(1 for kw in ai_keywords if kw in all_terms_text)

        # Count NN-internal terms (strong signal that NN IS the subject)
        nn_internal_keywords = [
            "weight clipping",
            "weight quantization",
            "model compression",
            "pruning",
            "distillation",
            "activation",
            "gradient",
            "backpropagation",
            "layer",
            "parameter",
        ]
        nn_internal_count = sum(
            1 for kw in nn_internal_keywords if kw in all_terms_text
        )

        # Apply modality-based correction
        if nn_internal_count >= 2:
            # Strong signal: NN is the subject matter itself
            scores["G06N"] = scores.get("G06N", 0) + 1.5
            logger.info(
                "Modality correction: NN INTERNAL subject matter (count=%d). "
                "Boosting G06N strongly.",
                nn_internal_count,
            )

        if vision_count > 0 and vision_count >= ai_count:
            # Vision is primary modality
            for family in ["G06V", "G06T"]:
                if family in scores:
                    scores[family] *= 1.3  # Boost vision
            if "G06N" in scores and nn_internal_count < 2:
                # Only penalize AI if NN is NOT the subject
                scores["G06N"] *= 0.7
            logger.info(
                "Modality correction: VISION primary (vision=%d, ai=%d, nn_internal=%d)",
                vision_count,
                ai_count,
                nn_internal_count,
            )

        elif ai_count > 0 and ai_count > vision_count * 2:
            # AI might be primary (but only if no strong purpose signal)
            purpose_families = ["G06V", "G06T", "B60", "A61", "B23"]
            has_strong_purpose = any(
                scores.get(f, 0) > scores.get("G06N", 0) * 0.8 for f in purpose_families
            )

            if not has_strong_purpose or nn_internal_count >= 2:
                # Pure AI invention or NN is subject matter
                logger.info(
                    "Modality correction: AI primary (nn_internal=%d)",
                    nn_internal_count,
                )
            else:
                # AI is tool, purpose is stronger
                if "G06N" in scores and nn_internal_count < 2:
                    scores["G06N"] *= 0.8
                logger.info("Modality correction: PURPOSE primary, AI is tool")

        return scores

    def _select_families(
        self, scores: Dict[str, float], phase1_data: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        """
        Select top families from scores with evidence-based reasoning.

        No legacy fallback. If scores are empty, returns empty families
        with zero confidence — the caller must handle or retry.
        """
        strategy = phase1_data.get("classification_strategy", "unknown")

        sorted_families = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_families:
            logger.error(
                "Phase 2A: No CPC families scored from %s — "
                "returning empty. Pipeline should retry or report error.",
                source,
            )
            return {
                "families": [],
                "primary": "",
                "secondary": [],
                "reasoning": "No CPC family candidates produced. Source was empty or weak.",
                "scores": {},
                "source": source,
                "modality": self._detect_modality(phase1_data),
            }

        top_families = [family for family, _ in sorted_families[: self.max_families]]
        primary = top_families[0]
        secondary = top_families[1:] if len(top_families) > 1 else []

        modality = self._detect_modality(phase1_data)

        top_scores_str = ", ".join([f"{f}={s:.2f}" for f, s in sorted_families[:3]])
        reasoning = (
            f"Primary modality: {modality}. "
            f"Top families: {top_scores_str}. "
            f"Primary={primary}, secondary={secondary}. "
            f"Source={source}, strategy={strategy}."
        )

        logger.info("Phase 2A: %s", reasoning)

        return {
            "families": top_families[: self.max_families],
            "primary": primary,
            "secondary": secondary[: self.max_families - 1],
            "reasoning": reasoning,
            "scores": {
                family: round(score, 4)
                for family, score in sorted_families[: self.max_families]
            },
            "source": source,
            "modality": modality,
        }

    def _detect_modality(self, phase1_data: Dict[str, Any]) -> str:
        """Detect the primary technical modality of the invention."""
        terms = phase1_data.get("terms", [])
        all_text = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )

        vision_keywords = [
            "image",
            "vision",
            "segmentation",
            "pixel",
            "optical",
            "camera",
            "visual",
        ]
        nlp_keywords = ["text", "language", "nlp", "chatbot", "dialog", "sentiment"]
        audio_keywords = ["speech", "voice", "audio", "sound", "acoustic"]
        mechanical_keywords = [
            "vehicle",
            "engine",
            "mechanical",
            "structural",
            "physical",
        ]

        counts = {
            "vision": sum(1 for kw in vision_keywords if kw in all_text),
            "nlp": sum(1 for kw in nlp_keywords if kw in all_text),
            "audio": sum(1 for kw in audio_keywords if kw in all_text),
            "mechanical": sum(1 for kw in mechanical_keywords if kw in all_text),
        }

        modality = max(counts.items(), key=lambda x: x[1])[0]
        return modality if counts[modality] > 0 else "general"

    def _get_top_terms(self, phase1_data: Dict[str, Any], n: int = 15) -> List[str]:
        """Extract top N terms by importance."""
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        if not terms:
            return []

        scored_terms = []
        for t in terms:
            if isinstance(t, dict):
                term = t.get("term", "").lower().strip()
                importance = t.get("importance", 5)
                if term and len(term) > 2:
                    scored_terms.append((term, importance))
            elif isinstance(t, str):
                term = t.lower().strip()
                if term and len(term) > 2:
                    scored_terms.append((term, 5))

        scored_terms.sort(key=lambda x: -x[1])
        return [term for term, _ in scored_terms[:n]]


def extract_3char_families(codes: List[str]) -> List[str]:
    """Extract unique 3-char families from CPC codes."""
    families = []
    seen = set()
    for code in codes:
        if len(code) >= 3:
            family = code[:3]
            if family not in seen:
                seen.add(family)
                families.append(family)
    return families
