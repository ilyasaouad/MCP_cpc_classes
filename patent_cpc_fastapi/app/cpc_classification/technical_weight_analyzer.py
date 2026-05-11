"""
technical_weight_analyzer.py - Technical Weight Analysis Engine

Processes Phase 1 output to determine the invention's technical character.
Used to guide CPC classification toward computational vs physical domain.

KEY ANALYSIS:
1. Segmented Text Extraction: CLAIMS, ABSTRACT, DETAILED_DESCRIPTION_OF_DRAWINGS
2. Bucket Classification: COMPUTATIONAL vs PHYSICAL/DOMAIN keywords
3. Drawing-First Multiplier: 1.5x weight for terms in DETAILED_DESCRIPTION
4. Technical Character Ratio (TCR): Computational / Physical weight

USAGE:
    analyzer = TechnicalWeightAnalyzer()
    result = analyzer.analyze(phase1_data)
    # Returns: {tcr, force_flag, analysis_details, ...}

TCR Thresholds:
- TCR > 2.0: FORCE_SOFTWARE_CORE (G06F/G06N primary)
- TCR < 0.5: FORCE_DOMAIN_CORE (A/B/C/F sections primary)
- 0.5 <= TCR <= 2.0: HYBRID_INVENTION (both layers)
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# COMPUTATIONAL keywords (The Tool)
COMPUTATIONAL_KEYWORDS = {
    "prompt",
    "llm",
    "large language model",
    "code",
    "syntax",
    "algorithm",
    "data structure",
    "executable",
    "software",
    "neural",
    "inference",
    "training",
    "sequence",
    "token",
    "embedding",
    "vector",
    "model",
    "network",
    "layer",
    "weight",
    "parameter",
    "gradient",
    "optimization",
    "loss",
    "function",
    "computation",
    "processing",
    "pipeline",
    "workflow",
    "orchestration",
    "middleware",
    "api",
    "interface",
    "query",
    "database",
    "graph",
    "rdf",
    "sparql",
    "semantic",
    "nlp",
    "nlu",
    "parser",
    "compiler",
    "runtime",
    "instruction",
    "bytecode",
    "script",
    "programming",
    "classification",
    "prediction",
    "regression",
    "clustering",
    "feature",
    "encoding",
    "decoding",
    "transformer",
    "attention",
    "softmax",
    "tensor",
    "gpu",
    "compute",
}

# PHYSICAL/DOMAIN keywords (The Target)
PHYSICAL_KEYWORDS = {
    "plc",
    "industrial plant",
    "vehicle",
    "motor",
    "chemical",
    "molecular",
    "pressure",
    "valve",
    "biological",
    "patient",
    "hardware",
    "circuit",
    "transistor",
    "semiconductor",
    "chip",
    "pcb",
    "sensor",
    "actuator",
    "mechanical",
    "engine",
    "pump",
    "turbine",
    "generator",
    "battery",
    "electrical",
    "electronic",
    "hydraulic",
    "pneumatic",
    "fluid",
    "pipe",
    "tank",
    "vessel",
    "reactor",
    "catalyst",
    "polymer",
    "compound",
    "formulation",
    "dosage",
    "therapeutic",
    "diagnosis",
    "surgical",
    "implant",
    "prosthetic",
    "therapeutic",
    "biomaterial",
    "pharmaceutical",
    "drug",
    "crop",
    "plant",
    "soil",
    "harvest",
    "livestock",
    "soil",
    "fertilizer",
    "irrigation",
    "climate",
    "weather",
    "temperature",
    "humidity",
    "material",
    "metal",
    "polymer",
    "composite",
    "ceramic",
    "alloy",
}


class TechnicalWeightAnalyzer:
    """
    Analyzes Phase 1 output to determine technical character.

    Extracts text from CLAIMS, ABSTRACT, and DETAILED_DESCRIPTION_OF_DRAWINGS.
    Classifies terms into COMPUTATIONAL vs PHYSICAL buckets.
    Applies Drawing-First Multiplier (1.5x) for DETAILED_DESCRIPTION terms.
    Calculates Technical Character Ratio (TCR).
    """

    def __init__(
        self,
        computational_keywords: Optional[set] = None,
        physical_keywords: Optional[set] = None,
        drawing_multiplier: float = 1.5,
    ):
        self.computational_keywords = computational_keywords or COMPUTATIONAL_KEYWORDS
        self.physical_keywords = physical_keywords or PHYSICAL_KEYWORDS
        self.drawing_multiplier = drawing_multiplier

    def analyze(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Phase 1 data to determine technical character.

        Args:
            phase1_data: Output from Phase 1 containing:
                - terms
                - essential_terms
                - technical_object
                - core_function
                - system_context

        Returns:
            Dict with:
                - tcr: Technical Character Ratio (float)
                - force_flag: FORCE_SOFTWARE_CORE | FORCE_DOMAIN_CORE | HYBRID_INVENTION
                - computational_weight: Total weight in computational bucket
                - physical_weight: Total weight in physical bucket
                - computational_terms: List of matched computational terms
                - physical_terms: List of matched physical terms
                - drawing_terms: Terms found in DETAILED_DESCRIPTION
                - analysis_details: Breakdown by section
        """
        # Extract text by section
        claims_text = self._extract_section_text(phase1_data, "claims")
        abstract_text = self._extract_section_text(phase1_data, "abstract")
        detailed_text = self._extract_section_text(phase1_data, "detailed")

        # Extract terms from all sections
        all_terms = self._extract_all_terms(phase1_data)

        # Classify terms into buckets
        computational_terms = []
        physical_terms = []
        drawing_terms = []

        for term_info in all_terms:
            term = term_info.get("term", "").lower()
            section = term_info.get("source_section", "unknown").lower()
            importance = term_info.get("importance", 5)

            # Check computational bucket
            if self._matches_keywords(term, self.computational_keywords):
                # Apply multiplier if from detailed description
                multiplier = self.drawing_multiplier if "detailed" in section else 1.0
                weighted_importance = importance * multiplier

                computational_terms.append(
                    {
                        "term": term,
                        "section": section,
                        "importance": importance,
                        "weighted_importance": weighted_importance,
                        "multiplier": multiplier,
                        "bucket": "computational",
                    }
                )

                if "detailed" in section:
                    drawing_terms.append(term)

            # Check physical bucket
            elif self._matches_keywords(term, self.physical_keywords):
                multiplier = self.drawing_multiplier if "detailed" in section else 1.0
                weighted_importance = importance * multiplier

                physical_terms.append(
                    {
                        "term": term,
                        "section": section,
                        "importance": importance,
                        "weighted_importance": weighted_importance,
                        "multiplier": multiplier,
                        "bucket": "physical",
                    }
                )

                if "detailed" in section:
                    drawing_terms.append(term)

        # Calculate weighted totals
        computational_weight = sum(
            t["weighted_importance"] for t in computational_terms
        )
        physical_weight = sum(t["weighted_importance"] for t in physical_terms)

        # Calculate TCR
        if physical_weight > 0:
            tcr = computational_weight / physical_weight
        else:
            # If no physical terms, assume computational
            tcr = computational_weight if computational_weight > 0 else 1.0

        # Determine force flag
        force_flag = self._determine_force_flag(tcr)

        # Analysis breakdown
        analysis_details = {
            "claims_computational": self._count_terms(computational_terms, "claims"),
            "claims_physical": self._count_terms(physical_terms, "claims"),
            "abstract_computational": self._count_terms(
                computational_terms, "abstract"
            ),
            "abstract_physical": self._count_terms(physical_terms, "abstract"),
            "detailed_computational": self._count_terms(
                computational_terms, "detailed"
            ),
            "detailed_physical": self._count_terms(physical_terms, "detailed"),
            "drawing_term_count": len(drawing_terms),
        }

        result = {
            "tcr": round(tcr, 3),
            "force_flag": force_flag,
            "computational_weight": round(computational_weight, 2),
            "physical_weight": round(physical_weight, 2),
            "computational_term_count": len(computational_terms),
            "physical_term_count": len(physical_terms),
            "computational_terms": computational_terms,
            "physical_terms": physical_terms,
            "drawing_terms": drawing_terms,
            "drawing_multiplier_applied": len(drawing_terms) > 0,
            "analysis_details": analysis_details,
            "dominant_bucket": "computational"
            if computational_weight > physical_weight
            else "physical",
        }

        logger.info(
            "Technical Weight Analysis: TCR=%.3f, Flag=%s, Comp=%.2f, Phys=%.2f",
            tcr,
            force_flag,
            computational_weight,
            physical_weight,
        )

        return result

    def _extract_section_text(
        self, phase1_data: Dict[str, Any], section_type: str
    ) -> str:
        """Extract text from specific section of Phase 1 data."""
        parts = []

        # Check terms for section source
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        for term_info in terms:
            section = term_info.get("source_section", "").lower()
            if section_type in section:
                term = term_info.get("term", "")
                if term:
                    parts.append(term)

        return " ".join(parts)

    def _extract_all_terms(self, phase1_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all terms from Phase 1 data with importance and source."""
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))

        result = []
        for term_info in terms:
            if isinstance(term_info, dict):
                result.append(
                    {
                        "term": term_info.get("term", ""),
                        "importance": term_info.get("importance", 5),
                        "source_section": term_info.get(
                            "source_section", term_info.get("source", "unknown")
                        ),
                    }
                )
            else:
                result.append(
                    {
                        "term": str(term_info),
                        "importance": 5,
                        "source_section": "unknown",
                    }
                )

        return result

    def _matches_keywords(self, term: str, keyword_set: set) -> bool:
        """Check if term matches any keyword in the set."""
        term = term.lower()
        for keyword in keyword_set:
            if keyword in term or term in keyword:
                return True
        return False

    def _count_terms(self, terms: List[Dict], section_type: str) -> int:
        """Count terms from specific section type."""
        count = 0
        for t in terms:
            if section_type in t.get("section", "").lower():
                count += 1
        return count

    def _determine_force_flag(self, tcr: float) -> str:
        """Determine the force flag based on TCR."""
        if tcr > 2.0:
            return "FORCE_SOFTWARE_CORE"
        elif tcr < 0.5:
            return "FORCE_DOMAIN_CORE"
        else:
            return "HYBRID_INVENTION"

    def get_recommendation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get CPC classification recommendation based on TCR analysis.

        Returns guidance for how to weight CPC families.
        """
        tcr = result.get("tcr", 1.0)
        force_flag = result.get("force_flag", "HYBRID_INVENTION")
        dominant = result.get("dominant_bucket", "computational")

        recommendations = {
            "primary_core_priority": [],
            "secondary_core_priority": [],
            "deprioritize": [],
        }

        if force_flag == "FORCE_SOFTWARE_CORE":
            recommendations["primary_core_priority"] = ["G06F", "G06N", "G06K", "G06Q"]
            recommendations["secondary_core_priority"] = ["G05B", "G10L"]
            recommendations["deprioritize"] = ["A61", "B23", "F16", "C08"]
        elif force_flag == "FORCE_DOMAIN_CORE":
            recommendations["primary_core_priority"] = [
                "A61",
                "B23",
                "B60",
                "F16",
                "C08",
                "E21",
            ]
            recommendations["secondary_core_priority"] = ["G06F", "G05B"]
            recommendations["deprioritize"] = ["G06N"]
        else:  # HYBRID_INVENTION
            recommendations["primary_core_priority"] = [
                "G06F",
                "G06N",
                dominant.upper(),
            ]
            recommendations["secondary_core_priority"] = ["G05B", "G10L"]
            recommendations["deprioritize"] = []

        recommendations["dominant_bucket"] = dominant
        recommendations["force_flag"] = force_flag
        recommendations["tcr"] = tcr

        return recommendations


def apply_technical_weight_analysis(
    phase1_data: Dict[str, Any],
    analyzer: Optional[TechnicalWeightAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Quick function to apply technical weight analysis.

    Args:
        phase1_data: Output from Phase 1
        analyzer: Optional pre-configured analyzer

    Returns:
        Analysis result with TCR and force flag
    """
    if analyzer is None:
        analyzer = TechnicalWeightAnalyzer()
    return analyzer.analyze(phase1_data)
