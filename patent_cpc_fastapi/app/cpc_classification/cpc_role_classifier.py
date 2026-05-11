"""
cpc_role_classifier.py - Phase 1.5: Invention Role Classifier

Determines the ROLE of the invention (CORE_TECH, SYSTEM, APPLICATION, SUPPORT).
This becomes the primary driver of CPC classification, overriding raw domain signals.

ROLE DEFINITIONS:
- CORE_TECH:  The invention modifies/improves the underlying technology itself
              (new algorithms, model architecture, training methods, signal processing)
- SYSTEM:     The invention orchestrates/coordinates/manages components
              (pipelines, multi-component systems, data/control flow)
- APPLICATION: The invention applies known technology to a specific domain
              (medical, automotive, finance, domain-specific deployment)
- SUPPORT:     Auxiliary functionality (logging, storage, UI, monitoring)

CRITICAL RULES:
1. Presence of AI/ML does NOT imply CORE_TECH
2. If AI is used WITHOUT modifying its internal structure → SYSTEM
3. If multiple components interact → strong bias toward SYSTEM
4. If no clear algorithmic or physical innovation → DO NOT select CORE_TECH

Usage:
    classifier = CPCRoleClassifier(llm)
    role_result = classifier.classify_role(phase1_output)
    # Returns: {"role": "SYSTEM", "confidence": 0.85, "reasoning": [...], "evidence": [...]}
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from search_core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a patent role classification expert.

Your task is to determine the INVENTION ROLE - the fundamental nature of what the invention does.

The invention role determines which CPC family should be PRIMARY:
- CORE_TECH inventions  → G06N, G06T, G06V, G10L (technology-native classes)
- SYSTEM inventions      → G06F, H04L, G05B (system orchestration classes)
- APPLICATION inventions → A, B, E, F sections (domain-specific classes)
- SUPPORT inventions    → G06F general computing

"""

ROLE_CLASSIFICATION_PROMPT = """
=== INVENTION PROFILE (from Phase 1) ===
Technical Object: {technical_object}
Problem Solved: {problem_solved}
Core Function: {core_function}
System Context: {system_context}
Classification Strategy: {strategy}

=== EXTRACTED TERMS ===
{terms_text}

=== TASK ===
Classify this invention into ONE primary role.

=== ROLE DEFINITIONS ===

1. CORE_TECH
   The invention modifies or improves the UNDERLYING TECHNOLOGY ITSELF.
   Includes:
   - new algorithms or computational methods
   - model architecture changes (new layer types, new network structures)
   - training methods (new optimization, new loss functions, new regularization)
   - signal/image/audio processing methods (new filters, new transformations)
   - physical/mechanical innovations (new mechanisms, new structures)
   
   Strong indicators:
   - "training", "weights", "parameters", "optimization"
   - "new method for processing..."
   - "improving accuracy/efficiency of the model/engine itself"
   - "neural network layer", "architecture", "model structure"
   - "compression algorithm", "encoding scheme"

   CRITICAL: AI/ML presence does NOT imply CORE_TECH!
   - "using an LLM to classify documents" = SYSTEM (AI is a tool)
   - "improving LLM architecture" = CORE_TECH (modifying the model)
   - "quantizing neural network weights" = CORE_TECH (modifying model internals)

2. SYSTEM
   The invention ORCHESTRATES, COORDINATES, or MANAGES components.
   Focus is on WORKFLOW, INTERACTION, or CONTROL LOGIC.
   Includes:
   - pipelines (receiving input → processing → forwarding → updating)
   - multi-component systems (multiple models, devices, or subsystems interacting)
   - data/control flow between modules
   - workflow management, orchestration, coordination
   - APIs, interfaces, protocols for component interaction
   
   Strong indicators:
   - "managing", "coordinating", "orchestrating"
   - "receiving input → processing → forwarding"
   - multiple components interacting
   - "pipeline", "workflow", "orchestration"
   - "interface", "API", "protocol", "data flow"

   CRITICAL: If AI is used WITHOUT modifying its internal structure → SYSTEM
   - "using LLM for classification" = SYSTEM (LLM is an external component)
   - "pipeline with multiple models" = SYSTEM (multiple components)

3. APPLICATION
   The invention APPLIES known technology to a SPECIFIC DOMAIN.
   Includes:
   - medical use (diagnosis, treatment, healthcare systems)
   - automotive systems (vehicle control, ADAS, electric vehicles)
   - finance/business applications (fraud detection, trading, banking)
   - domain-specific deployment
   - industry-specific language and context
   
   Strong indicators:
   - industry-specific language (medical, automotive, agricultural)
   - "applied to..." / "used for..." / "for use in..."
   - domain-specific terminology

4. SUPPORT
   Auxiliary functionality, NOT central to the technical operation.
   Includes:
   - logging, monitoring
   - storage, data management
   - UI, visualization
   - metadata handling
   - configuration, settings
   
   Strong indicators:
   - "logging", "monitoring", "tracking"
   - "storing", "caching", "buffering"
   - "user interface", "display", "visualization"

=== CRITICAL RULES ===

1. Presence of AI/ML (LLM, neural network) DOES NOT imply CORE_TECH
2. If AI is used WITHOUT modifying its internal structure → classify as SYSTEM
3. If multiple components interact → strong bias toward SYSTEM
4. If no clear algorithmic or physical innovation → DO NOT select CORE_TECH
5. If the invention is about USING technology for a domain → APPLICATION
6. If the invention is about MANAGING/ORCHESTRATING tech → SYSTEM
7. If the invention is about IMPROVING the technology itself → CORE_TECH

=== DECISION MATRIX ===

| What the invention does                        | Role       |
|------------------------------------------------|------------|
| New neural network architecture                | CORE_TECH  |
| New training method                            | CORE_TECH  |
| New signal processing algorithm                | CORE_TECH  |
| New image encoding/compression method          | CORE_TECH  |
| Pipeline coordinating multiple models          | SYSTEM     |
| Managing data flow between components           | SYSTEM     |
| Using AI for medical diagnosis                 | APPLICATION|
| Using AI for vehicle control                   | APPLICATION|
| Applying technology to a specific industry     | APPLICATION|
| Logging/monitoring system                      | SUPPORT    |
| UI for a technical system                      | SUPPORT    |
| Storing/caching data                          | SUPPORT    |

=== OUTPUT FORMAT (STRICT JSON - no markdown, no text outside) ===

{{
  "role": "CORE_TECH | SYSTEM | APPLICATION | SUPPORT",
  "confidence": 0.0-1.0,
  "reasoning": [
    "short, precise justification 1",
    "short, precise justification 2",
    "..."
  ],
  "evidence": [
    "quoted or paraphrased signals from Phase 1"
  ],
  "rejected_roles": [
    {{"role": "APPLICATION", "reason": "why it doesn't fit"}},
    ...
  ]
}}
"""


class CPCRoleClassifier:
    """
    Phase 1.5: Classifies invention into CORE_TECH / SYSTEM / APPLICATION / SUPPORT.

    This is the MOST IMPORTANT classification signal for CPC mapping.
    It overrides raw domain signals by determining the abstraction level.
    """

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm

    def classify_role(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the invention's role based on Phase 1 output.

        Args:
            phase1_data: Output from Phase 1 containing:
                - technical_object
                - problem_solved
                - core_function
                - system_context
                - classification_strategy
                - terms/essential_terms

        Returns:
            Dict with:
                - role: CORE_TECH | SYSTEM | APPLICATION | SUPPORT
                - confidence: 0.0-1.0
                - reasoning: list of justification strings
                - evidence: list of quoted/paraphrased signals
        """
        if not phase1_data:
            return self._default_result("SYSTEM", 0.5, ["No Phase 1 data available"])

        # Extract key fields
        technical_object = phase1_data.get("technical_object", "")
        problem_solved = phase1_data.get("problem_solved", "")
        core_function = phase1_data.get("core_function", "")
        system_context = phase1_data.get("system_context", "")
        strategy = phase1_data.get("classification_strategy", "")

        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        terms_text = self._format_terms(terms)

        # Build prompt
        prompt = ROLE_CLASSIFICATION_PROMPT.format(
            technical_object=technical_object,
            problem_solved=problem_solved,
            core_function=core_function,
            system_context=system_context,
            strategy=strategy,
            terms_text=terms_text,
        )

        # Use LLM if available
        if self.llm:
            try:
                response = self.llm.chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_message=prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
                result = self._parse_response(response)
                if result:
                    logger.info(
                        "Phase 1.5: Role=%s (conf=%.2f)",
                        result.get("role"),
                        result.get("confidence"),
                    )
                    return result
            except Exception as e:
                logger.warning("Phase 1.5 LLM classification failed: %s", e)

        # Fallback to rule-based classification
        return self._rule_based_classification(phase1_data)

    def _format_terms(self, terms: List[Any], max_terms: int = 15) -> str:
        """Format terms for prompt."""
        if not terms:
            return "No terms extracted"

        lines = []
        for t in terms[:max_terms]:
            if isinstance(t, dict):
                term = t.get("term", "")
                importance = t.get("importance", 5)
                source = t.get("source_section", t.get("source", "unknown"))
                lines.append(f"- {term} (importance: {importance}, source: {source})")
            elif isinstance(t, str):
                lines.append(f"- {t}")

        return "\n".join(lines) if lines else "No terms extracted"

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        if not response:
            return None

        try:
            return json.loads(response)
        except Exception:
            pass

        # Try stripping markdown
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Try regex extraction
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _rule_based_classification(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based role classification.

        Uses keyword patterns when LLM is unavailable.
        """
        technical_object = phase1_data.get("technical_object", "").lower()
        core_function = phase1_data.get("core_function", "").lower()
        system_context = phase1_data.get("system_context", "").lower()

        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms_text = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )

        combined_text = (
            f"{technical_object} {core_function} {system_context} {all_terms_text}"
        )

        scores = {
            "CORE_TECH": 0.0,
            "SYSTEM": 0.0,
            "APPLICATION": 0.0,
            "SUPPORT": 0.0,
        }

        # CORE_TECH signals
        core_tech_signals = [
            "algorithm",
            "architecture",
            "layer",
            "training method",
            "optimization method",
            "weight",
            "parameter",
            "quantization",
            "pruning",
            "distillation",
            "compression algorithm",
            "encoding scheme",
            "filter design",
            "model structure",
            "network architecture",
            "activation function",
            "loss function",
            "gradient",
            "backpropagation",
            "neural network layer",
            "improving the model",
            "modifying the model",
            "new neural network",
        ]
        for signal in core_tech_signals:
            if signal in combined_text:
                scores["CORE_TECH"] += 1.0

        # SYSTEM signals
        system_signals = [
            "pipeline",
            "orchestrating",
            "coordinating",
            "managing multiple",
            "receiving input",
            "processing",
            "forwarding",
            "updating",
            "workflow",
            "data flow",
            "control flow",
            "component interaction",
            "multi-component",
            "multi-model",
            "interface",
            "API",
            "protocol",
            "routing",
            "dispatching",
            "orchestration",
        ]
        for signal in system_signals:
            if signal in combined_text:
                scores["SYSTEM"] += 1.0

        # APPLICATION signals
        application_signals = [
            "medical",
            "healthcare",
            "diagnosis",
            "treatment",
            "patient",
            "automotive",
            "vehicle",
            "car",
            "driving",
            "ADAS",
            "agricultural",
            "farming",
            "crop",
            "financial",
            "banking",
            "trading",
            "fraud detection",
            "industrial",
            "manufacturing",
            "factory",
            "energy",
            "power grid",
            "renewable",
        ]
        for signal in application_signals:
            if signal in combined_text:
                scores["APPLICATION"] += 1.0

        # SUPPORT signals
        support_signals = [
            "logging",
            "monitoring",
            "tracking",
            "visualization",
            "user interface",
            "display",
            "dashboard",
            "UI",
            "storing",
            "caching",
            "buffering",
            "memory management",
            "configuration",
            "settings",
            "metadata",
        ]
        for signal in support_signals:
            if signal in combined_text:
                scores["SUPPORT"] += 1.0

        # Anti-CORE_TECH signals (these suggest SYSTEM or APPLICATION instead)
        anti_core_tech = [
            "using a neural network",
            "using an LLM",
            "using machine learning",
            "applying neural network",
            "deploying AI",
            "classification using",
            "detection using",
            "recognition using",
        ]
        for signal in anti_core_tech:
            if signal in combined_text:
                scores["CORE_TECH"] -= 1.5
                scores["SYSTEM"] += 0.5
                scores["APPLICATION"] += 0.5

        # Determine role
        best_role = max(scores.items(), key=lambda x: x[1])
        role = best_role[0]

        # If scores are too low, default to SYSTEM (most common)
        if best_role[1] < 0.5:
            role = "SYSTEM"
            confidence = 0.5
        else:
            # Normalize confidence
            total = sum(scores.values())
            if total > 0:
                confidence = min(best_role[1] / max(sum(scores.values()), 1), 1.0)
            else:
                confidence = 0.5

        reasoning = [f"Rule-based classification: {role} with score {best_role[1]:.2f}"]
        evidence = [f"Best matching signals for {role}"]

        logger.info("Phase 1.5 (rule-based): Role=%s (conf=%.2f)", role, confidence)

        return {
            "role": role,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "evidence": evidence,
        }

    def _default_result(
        self, role: str, confidence: float, reasoning: List[str]
    ) -> Dict[str, Any]:
        """Return default result when classification fails."""
        return {
            "role": role,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": [],
        }


def get_role_family_boosts(role: str) -> Dict[str, float]:
    """
    Get family boost multipliers based on role.

    This is used in Phase 2A scoring to adjust family weights.

    Args:
        role: CORE_TECH | SYSTEM | APPLICATION | SUPPORT

    Returns:
        Dict mapping CPC family prefixes to boost multipliers
    """
    if role == "CORE_TECH":
        # Boost technology-native classes
        return {
            "G06N": 1.5,  # AI/neural networks
            "G06T": 1.3,  # Image processing
            "G06V": 1.3,  # Vision
            "G10L": 1.3,  # Audio
            "G06K": 1.2,  # Pattern recognition
            "G06F": 0.7,  # Deprioritize general computing
            "H04L": 0.7,  # Deprioritize telecom
        }
    elif role == "SYSTEM":
        # Boost system orchestration classes
        return {
            "G06F": 1.5,  # Data processing/computing
            "H04L": 1.4,  # Data transmission/networking
            "G05B": 1.3,  # Control systems
            "G06N": 0.6,  # Deprioritize unless NN-internal signals
            "G06T": 0.7,  # Deprioritize unless image-native
            "G06V": 0.7,  # Deprioritize unless vision-native
        }
    elif role == "APPLICATION":
        # Boost domain application sections
        return {
            "A61": 1.5,  # Medical
            "B60": 1.4,  # Vehicles
            "B23": 1.3,  # Manufacturing
            "E21": 1.3,  # Mining/drilling
            "A01": 1.3,  # Agriculture
            "G06F": 0.8,  # Soft deprioritization
            "G06N": 0.7,  # AI is tool, not subject
        }
    elif role == "SUPPORT":
        # Boost general computing for auxiliary functions
        return {
            "G06F": 1.5,  # General computing
            "G06N": 0.5,  # Deprioritize
            "G06T": 0.5,
        }
    else:
        return {}


def apply_role_scoring(
    family_scores: Dict[str, float], role: str, phase1_data: Dict[str, Any]
) -> Dict[str, float]:
    """
    Apply role-based scoring adjustments to family scores.

    Args:
        family_scores: Current dict of family -> score
        role: The classified role
        phase1_data: Phase 1 data for additional context

    Returns:
        Adjusted family_scores dict
    """
    boosts = get_role_family_boosts(role)

    # Check for NN-internal signals (overrides role boost for G06N)
    terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
    all_terms_text = " ".join(
        [
            t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
            for t in terms
        ]
    )

    nn_internal_signals = [
        "weight clipping",
        "weight quantization",
        "model compression",
        "pruning",
        "distillation",
        "parameter optimization",
        "neural network architecture",
        "layer design",
    ]
    has_nn_internal = any(sig in all_terms_text for sig in nn_internal_signals)

    # If CORE_TECH with NN-internal signals, boost G06N
    if role == "CORE_TECH" and has_nn_internal:
        boosts["G06N"] = 2.0  # Strong boost
        logger.info("Role scoring: CORE_TECH + NN-internal → G06N boost 2.0x")

    # Apply boosts
    adjusted_scores = {}
    for family, score in family_scores.items():
        # Find matching boost
        boost = 1.0
        for prefix, multiplier in boosts.items():
            if family.startswith(prefix):
                boost = multiplier
                break

        # Special check for role != CORE_TECH but NN internal signals
        if role != "CORE_TECH" and has_nn_internal and family.startswith("G06N"):
            boost *= 0.5  # Penalize G06N if role is not CORE_TECH
            logger.info("Role scoring: %s role with NN-internal → G06N penalty", role)

        adjusted_scores[family] = score * boost

    return adjusted_scores
