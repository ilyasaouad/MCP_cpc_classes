"""
cpc_decision_tree.py - Phase 3.5: Decision Tree Constraint Layer

Applies deterministic, rule-based constraints to CPC candidates BEFORE hypothesis clustering.
Refines candidate scores by enforcing domain correctness and eliminating false positives.

DESIGN PRINCIPLES:
- CONSTRAINS scoring, does not replace it
- Deterministic rules override statistical noise
- Domain correctness > keyword similarity
- Prefer false negatives over false positives
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# Domain mapping: domain name -> CPC families
DOMAIN_TO_CPC = {
    "ai": ["G06N"],
    "machine learning": ["G06N"],
    "neural networks": ["G06N"],
    "deep learning": ["G06N"],
    "general computing": ["G06F"],
    "data processing": ["G06F"],
    "image": ["G06T"],
    "image processing": ["G06T"],
    "video": ["G06T"],
    "computer graphics": ["G06T"],
    "speech": ["G10L"],
    "audio": ["G10L"],
    "acoustics": ["G10K"],
    "telecom": ["H04L", "H04W"],
    "telecommunications": ["H04L", "H04W"],
    "networking": ["H04L"],
    "wireless": ["H04W"],
    "control": ["G05B"],
    "control systems": ["G05B"],
    "medical": ["A61B", "A61K"],
    "medical technology": ["A61B"],
    "pharmaceutical": ["A61K"],
    "measurement": ["G01"],
    "sensors": ["G01"],
    "mechanical": ["F16"],
    "mechanical engineering": ["F16"],
    "engines": ["F01", "F02", "F03", "F04"],
    "vehicles": ["B60"],
    "manufacturing": ["B23"],
    "chemistry": [
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
    ],
    "semiconductors": ["H01L"],
    "circuits": ["H01L", "H03"],
}

# Invalid class patterns per domain
INVALID_PATTERNS = {
    "ai": [
        ("G06F2207", 0.1),  # BCD/numeric encoding
        ("G06F7", 0.1),  # Digital computing
        ("G10K", 0.1),  # Acoustics
        ("G06T", 0.2),  # Image (unless image signals)
        ("G06V", 0.2),  # Vision (unless vision signals)
    ],
    "image": [
        ("G06N", 0.2),  # AI (unless AI signals)
        ("G10L", 0.1),  # Speech
    ],
    "telecom": [
        ("G06T", 0.1),  # Image
        ("G10L", 0.1),  # Speech
        ("A61", 0.1),  # Medical
    ],
    "mechanical": [
        ("G06N", 0.1),  # AI
        ("G06T", 0.1),  # Image
        ("H04L", 0.1),  # Telecom
    ],
}

# Functional boosting rules
FUNCTIONAL_BOOSTS = {
    "optimization": {"G06N3/045": 1.5, "G06N3/": 1.3},
    "parameter tuning": {"G06N3/045": 1.5, "G06N3/": 1.3},
    "clipping": {"G06N3/045": 1.5, "G06N3/": 1.3},
    "quantization": {"G06N3/063": 1.5, "G06N3/": 1.3},
    "compression": {"G06N3/063": 1.5, "G06N3/": 1.3},
    "low-bit": {"G06N3/063": 1.5, "G06N3/": 1.3},
    "transmission": {"H04L1/": 1.3, "H04L29/": 1.3, "H04L": 1.2},
    "communication": {"H04L1/": 1.3, "H04L29/": 1.3, "H04L": 1.2},
    "image processing": {"G06T5/": 1.3, "G06T7/": 1.3, "G06T": 1.2},
    "filtering": {"G06T5/": 1.3, "G06T": 1.2},
    "sealing": {"F16J": 1.5, "F16": 1.3},
    "valve": {"F16K": 1.5, "F16": 1.3},
    "medical imaging": {"A61B5/": 1.5, "A61B": 1.3},
    "diagnosis": {"A61B5/": 1.5, "A61B": 1.3},
}

# Object-aware disambiguation
OBJECT_DISAMBIGUATION = {
    "clipping": {
        "weight": {"boost": {"G06N": 1.5}, "penalize": {"G06T": 0.2, "G10L": 0.2}},
        "image": {"boost": {"G06T": 1.5}, "penalize": {"G06N": 0.2}},
        "audio": {"boost": {"G10L": 1.5}, "penalize": {"G06N": 0.2}},
    },
    "encoding": {
        "model": {"boost": {"G06N3/063": 1.5}, "penalize": {"G06F2207": 0.1}},
        "decimal": {"boost": {"G06F2207": 1.5}, "penalize": {"G06N": 0.2}},
        "BCD": {"boost": {"G06F2207": 1.5}, "penalize": {"G06N": 0.2}},
    },
    "filtering": {
        "signal": {"boost": {"H04L": 1.3}, "penalize": {"G06T": 0.3}},
        "image": {"boost": {"G06T5/": 1.5}, "penalize": {"H04L": 0.3}},
    },
    "mapping": {
        "memory": {"boost": {"G06F": 1.3}, "penalize": {"G06T": 0.3}},
        "image": {"boost": {"G06T": 1.3}, "penalize": {"G06F": 0.3}},
    },
}

# ── CPC HIERARCHY PRIORITY SYSTEM (UNIVERSAL) ──
#
# Within each domain, CPC subclasses have hierarchical priority:
#   Level 1 (highest): parameter optimization, structural modification
#   Level 2: compression, efficiency, algorithmic transformation
#   Level 3: general system operation, inference, usage
#   Level 4 (lowest): abstract reasoning, generic computation
#
# RULE: When Level 1-2 signals exist, boost Level 1-2 subclasses
#       and penalize Level 3-4 subclasses.

CPC_HIERARCHY = {
    # G06N: AI / Neural Networks
    "G06N": {
        1: {  # Level 1: Parameter optimization / structural modification
            "prefixes": ["G06N3/045", "G06N3/048", "G06N3/044"],
            "signals": [
                "parameter optimization",
                "weight clipping",
                "gradient clipping",
                "parameter tuning",
                "hyperparameter",
                "learning rate",
                "momentum",
                "weight decay",
                "regularization",
                "dropout",
                "batch normalization",
                "layer normalization",
                "weight initialization",
            ],
            "description": "Parameter optimization and structural model changes",
        },
        2: {  # Level 2: Compression / efficiency
            "prefixes": ["G06N3/063", "G06N3/065", "G06N3/067"],
            "signals": [
                "quantization",
                "model compression",
                "pruning",
                "distillation",
                "low-bit",
                "sparse",
                "sparsity",
                "compressed",
                "compact model",
                "knowledge distillation",
                "model pruning",
                "weight pruning",
                "activation quantization",
                "weight quantization",
                "int8",
                "fp16",
                "mixed precision",
                "model size reduction",
                "memory efficient",
            ],
            "description": "Model compression, quantization, and efficiency",
        },
        3: {  # Level 3: General NN structures / operation
            "prefixes": ["G06N3/08", "G06N3/02", "G06N3/098", "G06N3/084"],
            "signals": [
                "neural network architecture",
                "network structure",
                "layer design",
                "convolutional",
                "recurrent",
                "transformer",
                "attention mechanism",
                "feedforward",
                "backpropagation",
                "training",
                "general neural",
            ],
            "description": "General neural network structures and training",
        },
        4: {  # Level 4: Inference / reasoning / semantic
            "prefixes": ["G06N5/04", "G06N5/", "G06N7/", "G06N20/"],
            "signals": [
                "inference",
                "reasoning",
                "logical reasoning",
                "expert system",
                "knowledge base",
                "semantic",
                "ontology",
                "symbolic",
                "rule-based",
                "decision support",
                "cognitive",
                "interpretability",
                "explainable",
                "attention visualization",
                "model interpretation",
            ],
            "description": "Inference, reasoning, and semantic AI",
        },
    },
    # G06T: Image Processing
    "G06T": {
        1: {  # Level 1: Image structure / geometry
            "prefixes": ["G06T17/", "G06T19/", "G06T15/"],
            "signals": [
                "3D modeling",
                "geometric transformation",
                "mesh",
                "surface",
                "geometry",
                "shape",
                "structure",
                "reconstruction",
            ],
            "description": "Image geometry and structural processing",
        },
        2: {  # Level 2: Image enhancement / transformation
            "prefixes": ["G06T5/", "G06T3/"],
            "signals": [
                "filtering",
                "enhancement",
                "restoration",
                "denoising",
                "sharpening",
                "contrast",
                "brightness",
                "color correction",
                "image transformation",
                "morphing",
                "warping",
            ],
            "description": "Image enhancement and transformation",
        },
        3: {  # Level 3: Image analysis / recognition
            "prefixes": ["G06T7/", "G06T9/"],
            "signals": [
                "segmentation",
                "edge detection",
                "feature extraction",
                "pattern recognition",
                "classification",
                "object detection",
                "image analysis",
                "texture analysis",
            ],
            "description": "Image analysis and feature extraction",
        },
        4: {  # Level 4: General image processing / rendering
            "prefixes": ["G06T11/", "G06T13/", "G06T15/"],
            "signals": [
                "rendering",
                "animation",
                "graphics",
                "display",
                "visualization",
                "general image",
                "computer graphics",
            ],
            "description": "Rendering and general graphics",
        },
    },
    # H04L: Telecommunications
    "H04L": {
        1: {  # Level 1: Protocol / architecture
            "prefixes": ["H04L29/", "H04L45/", "H04L47/"],
            "signals": [
                "protocol",
                "network architecture",
                "routing",
                "switching",
                "packet",
                "frame structure",
                "header",
                "control plane",
                "network topology",
                "topology",
            ],
            "description": "Network protocols and architecture",
        },
        2: {  # Level 2: Error correction / reliability
            "prefixes": ["H04L1/", "H04L25/"],
            "signals": [
                "error correction",
                "channel coding",
                "forward error",
                "redundancy",
                "retransmission",
                "ARQ",
                "FEC",
                "reliability",
                "error detection",
                "checksum",
            ],
            "description": "Error correction and channel coding",
        },
        3: {  # Level 3: Transmission / modulation
            "prefixes": ["H04L27/", "H04L25/"],
            "signals": [
                "modulation",
                "demodulation",
                "transmission",
                "signal",
                "carrier",
                "frequency",
                "bandwidth",
                "spectrum",
            ],
            "description": "Signal transmission and modulation",
        },
        4: {  # Level 4: General communication / security
            "prefixes": ["H04L9/", "H04L67/", "H04L69/"],
            "signals": [
                "security",
                "encryption",
                "authentication",
                "general communication",
                "application layer",
                "service",
                "general network",
            ],
            "description": "Security and general communication services",
        },
    },
    # F16: Mechanical Engineering
    "F16": {
        1: {  # Level 1: Structural components
            "prefixes": ["F16J", "F16K", "F16L", "F16M"],
            "signals": [
                "seal",
                "sealing",
                "gasket",
                "valve",
                "pipe",
                "fitting",
                "connector",
                "fastener",
                "bearing",
                "structural",
                "mechanical joint",
                "coupling",
            ],
            "description": "Mechanical structural components",
        },
        2: {  # Level 2: Mechanical systems / mechanisms
            "prefixes": ["F16H", "F16D", "F16F"],
            "signals": [
                "gear",
                "transmission",
                "clutch",
                "brake",
                "suspension",
                "damper",
                "spring",
                "mechanism",
                "kinematic",
            ],
            "description": "Mechanical systems and mechanisms",
        },
        3: {  # Level 3: General mechanical
            "prefixes": ["F16B", "F16C"],
            "signals": [
                "general mechanical",
                "assembly",
                "mounting",
                "support",
                "housing",
                "frame",
                "general structure",
            ],
            "description": "General mechanical engineering",
        },
        4: {  # Level 4: Abstract / generic
            "prefixes": ["F16P", "F16S"],
            "signals": [
                "safety",
                "monitoring",
                "abstract",
                "generic mechanical",
            ],
            "description": "Safety and generic mechanical concepts",
        },
    },
    # A61: Medical
    "A61": {
        1: {  # Level 1: Diagnostic / therapeutic devices
            "prefixes": ["A61B5/", "A61B8/", "A61B17/"],
            "signals": [
                "diagnosis",
                "diagnostic",
                "imaging",
                "MRI",
                "CT",
                "ultrasound",
                "surgical",
                "therapy",
                "treatment",
                "intervention",
                "medical device",
                "biopsy",
                "endoscopy",
            ],
            "description": "Diagnostic and therapeutic devices",
        },
        2: {  # Level 2: Monitoring / sensors
            "prefixes": ["A61B5/", "A61B5/145"],
            "signals": [
                "monitoring",
                "vital signs",
                "heart rate",
                "blood pressure",
                "glucose",
                "sensor",
                "wearable",
                "patient monitoring",
            ],
            "description": "Patient monitoring and sensors",
        },
        3: {  # Level 3: General medical
            "prefixes": ["A61B1/", "A61B90/"],
            "signals": [
                "general medical",
                "examination",
                "patient",
                "clinical",
            ],
            "description": "General medical examination",
        },
        4: {  # Level 4: Pharmaceutical / biochemical
            "prefixes": ["A61K", "A61P"],
            "signals": [
                "drug",
                "pharmaceutical",
                "medication",
                "vaccine",
                "compound",
                "biochemical",
                "therapeutic agent",
            ],
            "description": "Pharmaceutical and therapeutic agents",
        },
    },
    # G06F: Computing
    "G06F": {
        1: {  # Level 1: System architecture / hardware
            "prefixes": ["G06F9/", "G06F12/", "G06F13/"],
            "signals": [
                "processor",
                "CPU",
                "GPU",
                "memory",
                "cache",
                "bus",
                "architecture",
                "hardware",
                "instruction set",
                "pipeline",
                "multiprocessor",
                "multicore",
                "parallel processing",
            ],
            "description": "Computer architecture and hardware",
        },
        2: {  # Level 2: Software / OS / compilation
            "prefixes": ["G06F8/", "G06F9/", "G06F11/"],
            "signals": [
                "compiler",
                "operating system",
                "scheduler",
                "virtual memory",
                "software",
                "program",
                "application",
                "debugging",
                "testing",
                "verification",
                "fault tolerance",
            ],
            "description": "Software and operating systems",
        },
        3: {  # Level 3: Data processing / databases
            "prefixes": ["G06F16/", "G06F17/", "G06F21/"],
            "signals": [
                "database",
                "data processing",
                "search",
                "retrieval",
                "information",
                "text processing",
                "security",
                "access control",
            ],
            "description": "Data processing and information retrieval",
        },
        4: {  # Level 4: General computing / input-output
            "prefixes": ["G06F3/", "G06F1/"],
            "signals": [
                "input",
                "output",
                "display",
                "keyboard",
                "mouse",
                "interface",
                "general computing",
                "peripheral",
            ],
            "description": "General computing and interfaces",
        },
    },
}

# Hierarchy level multipliers
HIERARCHY_BOOST = {
    1: 2.0,  # Strongest boost for Level 1
    2: 1.5,  # Moderate boost for Level 2
    3: 1.0,  # Neutral for Level 3
    4: 0.4,  # Penalty for Level 4
}

# ── TECHNICAL CONTRIBUTION TYPE MAP (FINE-GRAINED) ──
#
# Maps specific technical operations to their EXACT CPC subgroups.
# This prevents confusion between subgroups at the same coarse level.
#
# Example: G06N3/063 (quantization) vs G06N3/043 (fuzzy logic)
# Both are G06N3/ but completely different technical contributions.

TECHNICAL_CONTRIBUTION_MAP = {
    # G06N: Neural Networks - Fine-grained contribution types
    "G06N": {
        "parameter_optimization": {
            "codes": ["G06N3/045", "G06N3/048", "G06N3/044"],
            "signals": [
                "parameter optimization",
                "weight clipping",
                "gradient clipping",
                "parameter tuning",
                "hyperparameter",
                "learning rate",
                "momentum",
                "weight decay",
                "regularization",
                "dropout",
                "batch normalization",
                "layer normalization",
                "weight initialization",
                "optimizer",
                "SGD",
                "Adam",
                "RMSprop",
                "gradient descent",
            ],
            "description": "Parameter optimization and training modification",
        },
        "model_compression": {
            "codes": ["G06N3/063", "G06N3/065", "G06N3/067"],
            "signals": [
                "quantization",
                "model compression",
                "pruning",
                "distillation",
                "low-bit",
                "sparse",
                "sparsity",
                "compressed",
                "compact model",
                "knowledge distillation",
                "model pruning",
                "weight pruning",
                "activation quantization",
                "weight quantization",
                "int8",
                "fp16",
                "mixed precision",
                "model size reduction",
                "memory efficient",
                "tensor decomposition",
                "low rank",
                "bitwidth reduction",
            ],
            "description": "Model compression and quantization",
        },
        "fuzzy_logic": {
            "codes": ["G06N3/043", "G06N3/042", "G06N3/046"],
            "signals": [
                "fuzzy logic",
                "fuzzy set",
                "fuzzy inference",
                "neuro-fuzzy",
                "membership function",
                "fuzzy controller",
                "fuzzy system",
                "linguistic variable",
                "fuzzy rule",
            ],
            "description": "Fuzzy logic and neuro-fuzzy systems",
        },
        "general_nn": {
            "codes": ["G06N3/08", "G06N3/02", "G06N3/098", "G06N3/084"],
            "signals": [
                "neural network architecture",
                "network structure",
                "layer design",
                "convolutional",
                "recurrent",
                "LSTM",
                "GRU",
                "feedforward",
                "backpropagation",
                "general neural",
                "network topology",
            ],
            "description": "General neural network structures",
        },
        "inference_reasoning": {
            "codes": ["G06N5/04", "G06N5/02", "G06N5/046", "G06N5/048"],
            "signals": [
                "inference",
                "reasoning",
                "logical reasoning",
                "expert system",
                "knowledge base",
                "rule-based",
                "decision support",
                "forward chaining",
                "backward chaining",
                "inference engine",
            ],
            "description": "Inference and reasoning systems",
        },
        "semantic_cognitive": {
            "codes": ["G06N5/08", "G06N5/06", "G06N7/"],
            "signals": [
                "semantic",
                "ontology",
                "cognitive",
                "knowledge representation",
                "conceptual",
                "symbolic AI",
                "cognitive model",
                "mental model",
            ],
            "description": "Semantic and cognitive systems",
        },
    },
    # G06T: Image Processing - Fine-grained
    "G06T": {
        "image_filtering": {
            "codes": ["G06T5/", "G06T5/00", "G06T5/20", "G06T5/10"],
            "signals": [
                "image filtering",
                "denoising",
                "smoothing",
                "sharpening",
                "edge enhancement",
                "morphological",
                "filter kernel",
            ],
            "description": "Image filtering and enhancement",
        },
        "image_analysis": {
            "codes": ["G06T7/", "G06T7/00", "G06T7/10", "G06T7/20"],
            "signals": [
                "image analysis",
                "feature extraction",
                "edge detection",
                "texture analysis",
                "pattern recognition",
                "segmentation",
            ],
            "description": "Image analysis and feature extraction",
        },
        "rendering_graphics": {
            "codes": ["G06T15/", "G06T15/00", "G06T15/10", "G06T15/20"],
            "signals": [
                "rendering",
                "graphics",
                "3D rendering",
                "ray tracing",
                "shading",
                "illumination",
                "computer graphics",
            ],
            "description": "Rendering and computer graphics",
        },
        "image_geometry": {
            "codes": ["G06T17/", "G06T19/", "G06T17/00"],
            "signals": [
                "3D modeling",
                "geometric transformation",
                "mesh",
                "surface",
                "geometry",
                "shape",
                "reconstruction",
                "point cloud",
            ],
            "description": "Image geometry and 3D processing",
        },
    },
    # H04L: Telecom - Fine-grained
    "H04L": {
        "error_correction": {
            "codes": ["H04L1/", "H04L1/00", "H04L1/16", "H04L1/18"],
            "signals": [
                "error correction",
                "channel coding",
                "forward error correction",
                "FEC",
                "ARQ",
                "redundancy",
                "checksum",
                "parity",
            ],
            "description": "Error correction and channel coding",
        },
        "protocol_architecture": {
            "codes": ["H04L29/", "H04L45/", "H04L47/", "H04L29/06"],
            "signals": [
                "protocol",
                "network architecture",
                "routing",
                "switching",
                "packet",
                "frame structure",
                "control plane",
                "topology",
            ],
            "description": "Network protocols and architecture",
        },
        "transmission_signal": {
            "codes": ["H04L27/", "H04L25/", "H04L27/00"],
            "signals": [
                "modulation",
                "demodulation",
                "transmission",
                "signal processing",
                "carrier",
                "frequency",
                "bandwidth",
                "spectrum",
            ],
            "description": "Signal transmission and modulation",
        },
        "security_services": {
            "codes": ["H04L9/", "H04L67/", "H04L9/00"],
            "signals": [
                "security",
                "encryption",
                "authentication",
                "privacy",
                "service",
                "application layer",
                "general network",
            ],
            "description": "Security and network services",
        },
    },
    # F16: Mechanical - Fine-grained
    "F16": {
        "sealing": {
            "codes": ["F16J", "F16J15/", "F16J15/00"],
            "signals": [
                "seal",
                "sealing",
                "gasket",
                "packing",
                "O-ring",
                "compression seal",
                "mechanical seal",
                "leak prevention",
            ],
            "description": "Sealing and gaskets",
        },
        "valves": {
            "codes": ["F16K", "F16K1/", "F16K15/", "F16K31/"],
            "signals": [
                "valve",
                "control valve",
                "check valve",
                "solenoid valve",
                "gate valve",
                "ball valve",
                "flow control",
            ],
            "description": "Valves and flow control",
        },
        "transmission": {
            "codes": ["F16H", "F16H1/", "F16H37/", "F16H55/"],
            "signals": [
                "gear",
                "transmission",
                "gearbox",
                "speed reducer",
                "chain drive",
                "belt drive",
                "power transmission",
            ],
            "description": "Mechanical transmission",
        },
        "general_mechanical": {
            "codes": ["F16B", "F16C", "F16D", "F16F"],
            "signals": [
                "fastener",
                "bearing",
                "clutch",
                "brake",
                "spring",
                "damper",
                "general mechanical",
                "assembly",
            ],
            "description": "General mechanical components",
        },
    },
}


class CPCDecisionTreeConstraint:
    """
    Phase 3.5: Decision Tree Constraint Layer.

    Applies deterministic, rule-based constraints to CPC candidates.
    """

    def __init__(self, boost_multiplier: float = 2.0, penalty_multiplier: float = 0.3):
        self.boost_multiplier = boost_multiplier
        self.penalty_multiplier = penalty_multiplier
        self.rules_log: List[Dict[str, Any]] = []

    def apply_constraints(
        self,
        ranked_candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply decision tree constraints to Phase 3 candidates.

        Returns:
            Dict with adjusted candidates, rules log, and domain decision.
        """
        if not ranked_candidates:
            return self._empty_result()

        self.rules_log = []
        candidates = [dict(c) for c in ranked_candidates]  # Copy

        # Step 1: Detect primary domain
        primary_domain, domain_confidence = self._detect_primary_domain(phase1_data)

        # Step 2: Domain dominance constraint
        if primary_domain and domain_confidence >= 0.6:
            candidates = self._apply_domain_dominance(
                candidates, primary_domain, phase1_data
            )

        # Step 3: Object-aware disambiguation
        candidates = self._apply_disambiguation(candidates, phase1_data)

        # Step 4: Functional boosting
        candidates = self._apply_functional_boosting(candidates, phase1_data)

        # Step 4.5: Hierarchy priority (CRITICAL - prevents intra-domain drift)
        if primary_domain:
            candidates = self._apply_hierarchy_priority(
                candidates, phase1_data, primary_domain
            )

        # Step 4.6: Fine-grained contribution filter (CRITICAL - prevents fuzzy logic when quantization detected)
        if primary_domain:
            candidates = self._apply_contribution_filter(
                candidates, phase1_data, primary_domain
            )

        # Step 5: Invalid class filtering
        if primary_domain:
            candidates = self._apply_invalid_filtering(candidates, primary_domain)

        # Step 6: Multi-domain support
        candidates = self._handle_multi_domain(candidates, phase1_data)

        # Step 7: Normalization
        candidates = self._normalize_scores(candidates)

        # Keep top 10
        candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[
            :10
        ]

        return {
            "phase35_candidates": candidates,
            "phase35_rules_log": self.rules_log,
            "phase35_domain": primary_domain,
            "phase35_domain_confidence": domain_confidence,
            "phase35_adjustments": len(self.rules_log),
        }

    def _detect_primary_domain(self, phase1_data: Dict[str, Any]) -> tuple:
        """Detect primary domain from Phase 1 data."""
        # Check primary_domain field first
        primary_domain = phase1_data.get("primary_domain", {})
        if primary_domain and isinstance(primary_domain, dict):
            pd_name = primary_domain.get("name", "").lower()
            pd_conf = primary_domain.get("confidence", 0)

            # Map to canonical domain
            for domain_name, families in DOMAIN_TO_CPC.items():
                if domain_name in pd_name or pd_name in domain_name:
                    return domain_name, pd_conf

        # Check domain signals
        domain_signals = phase1_data.get("domain_signals", [])
        if domain_signals:
            best_signal = None
            best_conf = 0
            for signal in domain_signals:
                if isinstance(signal, dict):
                    role = signal.get("role", "").lower()
                    conf = signal.get("confidence", 0)
                    name = signal.get("name", "").lower()

                    if role == "primary" and conf > best_conf:
                        best_conf = conf
                        best_signal = name
                    elif conf > best_conf and role != "negative":
                        best_conf = conf
                        best_signal = name

            if best_signal and best_conf >= 0.6:
                for domain_name, families in DOMAIN_TO_CPC.items():
                    if domain_name in best_signal or best_signal in domain_name:
                        return domain_name, best_conf

        # Check terms
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )

        domain_counts = {}
        for domain_name, families in DOMAIN_TO_CPC.items():
            count = sum(1 for kw in domain_name.split() if kw in all_terms)
            if count > 0:
                domain_counts[domain_name] = count

        if domain_counts:
            best_domain = max(domain_counts.items(), key=lambda x: x[1])
            return best_domain[0], min(best_domain[1] / 3, 0.9)

        return None, 0.0

    def _apply_domain_dominance(
        self,
        candidates: List[Dict[str, Any]],
        primary_domain: str,
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Boost candidates in primary domain, penalize unrelated."""
        target_families = DOMAIN_TO_CPC.get(primary_domain, [])
        if not target_families:
            return candidates

        # Check system context for explicit multi-domain
        system_context = phase1_data.get("system_context", "").lower()

        # Find families explicitly mentioned in system context
        context_families = set()
        for domain_name, families in DOMAIN_TO_CPC.items():
            if domain_name in system_context:
                context_families.update(families)

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            # Check if candidate is in target domain
            in_target = any(symbol.startswith(fam) for fam in target_families)
            in_context = any(symbol.startswith(fam) for fam in context_families)

            if in_target:
                # Boost
                old_score = score
                score *= self.boost_multiplier
                self._log_rule(
                    "DOMAIN_BOOST",
                    symbol,
                    old_score,
                    score,
                    f"Primary domain '{primary_domain}' boost",
                )
            elif in_context:
                # Context family - reduced penalty
                old_score = score
                score *= 0.6
                self._log_rule(
                    "CONTEXT_FAMILY",
                    symbol,
                    old_score,
                    score,
                    f"Context family (system context mentions related domain)",
                )
            else:
                # Penalize unrelated
                old_score = score
                score *= self.penalty_multiplier
                self._log_rule(
                    "DOMAIN_PENALTY",
                    symbol,
                    old_score,
                    score,
                    f"Unrelated to primary domain '{primary_domain}'",
                )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _apply_disambiguation(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply object-aware disambiguation to ambiguous terms."""
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            # Check each ambiguous term
            for term, contexts in OBJECT_DISAMBIGUATION.items():
                if term in all_terms:
                    # Find matching context
                    matched_context = None
                    for context, rules in contexts.items():
                        if context in all_terms:
                            matched_context = rules
                            break

                    if matched_context:
                        # Apply boost if candidate matches
                        boost_families = matched_context.get("boost", {})
                        for fam, multiplier in boost_families.items():
                            if symbol.startswith(fam):
                                old_score = score
                                score *= multiplier
                                self._log_rule(
                                    "DISAMBIGUATION_BOOST",
                                    symbol,
                                    old_score,
                                    score,
                                    f"'{term}' with '{list(contexts.keys())[0]}' context → {fam}",
                                )

                        # Apply penalty if candidate should be avoided
                        penalize_families = matched_context.get("penalize", {})
                        for fam, multiplier in penalize_families.items():
                            if symbol.startswith(fam):
                                old_score = score
                                score *= multiplier
                                self._log_rule(
                                    "DISAMBIGUATION_PENALTY",
                                    symbol,
                                    old_score,
                                    score,
                                    f"'{term}' does NOT match {fam} context",
                                )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _apply_functional_boosting(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Boost candidates matching core function."""
        core_function = phase1_data.get("core_function", "").lower()

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            for func_keyword, boosts in FUNCTIONAL_BOOSTS.items():
                if func_keyword in core_function:
                    for prefix, multiplier in boosts.items():
                        if symbol.startswith(prefix):
                            old_score = score
                            score *= multiplier
                            self._log_rule(
                                "FUNCTIONAL_BOOST",
                                symbol,
                                old_score,
                                score,
                                f"Core function '{func_keyword}' matches {prefix}",
                            )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _apply_hierarchy_priority(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        primary_domain: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply CPC hierarchy priority to prevent intra-domain drift.

        Within a domain, prefer Level 1-2 (structural/optimization) over
        Level 3-4 (operational/semantic) when structural signals exist.
        """
        # Get all text for analysis
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )
        core_function = phase1_data.get("core_function", "").lower()
        all_text = f"{all_terms} {core_function}"

        # Find the matching CPC family for this domain
        target_families = DOMAIN_TO_CPC.get(primary_domain, [])
        if not target_families:
            return candidates

        # Find the hierarchy definition for the target families
        hierarchy = None
        for fam in target_families:
            if fam in CPC_HIERARCHY:
                hierarchy = CPC_HIERARCHY[fam]
                break

        if not hierarchy:
            return candidates

        # Detect which hierarchy levels have signals
        detected_levels = set()
        level_signals = {}
        for level, config in hierarchy.items():
            signals = config.get("signals", [])
            for signal in signals:
                if signal in all_text:
                    detected_levels.add(level)
                    level_signals.setdefault(level, []).append(signal)

        if not detected_levels:
            logger.info("No hierarchy signals detected for domain %s", primary_domain)
            return candidates

        # Determine max detected level (prefer lower levels = higher priority)
        max_detected_level = min(detected_levels)  # Level 1 is highest priority

        logger.info(
            "Hierarchy priority: domain=%s, detected_levels=%s, max_priority=%d",
            primary_domain,
            sorted(detected_levels),
            max_detected_level,
        )

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            # Find which hierarchy level this candidate belongs to
            candidate_level = None
            for level, config in hierarchy.items():
                prefixes = config.get("prefixes", [])
                for prefix in prefixes:
                    if symbol.startswith(prefix):
                        candidate_level = level
                        break
                if candidate_level is not None:
                    break

            if candidate_level is None:
                # Candidate doesn't match any hierarchy level - neutral
                adjusted.append(candidate)
                continue

            if candidate_level == max_detected_level:
                # Candidate is at the highest detected priority level - boost
                old_score = score
                score *= HIERARCHY_BOOST[candidate_level]
                self._log_rule(
                    "HIERARCHY_BOOST",
                    symbol,
                    old_score,
                    score,
                    f"Level {candidate_level} matches detected structural signals: {level_signals.get(max_detected_level, [])[:3]}",
                )
            elif candidate_level < max_detected_level:
                # Candidate is at a HIGHER priority level than detected (shouldn't happen)
                pass
            else:
                # Candidate is at a LOWER priority level than detected - penalize
                # e.g., detected Level 1 (optimization) but candidate is Level 4 (inference)
                old_score = score
                score *= HIERARCHY_BOOST[candidate_level]
                self._log_rule(
                    "HIERARCHY_PENALTY",
                    symbol,
                    old_score,
                    score,
                    f"Level {candidate_level} (lower priority) when Level {max_detected_level} signals exist: {level_signals.get(max_detected_level, [])[:3]}",
                )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _apply_contribution_filter(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        primary_domain: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply fine-grained technical contribution filter.

        Prevents selection of subgroups that are in the correct domain
        but represent a completely different technical contribution.

        Example: G06N3/063 (quantization) vs G06N3/043 (fuzzy logic)
        Both are G06N3/ but completely different contributions.
        """
        # Map canonical domain name to CPC family codes
        target_families = DOMAIN_TO_CPC.get(primary_domain, [])
        if not target_families:
            return candidates

        # Get contribution map for the first matching CPC family
        contribution_map = {}
        for family in target_families:
            if family in TECHNICAL_CONTRIBUTION_MAP:
                contribution_map = TECHNICAL_CONTRIBUTION_MAP[family]
                break

        if not contribution_map:
            return candidates

        # Get all text for analysis
        terms = phase1_data.get("terms", phase1_data.get("essential_terms", []))
        all_terms = " ".join(
            [
                t.get("term", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in terms
            ]
        )
        core_function = phase1_data.get("core_function", "").lower()
        all_text = f"{all_terms} {core_function}"

        # Detect which contribution types are present
        detected_contributions = {}
        for contrib_name, config in contribution_map.items():
            signals = config.get("signals", [])
            matched_signals = [s for s in signals if s in all_text]
            if matched_signals:
                detected_contributions[contrib_name] = {
                    "signals": matched_signals,
                    "codes": config.get("codes", []),
                    "priority": self._get_contribution_priority(contrib_name),
                }

        if not detected_contributions:
            logger.info(
                "No specific contribution signals detected for domain %s",
                primary_domain,
            )
            return candidates

        # Find the highest priority level (lowest number)
        min_priority = min(d[1]["priority"] for d in detected_contributions.items())

        # Collect ALL contribution types at the highest priority level
        top_contributions = {
            name: data
            for name, data in detected_contributions.items()
            if data["priority"] == min_priority
        }

        # Collect all codes from top-priority contributions
        top_codes = []
        top_names = []
        for name, data in top_contributions.items():
            top_codes.extend(data["codes"])
            top_names.append(name)

        logger.info(
            "Contribution filter: domain=%s, top_priority=%d, contributions=%s",
            primary_domain,
            min_priority,
            top_names,
        )

        # Get all contribution code prefixes for this domain
        all_contrib_codes = set()
        for config in contribution_map.values():
            all_contrib_codes.update(config.get("codes", []))

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            # Check if candidate matches ANY top-priority contribution
            matches_top = any(symbol.startswith(code) for code in top_codes)

            # Check if candidate matches ANY contribution type in this domain
            matches_any_contrib = any(
                symbol.startswith(code) for code in all_contrib_codes
            )

            # Check if candidate matches a lower-priority contribution
            matches_lower = False
            lower_name = ""
            if matches_any_contrib and not matches_top:
                for name, data in detected_contributions.items():
                    if data["priority"] > min_priority:
                        for code in data["codes"]:
                            if symbol.startswith(code):
                                matches_lower = True
                                lower_name = name
                                break
                    if matches_lower:
                        break

            if matches_top:
                # Strong boost for matching top-priority contribution
                old_score = score
                score *= 2.5
                self._log_rule(
                    "CONTRIBUTION_MATCH",
                    symbol,
                    old_score,
                    score,
                    f"Matches top-priority contribution: {top_names[:2]}",
                )
            elif matches_any_contrib:
                # Candidate matches a DIFFERENT contribution type - heavy penalty
                # e.g., fuzzy logic candidate when quantization is detected
                old_score = score
                score *= 0.15  # Strong penalty
                contrib_name = self._get_contribution_name_for_code(
                    symbol, contribution_map
                )
                self._log_rule(
                    "CONTRIBUTION_MISMATCH",
                    symbol,
                    old_score,
                    score,
                    f"Wrong contribution type: expected {top_names[:2]}, got '{contrib_name}'",
                )
            # If candidate doesn't match any fine-grained contribution, leave as-is

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _get_contribution_priority(self, contrib_name: str) -> int:
        """Get priority level for a contribution type (lower = higher priority)."""
        priorities = {
            # G06N
            "parameter_optimization": 1,
            "model_compression": 1,
            "general_nn": 3,
            "inference_reasoning": 4,
            "semantic_cognitive": 4,
            "fuzzy_logic": 4,
            # G06T
            "image_geometry": 1,
            "image_filtering": 2,
            "image_analysis": 3,
            "rendering_graphics": 4,
            # H04L
            "protocol_architecture": 1,
            "error_correction": 2,
            "transmission_signal": 3,
            "security_services": 4,
            # F16
            "sealing": 1,
            "valves": 1,
            "transmission": 2,
            "general_mechanical": 3,
        }
        return priorities.get(contrib_name, 99)

    def _get_contribution_name_for_code(
        self, symbol: str, contribution_map: Dict[str, Any]
    ) -> str:
        """Find which contribution type a code belongs to."""
        for name, config in contribution_map.items():
            for code in config.get("codes", []):
                if symbol.startswith(code):
                    return name
        return "unknown"

    def _apply_invalid_filtering(
        self,
        candidates: List[Dict[str, Any]],
        primary_domain: str,
    ) -> List[Dict[str, Any]]:
        """Filter invalid classes for the primary domain."""
        invalid_patterns = INVALID_PATTERNS.get(primary_domain, [])
        if not invalid_patterns:
            return candidates

        adjusted = []
        for candidate in candidates:
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)

            for pattern, penalty in invalid_patterns:
                if symbol.startswith(pattern):
                    old_score = score
                    score *= penalty
                    self._log_rule(
                        "INVALID_FILTER",
                        symbol,
                        old_score,
                        score,
                        f"Invalid for domain '{primary_domain}': {pattern}*",
                    )

            candidate["score"] = round(score, 4)
            adjusted.append(candidate)

        return adjusted

    def _handle_multi_domain(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Handle multi-domain inventions."""
        domain_signals = phase1_data.get("domain_signals", [])
        primary_domains = []

        for signal in domain_signals:
            if isinstance(signal, dict):
                conf = signal.get("confidence", 0)
                role = signal.get("role", "").lower()
                if conf >= 0.6 and role in ("primary", "secondary"):
                    primary_domains.append(signal)

        if len(primary_domains) >= 2:
            # Multi-domain detected - preserve candidates from both
            logger.info(
                "Phase 3.5: Multi-domain detected (%d domains). Preserving candidates.",
                len(primary_domains),
            )

            # Mark candidates with domain info
            for candidate in candidates:
                symbol = candidate.get("symbol", "")
                for domain_signal in primary_domains:
                    family = domain_signal.get("cpc_family", "")
                    if family and symbol.startswith(family):
                        candidate.setdefault("domain_tags", []).append(
                            domain_signal.get("name", "")
                        )

        return candidates

    def _normalize_scores(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize scores after all adjustments."""
        if not candidates:
            return candidates

        scores = [c.get("score", 0) for c in candidates]
        max_score = max(scores)

        if max_score > 0:
            for candidate in candidates:
                old_score = candidate.get("score", 0)
                candidate["score"] = round(old_score / max_score, 4)

        return candidates

    def _log_rule(
        self,
        rule: str,
        symbol: str,
        before: float,
        after: float,
        reason: str,
    ) -> None:
        """Log a rule application."""
        self.rules_log.append(
            {
                "rule": rule,
                "symbol": symbol,
                "score_before": round(before, 4),
                "score_after": round(after, 4),
                "reason": reason,
            }
        )
        logger.debug(
            "Phase 3.5: %s | %s | %.4f → %.4f | %s",
            rule,
            symbol,
            before,
            after,
            reason,
        )

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result."""
        return {
            "phase35_candidates": [],
            "phase35_rules_log": [],
            "phase35_domain": None,
            "phase35_domain_confidence": 0.0,
            "phase35_adjustments": 0,
        }


def apply_decision_tree_constraints(
    ranked_candidates: List[Dict[str, Any]],
    phase1_data: Dict[str, Any],
    boost_multiplier: float = 2.0,
    penalty_multiplier: float = 0.3,
) -> Dict[str, Any]:
    """Quick function to apply Phase 3.5 constraints."""
    constraint = CPCDecisionTreeConstraint(boost_multiplier, penalty_multiplier)
    return constraint.apply_constraints(ranked_candidates, phase1_data)
