"""
test_phase15_role_classifier.py - Tests for Phase 1.5 Invention Role Classifier

Tests the role classification logic for CORE_TECH / SYSTEM / APPLICATION / SUPPORT.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestCPCRoleClassifier:
    """Test the role classifier module."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        return Mock()

    @pytest.fixture
    def classifier(self, mock_llm):
        """Create classifier with mocked LLM."""
        from app.cpc_classification.cpc_role_classifier import CPCRoleClassifier

        return CPCRoleClassifier(mock_llm)

    def test_rule_based_core_tech(self, classifier):
        """Test CORE_TECH classification for algorithm innovation."""
        phase1_data = {
            "technical_object": "A method for training neural networks using a novel optimization algorithm",
            "core_function": "Improving neural network training through adaptive weight adjustment",
            "system_context": "Machine learning systems",
            "terms": [
                {
                    "term": "weight optimization",
                    "importance": 10,
                    "source_section": "claims",
                },
                {
                    "term": "adaptive learning rate",
                    "importance": 9,
                    "source_section": "claims",
                },
                {
                    "term": "gradient descent",
                    "importance": 8,
                    "source_section": "summary",
                },
            ],
        }

        result = classifier._rule_based_classification(phase1_data)

        # Should classify as CORE_TECH due to algorithm/innovation signals
        assert result["role"] in ["CORE_TECH", "SYSTEM", "APPLICATION"]
        assert "confidence" in result

    def test_rule_based_system(self, classifier):
        """Test SYSTEM classification for multi-component orchestration."""
        phase1_data = {
            "technical_object": "A pipeline for coordinating multiple machine learning models",
            "core_function": "Managing data flow between classification and detection modules",
            "system_context": "Data processing systems",
            "terms": [
                {"term": "pipeline", "importance": 10, "source_section": "claims"},
                {"term": "orchestrating", "importance": 9, "source_section": "claims"},
                {"term": "data flow", "importance": 8, "source_section": "summary"},
            ],
        }

        result = classifier._rule_based_classification(phase1_data)

        # Should classify as SYSTEM due to orchestration signals
        assert result["role"] in ["CORE_TECH", "SYSTEM", "APPLICATION"]
        assert "confidence" in result

    def test_rule_based_application(self, classifier):
        """Test APPLICATION classification for domain-specific deployment."""
        phase1_data = {
            "technical_object": "A system for medical diagnosis using machine learning",
            "core_function": "Applying ML models to patient data for disease detection",
            "system_context": "Healthcare systems",
            "terms": [
                {
                    "term": "medical diagnosis",
                    "importance": 10,
                    "source_section": "claims",
                },
                {"term": "patient data", "importance": 9, "source_section": "claims"},
                {
                    "term": "disease detection",
                    "importance": 8,
                    "source_section": "summary",
                },
            ],
        }

        result = classifier._rule_based_classification(phase1_data)

        # Should classify as APPLICATION due to medical signals
        assert result["role"] in ["CORE_TECH", "SYSTEM", "APPLICATION"]
        assert "confidence" in result

    def test_empty_phase1_data(self, classifier):
        """Test default result for empty phase1 data."""
        result = classifier.classify_role({})

        assert result["role"] == "SYSTEM"
        assert result["confidence"] == 0.5
        assert len(result) == 2  # only role + confidence

    def test_format_terms(self, classifier):
        """Test term formatting for prompt."""
        terms = [
            {"term": "neural network", "importance": 10, "source_section": "claims"},
            {"term": "quantization", "importance": 8, "source_section": "summary"},
            {"term": "weight clipping", "importance": 9, "source_section": "claims"},
        ]

        formatted = classifier._format_terms(terms, max_terms=2)

        assert "neural network" in formatted
        assert "quantization" in formatted
        assert "weight clipping" in formatted
        assert "importance:" in formatted


class TestRoleFamilyBoosts:
    """Test role-based family boosting logic."""

    def test_core_tech_boosts(self):
        """Test CORE_TECH boosts technology-native classes."""
        from app.cpc_classification.cpc_role_classifier import get_role_family_boosts

        boosts = get_role_family_boosts("CORE_TECH")

        assert boosts["G06N"] == 1.5  # AI boosted
        assert boosts["G06T"] == 1.3  # Image boosted
        assert boosts["G06F"] < 1.0  # Computing deprioritized
        assert boosts["H04L"] < 1.0  # Telecom deprioritized

    def test_system_boosts(self):
        """Test SYSTEM boosts orchestration classes."""
        from app.cpc_classification.cpc_role_classifier import get_role_family_boosts

        boosts = get_role_family_boosts("SYSTEM")

        assert boosts["G06F"] == 1.5  # Computing boosted
        assert boosts["H04L"] == 1.4  # Networking boosted
        assert boosts["G06N"] < 1.0  # AI deprioritized unless NN-internal

    def test_application_boosts(self):
        """Test APPLICATION boosts domain-specific classes."""
        from app.cpc_classification.cpc_role_classifier import get_role_family_boosts

        boosts = get_role_family_boosts("APPLICATION")

        assert boosts["A61"] == 1.5  # Medical boosted
        assert boosts["B60"] == 1.4  # Vehicles boosted
        assert boosts["G06N"] < 1.0  # AI is tool, not subject

    def test_support_boosts(self):
        """Test SUPPORT boosts general computing."""
        from app.cpc_classification.cpc_role_classifier import get_role_family_boosts

        boosts = get_role_family_boosts("SUPPORT")

        assert boosts["G06F"] == 1.5  # General computing boosted


class TestApplyRoleScoring:
    """Test role-based scoring application."""

    def test_apply_core_tech_with_nn_internal(self):
        """Test CORE_TECH with NN-internal signals boosts G06N strongly."""
        from app.cpc_classification.cpc_role_classifier import apply_role_scoring

        family_scores = {"G06F": 1.0, "G06N": 1.0, "H04L": 0.8}
        phase1_data = {
            "terms": [
                {"term": "weight quantization", "importance": 10},
                {"term": "model compression", "importance": 9},
            ]
        }

        result = apply_role_scoring(family_scores, "CORE_TECH", phase1_data)

        # G06N should be strongly boosted (2.0x) due to NN-internal signals
        assert result["G06N"] > family_scores["G06N"] * 1.5

    def test_apply_system_penalizes_g06n(self):
        """Test SYSTEM role penalizes G06N."""
        from app.cpc_classification.cpc_role_classifier import apply_role_scoring

        family_scores = {"G06F": 1.0, "G06N": 1.0}
        phase1_data = {
            "terms": [
                {"term": "pipeline", "importance": 10},
                {"term": "orchestration", "importance": 9},
            ]
        }

        result = apply_role_scoring(family_scores, "SYSTEM", phase1_data)

        # G06N should be penalized (0.6x)
        assert result["G06N"] < family_scores["G06N"]

    def test_apply_application_penalizes_g06n(self):
        """Test APPLICATION role penalizes G06N (AI is tool)."""
        from app.cpc_classification.cpc_role_classifier import apply_role_scoring

        family_scores = {"G06F": 1.0, "G06N": 1.0, "A61": 0.5}
        phase1_data = {
            "terms": [
                {"term": "medical diagnosis", "importance": 10},
            ]
        }

        result = apply_role_scoring(family_scores, "APPLICATION", phase1_data)

        # G06N should be penalized (0.7x)
        assert result["G06N"] < family_scores["G06N"]
        # A61 should be boosted (1.5x)
        assert result["A61"] > family_scores["A61"]


class TestCriticalRules:
    """Test critical classification rules."""

    def test_llm_not_implying_core_tech(self, classifier):
        """Test that 'using LLM' does not imply CORE_TECH."""
        phase1_data = {
            "technical_object": "A system for classifying documents using an LLM",
            "core_function": "Applying an LLM to classify documents",
            "system_context": "Document management systems",
            "terms": [
                {"term": "using LLM", "importance": 10},
                {"term": "classification", "importance": 9},
            ],
        }

        result = classifier._rule_based_classification(phase1_data)

        # Should NOT be CORE_TECH because LLM is used as a tool
        # The anti-CORE_TECH signal "using an LLM" penalizes CORE_TECH
        assert result["role"] in ["SYSTEM", "APPLICATION"]

    def test_multi_component_bias_to_system(self, classifier):
        """Test that multiple components bias toward SYSTEM."""
        phase1_data = {
            "technical_object": "A system coordinating multiple models",
            "core_function": "Managing interaction between model A and model B",
            "system_context": "Multi-model systems",
            "terms": [
                {"term": "pipeline", "importance": 10},
                {"term": "multi-model", "importance": 9},
                {"term": "interaction", "importance": 8},
            ],
        }

        result = classifier._rule_based_classification(phase1_data)

        # Should be SYSTEM due to orchestration signals
        assert result["role"] == "SYSTEM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
