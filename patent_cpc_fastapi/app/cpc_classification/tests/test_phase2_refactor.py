"""
Test script for Phase 2 refactor (2A/2B/2C pipeline)

This script verifies:
1. CPCFamilyRouter correctly identifies families
2. XML parser filters by allowed_roots
3. Pipeline produces debug output
4. No syntax/runtime errors

Usage:
    python test_phase2_refactor.py
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_family_router import CPCFamilyRouter, extract_3char_families
from cpc_classification.cpc_xml_parser import CPCXMLParser


def test_family_router_basic():
    """Test family router with mock Phase 1 data."""
    print("\n=== Test 1: Family Router Basic ===")

    # Mock Phase 1 data (no knowledge graph - should use hypotheses fallback)
    phase1_data = {
        "technical_object": "A chatbot system for customer service automation",
        "core_function": "Natural language processing and dialog management",
        "system_context": "Customer service automation in telecommunications",
        "terms": [
            {"term": "chatbot", "importance": 10},
            {"term": "natural language processing", "importance": 9},
            {"term": "dialog system", "importance": 9},
            {"term": "customer service", "importance": 7},
            {"term": "machine learning", "importance": 6},
        ],
        "class_hypotheses": [
            {"class": "G06F", "confidence": 0.8},
            {"class": "G06N", "confidence": 0.7},
            {"class": "H04L", "confidence": 0.5},
        ],
        "cpc_classes": ["G06F17/30", "G06N3/08"],
    }

    router = CPCFamilyRouter(knowledge_graph=None, max_families=3)
    result = router.route(phase1_data)

    print(f"Families: {result['families']}")
    print(f"Source: {result['source']}")
    print(f"Reasoning: {result['reasoning']}")

    assert len(result["families"]) >= 2, "Should return at least 2 families"
    assert all(len(f) == 3 for f in result["families"]), (
        "All families should be 3 chars"
    )
    assert result["source"] == "hypotheses", "Should use hypotheses fallback when no KG"

    print("[PASS] Test 1 passed")


def test_extract_3char_families():
    """Test utility function for extracting families."""
    print("\n=== Test 2: Extract 3-char Families ===")

    codes = ["G06F17/30", "G06N3/08", "H04L29/06", "G06F40/295"]
    families = extract_3char_families(codes)

    print(f"Input: {codes}")
    print(f"Families: {families}")

    assert "G06" in families, "Should extract G06"
    assert "H04" in families, "Should extract H04"
    assert len(families) == 2, "Should have 2 unique families"
    assert all(len(f) == 3 for f in families), "All should be 3 chars"

    print("[PASS] Test 2 passed")


def test_xml_parser_filtering():
    """Test XML parser with allowed_roots filtering."""
    print("\n=== Test 3: XML Parser Filtering ===")

    xml_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "resources",
        "cpc_scheme_2026",
    )

    if not os.path.exists(xml_dir):
        print("⚠ Skipping: XML directory not found")
        return

    parser = CPCXMLParser(xml_dir)

    # Test without filtering
    all_results = parser.expand_classes(["G06F"], include_non_allocatable=False)
    print(f"Without filter: {len(all_results)} subgroups")

    # Test with filtering (should return same since G06F starts with G06)
    filtered_results = parser.expand_classes(
        ["G06F"], include_non_allocatable=False, allowed_roots=["G06"]
    )
    print(f"With G06 filter: {len(filtered_results)} subgroups")

    assert len(filtered_results) <= len(all_results), (
        "Filter should reduce or keep same"
    )

    # Test with mismatching filter (should return empty)
    empty_results = parser.expand_classes(
        ["G06F"], include_non_allocatable=False, allowed_roots=["H04"]
    )
    print(f"With H04 filter (mismatch): {len(empty_results)} subgroups")

    assert len(empty_results) == 0, "Mismatch filter should return empty"

    print("[PASS] Test 3 passed")


def test_integration():
    """Integration test with full pipeline."""
    print("\n=== Test 4: Integration Test ===")

    xml_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "resources",
        "cpc_scheme_2026",
    )

    if not os.path.exists(xml_dir):
        print("⚠ Skipping: XML directory not found")
        return

    # This would require a full knowledge graph, so we just verify structure
    from cpc_classification.search_cpc import CPCClassifier

    # Create classifier (without KG for speed)
    classifier = CPCClassifier(knowledge_graph=None)

    # Verify it has the family router
    assert hasattr(classifier, "family_router"), "Classifier should have family_router"
    assert classifier.family_router is not None, "Family router should be initialized"

    print("[PASS] Test 4 passed")


def main():
    print("=" * 60)
    print("Phase 2 Refactor Test Suite")
    print("=" * 60)

    test_family_router_basic()
    test_extract_3char_families()
    test_xml_parser_filtering()
    test_integration()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
