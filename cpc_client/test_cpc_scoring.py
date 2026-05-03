#!/usr/bin/env python3
"""
Diagnostic script to test CPC classification scoring with new features:
- Separate claims/description terms (claims 2x weight)
- Negative signals
- Post-ranking LLM re-ranking
- Best code selection

Usage:
    python test_cpc_scoring.py
"""

import sys
import os

# Add the server app to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "patent_cpc_fastapi", "app")
)

from cpc_classification.search_cpc import CPCClassifier


def safe_print(text):
    """Print text safely handling Unicode encoding issues on Windows."""
    if text is None:
        print("None")
        return
    try:
        print(text)
    except UnicodeEncodeError:
        cleaned = str(text).encode("ascii", "replace").decode("ascii")
        print(cleaned)


# Test patent text - DOWNHOLE EXPLOSIVE CUTTER
TEST_PATENT = """
A downhole tool with explosive charge for cutting well completion tubing and accessory conduits.

TECHNICAL FIELD
The present disclosure relates to downhole tools for use in hydrocarbon wells, and more 
particularly to a partial radial cutter for cutting well completion tubing and accessory 
conduits disposed in a well.

BACKGROUND
In hydrocarbon well operations, it is sometimes necessary to cut or sever well completion 
tubing, casing, or accessory conduits (e.g., control lines, chemical injection lines, etc.) 
located in the wellbore. This may be required during well abandonment, workover operations, 
or when idling a well.

Existing cutting tools often use mechanical cutters or full radial explosive charges. 
Mechanical cutters require significant force and may not work reliably in deep wells. 
Full radial explosive charges can damage surrounding formations or other well components 
due to their 360-degree blast pattern.

The present disclosure provides a partial radial cutter that uses a shaped explosive charge 
with a limited angular extent (e.g., 90-130 degrees) to selectively cut only the target tubing 
or conduit while minimizing damage to surrounding structures.

SUMMARY
From one aspect, the present disclosure provides a downhole tool, comprising: a tubular support; 
and an explosive charge disposed within the tubular support, the explosive charge comprising: 
a casing having the form of a cylinder with a wedge portion removed to form a wedge opening; 
and a shaped charge shaped to fit the wedge opening.

The angular extent of the explosive charge is between about 90 degrees and about 130 degrees, 
inclusive. This partial radial configuration allows selective cutting of only the portion of 
tubing or conduit facing the wedge opening.

CLAIMS:
1. A downhole tool, comprising: a tubular support; and an explosive charge disposed within 
   the tubular support, the explosive charge comprising: a casing having the form of a cylinder 
   with a wedge portion removed to form a wedge opening; and a shaped charge shaped to fit the 
   wedge opening.
2. The downhole tool of claim 1, wherein an angular extent of the explosive charge is between 
   about 90 degrees and about 130 degrees, inclusive.
3. The downhole tool of claim 1, wherein the explosive charge and the casing have respective 
   cutouts that form a conduit along a central axis of the downhole tool.
4. The downhole tool of claim 1, comprising an additional tubular support and an additional 
   explosive charge disposed within the additional tubular support.
5. The downhole tool of claim 4, wherein an orientation of the additional explosive charge in 
   the additional tubular support is in a different direction than an orientation of the explosive 
   charge in the tubular support.
6. The downhole tool of claim 5, wherein the orientation of the additional explosive charge in 
   the additional tubular support is rotated about 90 degrees relative to the orientation of the 
   explosive charge in the tubular support.
7. The downhole tool of claim 1, wherein the casing has one or more alignment features that are 
   configured to position the downhole tool in a desired direction.
8. The downhole tool of claim 1, wherein the casing comprises metal, plastic, or both.
9. The downhole tool of claim 1, wherein the explosive charge comprises a metal container that 
   includes explosive material or a plastic container that includes the explosive material.
10. A method of idling a hydrocarbon well, comprising: positioning a partial radial cutter near 
    an interior wall of a well completion tubing of the well at a location opposite from an accessory 
    conduit disposed in the well outside the well completion tubing, with the discharge portion of 
    the partial radial cutter facing the interior wall of the well completion tubing; and discharging 
    the partial radial cutter to penetrate a portion of the interior wall and cut the accessory conduit.
11. The method of claim 10, wherein the partial radial cutter comprises an explosive charge.
12. The method of claim 11, wherein the explosive charge comprises: a casing having the form of a 
    cylinder with a wedge portion removed to form a wedge opening; and a shaped charge shaped to fit 
    the wedge opening.
13. The method of claim 11, wherein an angular extent of the explosive charge is between about 
    90 degrees and about 130 degrees, inclusive.
14. The method of claim 10, comprising removing the partial radial cutter from the well.
15. The method of claim 10, comprising repositioning the partial radial cutter near the interior 
    wall of the well completion tubing of the well at a second location opposite from a second 
    accessory conduit.
16. A downhole tool, comprising: a tubular support; and an explosive charge disposed within the 
    tubular support, the explosive charge comprising: a casing having the form of a cylinder with 
    a wedge portion removed to form a wedge opening; and a shaped charge shaped to fit the wedge 
    opening; wherein an angular extent of the explosive charge is between 90 degrees and 130 degrees, 
    inclusive.
17. The downhole tool of claim 16, wherein the angular extent is about 100 degrees.
18. The downhole tool of claim 16, comprising a partial radial cutter that includes the explosive charge.
19. The downhole tool of claim 16, comprising an additional tubular support and an additional 
    explosive charge disposed within the additional tubular support.
20. The downhole tool of claim 19, wherein an orientation of the additional explosive charge in 
    the additional tubular support is in a different direction than an orientation of the explosive 
    charge in the tubular support.
"""

# Expected CPC codes from examiner
EXPECTED_CODES = {
    "E21B29/02",  # Cutting by explosives
    "E21B29/04",  # Cutting tools with shaped charges
    "E21B43/11",  # Methods for completion/abandonment
}


def main():
    print("=" * 80)
    print("CPC CLASSIFICATION TEST - DOWNHOLE EXPLOSIVE CUTTER")
    print("=" * 80)
    print()

    classifier = CPCClassifier()

    print("Running classification pipeline...")
    print("(This may take 1-2 minutes due to multiple LLM calls)")
    print()

    result = classifier.classify(TEST_PATENT)

    # Show Phase 1 results
    phase1 = result.get("phase1", {})
    print("-" * 80)
    print("PHASE 1: LLM EXTRACTION")
    print("-" * 80)

    # Check if Phase 1 has valid data
    has_classes = bool(phase1.get("cpc_classes"))
    has_terms = bool(phase1.get("essential_terms"))

    if not has_classes and not has_terms:
        safe_print("WARNING: Phase 1 returned minimal data")
        raw = phase1.get("raw", "N/A")
        safe_print(f"Raw response type: {type(raw)}")
        safe_print(f"Raw response (first 300 chars): {str(raw)[:300]}")
        print()

    safe_print(f"Strategy:        {phase1.get('classification_strategy', 'N/A')}")
    safe_print(f"System Context:  {phase1.get('system_context', 'N/A')}")
    safe_print(f"Core Function:   {phase1.get('core_function', 'N/A')}")
    safe_print(f"CPC Classes:     {phase1.get('cpc_classes', [])}")
    print()

    # Show terms with source
    terms = phase1.get("essential_terms", [])
    if terms:
        print("Essential Terms (Claims terms have 2x weight):")
        print(f"{'Source':<15} {'Term':<30} {'Importance':<12} {'Weight'}")
        print("-" * 80)
        for t in terms[:15]:
            if isinstance(t, dict):
                source = t.get("source", "description")
                term = t.get("term", "?")
                importance = t.get("importance", "?")
                weight = "2x (claims)" if source == "claims" else "1x (desc)"
                safe_print(f"{source:<15} {term:<30} {str(importance):<12} {weight}")
        print()
    else:
        safe_print("No terms extracted")
        print()

    # Show Phase 2/3 results
    ranked = result.get("phase3", [])
    cpc_codes = [node["symbol"] for node in ranked]

    print("-" * 80)
    print("PHASE 2/3: TOP 5 RANKED CODES (After scoring)")
    print("-" * 80)
    if ranked:
        for i, node in enumerate(ranked[:5], 1):
            marker = "  "
            if node["symbol"] in EXPECTED_CODES:
                marker = "* "
            safe_print(
                f"{marker}{i}. {node['symbol']:<15} (score: {node['score']:.4f}) {node['title'][:45]}"
            )
    else:
        safe_print("No ranked codes found")
    print()

    # Show Phase 4 results
    phase4 = result.get("phase4", {})
    if phase4:
        print("-" * 80)
        print("PHASE 4: POST-RANKING LLM RE-RANKING")
        print("-" * 80)
        re_ranked = phase4.get("re_ranked", [])
        for item in re_ranked[:5]:
            safe_print(
                f"  Rank {item.get('rank', '?')}: {item.get('symbol', '?')} - {str(item.get('justification', ''))[:60]}"
            )

        best = phase4.get("best_code", {})
        print()
        safe_print(f"BEST CODE: {best.get('symbol', 'N/A')}")
        safe_print(f"TITLE:     {best.get('title', 'N/A')}")
        safe_print(f"CONFIDENCE: {best.get('confidence', 'N/A')}")
        safe_print(f"REASONING: {str(best.get('reasoning', 'N/A'))[:100]}")
        print()

    # Check expected codes
    print("-" * 80)
    print("EXPECTED CODE ANALYSIS")
    print("-" * 80)

    found_codes = set()
    for code in EXPECTED_CODES:
        if code in cpc_codes:
            rank = cpc_codes.index(code) + 1
            score = ranked[rank - 1]["score"]
            found_codes.add(code)
            safe_print(f"  [FOUND]   {code:<15} rank: {rank}, score: {score:.4f}")
        else:
            safe_print(f"  [MISSING] {code:<15} not in top 5")

    print()
    print("-" * 80)
    safe_print(
        f"SUMMARY: Found {len(found_codes)}/{len(EXPECTED_CODES)} expected codes"
    )
    if phase4:
        best_sym = phase4.get("best_code", {}).get("symbol", "N/A")
        safe_print(f"BEST CODE: {best_sym}")
    print("=" * 80)

    return len(found_codes), len(EXPECTED_CODES)


if __name__ == "__main__":
    try:
        found, total = main()
        sys.exit(0 if found == total else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
