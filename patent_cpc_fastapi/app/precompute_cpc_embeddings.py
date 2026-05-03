"""
Pre-compute CPC embeddings from XML scheme files.

Run this script once to generate a cache of all CPC subgroup title embeddings.
This avoids re-computing embeddings for the same titles on every request.

Usage:
    python precompute_cpc_embeddings.py

Output:
    Creates cpc_embeddings_cache.json in the resources directory.
"""

import json
import os
import sys
import time
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpc_classification.cpc_xml_parser import CPCXMLParser
from search_core.ollama_client import OllamaClient


def precompute_embeddings(
    xml_dir: str,
    output_path: str,
    model: str = "nomic-embed-text",
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Pre-compute embeddings for all CPC subgroup titles.

    Returns metadata about the operation.
    """
    print(f"Loading XML parser from {xml_dir}...")
    parser = CPCXMLParser(xml_dir)

    # Find all XML files
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} XML files")

    # Collect all unique titles
    all_codes = {}
    print("Parsing XML files...")
    for xml_file in xml_files:
        class_code = xml_file.replace("cpc-scheme-", "").replace(".xml", "")
        try:
            subgroups = parser.parse_file(class_code)
            for sg in subgroups:
                if sg["is_allocatable"] and sg["title"]:
                    # Use symbol as key
                    all_codes[sg["symbol"]] = {
                        "title": sg["title"],
                        "level": sg["level"],
                        "parent_chain": sg.get("parent_chain", []),
                    }
        except Exception as e:
            print(f"  Warning: Failed to parse {xml_file}: {e}")

    print(f"Found {len(all_codes)} unique allocatable codes with titles")

    # Compute embeddings
    client = OllamaClient()
    embeddings = {}

    print(f"Computing embeddings (batch size: {batch_size})...")
    start_time = time.time()

    items = list(all_codes.items())
    total = len(items)

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        print(
            f"  Processing batch {i // batch_size + 1}/{(total - 1) // batch_size + 1} ({i}-{min(i + batch_size, total)})..."
        )

        for symbol, data in batch:
            try:
                vec = client.embeddings(data["title"], model=model)
                embeddings[symbol] = {
                    "title": data["title"],
                    "level": data["level"],
                    "parent_chain": data["parent_chain"],
                    "embedding": vec,
                }
            except Exception as e:
                print(f"    Warning: Failed to embed {symbol}: {e}")

        # Save progress periodically
        if (i // batch_size + 1) % 10 == 0:
            _save_cache(output_path, embeddings)
            print(f"    Saved progress: {len(embeddings)} embeddings")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Total embeddings: {len(embeddings)}")

    # Final save
    _save_cache(output_path, embeddings)
    print(f"Saved to {output_path}")

    return {
        "total_codes": len(all_codes),
        "successful_embeddings": len(embeddings),
        "elapsed_seconds": elapsed,
    }


def _save_cache(path: str, embeddings: Dict) -> None:
    """Save embeddings cache to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)


if __name__ == "__main__":
    # Resolve paths
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_dir = os.path.join(here, "cpc_classification", "resources", "cpc_scheme_2026")
    output_path = os.path.join(
        here, "cpc_classification", "resources", "cpc_embeddings_cache.json"
    )

    print("=" * 60)
    print("CPC Embeddings Pre-computation")
    print("=" * 60)
    print()

    result = precompute_embeddings(xml_dir, output_path)

    print()
    print("Summary:")
    print(f"  Total codes: {result['total_codes']}")
    print(f"  Successful: {result['successful_embeddings']}")
    print(f"  Failed: {result['total_codes'] - result['successful_embeddings']}")
    print(f"  Time: {result['elapsed_seconds']:.1f}s")
