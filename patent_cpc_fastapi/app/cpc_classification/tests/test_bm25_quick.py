"""
Quick test BM25 index building and search.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cpc_classification.cpc_bm25_index import CPCBM25Index

cache_dir = os.path.join(
    os.path.dirname(__file__), "..", "resources", "cpc_scheme_2026"
)

if not os.path.exists(cache_dir):
    print(f"Cache dir not found: {cache_dir}")
    print("Skipping test")
    sys.exit(0)

print("Building BM25 index...")
index = CPCBM25Index(cache_dir)
index.build()

print(f"Index built: {len(index.symbols)} entries")

# Search
query = "neural network quantization"
results = index.search(query, top_k=10)

print(f"\nQuery: '{query}'")
print(f"Results: {len(results)}")
for symbol, score in results[:5]:
    print(f"  {symbol}: {score:.3f}")

print("\nBM25 test PASSED")
