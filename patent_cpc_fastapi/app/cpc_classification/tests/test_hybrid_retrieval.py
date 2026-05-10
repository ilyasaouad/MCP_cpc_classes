"""
Test BM25 + Cross-Encoder Hybrid Retrieval
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

print("=" * 70)
print("BM25 + CROSS-ENCODER TEST")
print("=" * 70)

# Test 1: BM25 Index
print("\n--- Test 1: BM25 Index ---")
try:
    from cpc_classification.cpc_bm25_index import CPCBM25Index

    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", "resources", "cpc_scheme_2026"
    )

    if os.path.exists(cache_dir):
        index = CPCBM25Index(cache_dir)
        index.build()

        print(f"  Built index with {len(index.symbols)} entries")

        # Search
        results = index.search("neural network quantization", top_k=10)
        print(f"  Search returned {len(results)} results")
        for symbol, score in results[:5]:
            print(f"    {symbol}: {score:.3f}")

        # Save/load
        test_path = "/tmp/test_bm25.pkl"
        index.save(test_path)

        index2 = CPCBM25Index()
        loaded = index2.load(test_path)
        print(f"  Save/load: {'OK' if loaded else 'FAILED'}")

        if loaded:
            os.remove(test_path)
    else:
        print(f"  Cache dir not found: {cache_dir}")
        print("  Skipping BM25 test")

except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Cross-Encoder (skip if model download fails)
print("\n--- Test 2: Cross-Encoder ---")
try:
    from cpc_classification.cpc_cross_encoder import CPCCrossEncoder

    print("  Loading cross-encoder model...")
    reranker = CPCCrossEncoder()

    candidates = [
        ("G06N3/063", "Language models"),
        ("G06N3/047", "Quantisation of neural network parameters"),
        ("G06F17/16", "Digital computing"),
    ]

    results = reranker.rerank(
        "neural network quantization",
        candidates,
        top_k=3,
    )

    print(f"  Reranked {len(results)} candidates:")
    for symbol, score in results:
        print(f"    {symbol}: {score:.3f}")

except Exception as e:
    print(f"  ERROR (model may need download): {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
