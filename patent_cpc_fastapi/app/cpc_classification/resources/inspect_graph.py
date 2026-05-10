"""
CPC Knowledge Graph Inspector
Location: patent_cpc_fastapi/app/cpc_classification/resources/
Usage: python inspect_graph.py [options]

Options:
  --section SECTION     Show only specific section (A, B, C, etc.)
  --class CLASS         Show specific 3-char class (e.g., G06, H04)
  --search TEXT         Search for text in titles
  --code CODE           Find specific CPC code (e.g., G06F17/30)
  --stats               Quick summary only
  --json                Output as JSON
  --check               Validate graph integrity

Examples:
  python inspect_graph.py
  python inspect_graph.py --section G
  python inspect_graph.py --class G06F
  python inspect_graph.py --search "neural network"
  python inspect_graph.py --code G06F17/30
  python inspect_graph.py --check
"""

import pickle
import os
import sys
import json
import argparse
from datetime import datetime
from collections import Counter

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(SCRIPT_DIR, "cpc_graph.pkl")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "cpc_embeddings.npz")
META_PATH = os.path.join(SCRIPT_DIR, "cpc_graph_meta.json")


def get_file_info(path):
    """Get file size and modification date."""
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {"size_mb": size_mb, "modified": mtime}


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def load_graph():
    """Load the knowledge graph from pickle."""
    if not os.path.exists(GRAPH_PATH):
        print(f"ERROR: Graph file not found: {GRAPH_PATH}")
        sys.exit(1)

    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    return graph


def show_file_stats():
    """Show information about cache files."""
    print_header("CACHE FILES")

    for name, path in [
        ("Graph", GRAPH_PATH),
        ("Embeddings", EMBEDDINGS_PATH),
        ("Metadata", META_PATH),
    ]:
        info = get_file_info(path)
        if info:
            print(
                f"  {name:12s}: {info['size_mb']:8.1f} MB  (modified: {info['modified']})"
            )
        else:
            print(f"  {name:12s}: NOT FOUND")
    print()


def show_basic_stats(graph):
    """Show basic graph statistics."""
    print_header("BASIC STATISTICS")
    print(f"  Graph type:  {type(graph).__name__}")
    print(f"  Total nodes: {graph.number_of_nodes():,}")
    print(f"  Total edges: {graph.number_of_edges():,}")

    # Node types
    node_types = Counter(
        data.get("type", "unknown") for _, data in graph.nodes(data=True)
    )
    print(f"\n  Node types:")
    for node_type, count in node_types.most_common():
        print(f"    {node_type:15s}: {count:>10,}")


def show_section_breakdown(graph):
    """Show breakdown by CPC section."""
    print_header("SECTIONS BREAKDOWN")

    sections = Counter()
    for node, data in graph.nodes(data=True):
        if data.get("type") == "subgroup" and len(node) >= 1:
            sections[node[0]] += 1

    total = sum(sections.values())
    for section, count in sorted(sections.items()):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  Section {section}: {count:>8,} ({pct:5.1f}%) {bar}")
    print(f"  {'Total':>10s}: {total:>8,}")


def show_classes(graph, section=None):
    """Show 3-char classes, optionally filtered by section."""
    if section:
        print_header(f"3-CHAR CLASSES IN SECTION {section}")
    else:
        print_header("3-CHAR CLASSES (top 30)")

    classes = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "class_3char":
            if section is None or node.startswith(section):
                count = len(list(graph.successors(node)))
                classes.append((node, data.get("title", "N/A"), count))

    classes.sort(key=lambda x: -x[2])

    for cls, title, count in classes[:30]:
        print(f"  {cls:6s} ({count:>5,} subgroups): {title[:50]}")

    if len(classes) > 30:
        print(f"  ... and {len(classes) - 30} more")


def search_titles(graph, query):
    """Search for text in subgroup titles - all words must match (AND logic)."""
    print_header(f'SEARCHING FOR "{query}"')

    query_words = [w.strip().lower() for w in query.split() if w.strip()]
    matches = []

    for node, data in graph.nodes(data=True):
        if data.get("type") == "subgroup":
            title = data.get("title", "").lower()
            if all(word in title for word in query_words):
                matches.append((node, data.get("title", "")))

    print(f"Found {len(matches)} matches\n")

    for symbol, title in matches[:20]:
        print(f"  {symbol:20s}: {title[:70]}")

    if len(matches) > 20:
        print(f"  ... and {len(matches) - 20} more matches")


def show_code(graph, code):
    """Show details for a specific CPC code."""
    print_header(f"CPC CODE: {code}")

    if code not in graph:
        print(f"  Code not found in graph")
        return

    data = graph.nodes[code]
    print(f"  Type:        {data.get('type', 'N/A')}")
    print(f"  Title:       {data.get('title', 'N/A')}")
    print(f"  Level:       {data.get('level', 'N/A')}")
    print(f"  Allocatable: {data.get('is_allocatable', 'N/A')}")
    print(f"  Class:       {data.get('prefix3', 'N/A')}")

    # Show parents
    predecessors = list(graph.predecessors(code))
    if predecessors:
        print(f"\n  Parents:")
        for parent in predecessors:
            parent_data = graph.nodes[parent]
            print(f"    {parent:15s}: {parent_data.get('title', 'N/A')[:50]}")

    # Show children
    successors = list(graph.successors(code))
    if successors:
        print(f"\n  Children ({len(successors)}):")
        for child in successors[:10]:
            child_data = graph.nodes[child]
            print(f"    {child:15s}: {child_data.get('title', 'N/A')[:50]}")
        if len(successors) > 10:
            print(f"    ... and {len(successors) - 10} more")


def show_class_details(graph, class_code):
    """Show details for a 3-char class."""
    print_header(f"CLASS: {class_code}")

    if class_code not in graph:
        print(f"  Class not found")
        return

    data = graph.nodes[class_code]
    print(f"  Title: {data.get('title', 'N/A')}")

    subgroups = list(graph.successors(class_code))
    print(f"\n  Subgroups: {len(subgroups)}")

    for sg in sorted(subgroups)[:15]:
        sg_data = graph.nodes[sg]
        print(f"    {sg:15s}: {sg_data.get('title', 'N/A')[:60]}")

    if len(subgroups) > 15:
        print(f"    ... and {len(subgroups) - 15} more")


def check_integrity(graph):
    """Validate graph integrity."""
    print_header("INTEGRITY CHECK")

    issues = []

    # Check for orphaned subgroups (no parent edge)
    orphaned = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "subgroup":
            if graph.in_degree(node) == 0 and data.get("level", 0) > 0:
                orphaned.append(node)

    if orphaned:
        issues.append(f"Found {len(orphaned)} orphaned subgroups (no parent)")
        print(f"  WARNING: {len(orphaned)} orphaned subgroups")
    else:
        print(f"  OK: No orphaned subgroups")

    # Check for class_3char nodes without edges
    empty_classes = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "class_3char":
            if graph.out_degree(node) == 0:
                empty_classes.append(node)

    if empty_classes:
        issues.append(f"Found {len(empty_classes)} empty classes")
        print(f"  WARNING: {len(empty_classes)} classes without subgroups")
    else:
        print(f"  OK: All classes have subgroups")

    # Check embeddings coverage
    if os.path.exists(EMBEDDINGS_PATH):
        import numpy as np

        data = np.load(EMBEDDINGS_PATH)
        embedding_count = len(data["symbols"])
        subgroup_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("type") == "subgroup"
        )

        if embedding_count == subgroup_count:
            print(f"  OK: All {subgroup_count:,} subgroups have embeddings")
        else:
            issues.append(
                f"Mismatch: {subgroup_count} subgroups but {embedding_count} embeddings"
            )
            print(
                f"  WARNING: {subgroup_count:,} subgroups but {embedding_count:,} embeddings"
            )
    else:
        print(f"  INFO: Embeddings file not found")

    if not issues:
        print(f"\n  [OK] Graph integrity check passed")
    else:
        print(f"\n  [WARNING] Found {len(issues)} issues")


def export_json(graph):
    """Export graph summary as JSON."""
    sections = Counter()
    for node, data in graph.nodes(data=True):
        if data.get("type") == "subgroup" and len(node) >= 1:
            sections[node[0]] += 1

    node_types = Counter(
        data.get("type", "unknown") for _, data in graph.nodes(data=True)
    )

    result = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_types": dict(node_types),
        "sections": dict(sorted(sections.items())),
        "cache_files": {},
    }

    for name, path in [
        ("graph", GRAPH_PATH),
        ("embeddings", EMBEDDINGS_PATH),
        ("metadata", META_PATH),
    ]:
        info = get_file_info(path)
        if info:
            result["cache_files"][name] = info

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect CPC Knowledge Graph")
    parser.add_argument("--section", help="Show only specific section (A, B, C, etc.)")
    parser.add_argument(
        "--class", dest="class_", help="Show specific 3-char class (e.g., G06, H04)"
    )
    parser.add_argument("--search", help="Search for text in titles")
    parser.add_argument("--code", help="Find specific CPC code (e.g., G06F17/30)")
    parser.add_argument("--stats", action="store_true", help="Quick summary only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check", action="store_true", help="Validate graph integrity")
    args = parser.parse_args()

    # Show file stats first
    show_file_stats()

    # Load graph
    graph = load_graph()

    # Handle different output modes
    if args.json:
        export_json(graph)
        return

    if args.stats:
        show_basic_stats(graph)
        show_section_breakdown(graph)
        return

    if args.check:
        show_basic_stats(graph)
        check_integrity(graph)
        return

    if args.search:
        search_titles(graph, args.search)
        return

    if args.code:
        show_code(graph, args.code)
        return

    if args.class_:
        show_class_details(graph, args.class_)
        return

    # Default: full report
    show_basic_stats(graph)
    show_section_breakdown(graph)
    show_classes(graph, args.section)
    check_integrity(graph)

    print("\n" + "=" * 60)
    print("Tips:")
    print("  python inspect_graph.py --section G     # Show only section G")
    print("  python inspect_graph.py --class G06F    # Show class G06F")
    print("  python inspect_graph.py --search chat   # Search titles")
    print("  python inspect_graph.py --code G06F17   # Show specific code")
    print("  python inspect_graph.py --stats         # Quick summary")
    print("  python inspect_graph.py --check         # Validate integrity")
    print("=" * 60)


if __name__ == "__main__":
    main()
