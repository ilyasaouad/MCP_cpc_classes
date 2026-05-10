"""
CPC Knowledge Graph Inspector - Streamlit App
Usage: streamlit run inspect_graph_streamlit.py
"""

import pickle
import os
import json
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="CPC Graph Inspector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(SCRIPT_DIR, "cpc_graph.pkl")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "cpc_embeddings.npz")
META_PATH = os.path.join(SCRIPT_DIR, "cpc_graph_meta.json")

# Global cache for embeddings data (loaded once)
_embeddings_data = None
_model = None


@st.cache_resource
def load_graph():
    """Load graph once and cache."""
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    return graph


def get_graph_stats(_graph):
    """Compute graph statistics."""
    # Node types
    node_types = Counter()
    sections = Counter()
    classes_3char = []

    for node, data in _graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types[node_type] += 1

        if node_type == "subgroup" and len(node) >= 1:
            sections[node[0]] += 1
        elif node_type == "class_3char":
            count = len(list(_graph.successors(node)))
            classes_3char.append(
                {"class": node, "title": data.get("title", "N/A"), "subgroups": count}
            )

    # Edges by relation
    relations = Counter()
    for _, _, data in _graph.edges(data=True):
        relations[data.get("relation", "unknown")] += 1

    return {
        "node_types": dict(node_types),
        "sections": dict(sorted(sections.items())),
        "classes": sorted(classes_3char, key=lambda x: -x["subgroups"]),
        "relations": dict(relations),
        "total_nodes": _graph.number_of_nodes(),
        "total_edges": _graph.number_of_edges(),
    }


@st.cache_data
def search_titles(_graph, query):
    """Search for text in titles - all words must match (AND logic)."""
    # Split query into words and remove empty strings
    query_words = [w.strip().lower() for w in query.split() if w.strip()]

    if not query_words:
        return []

    matches = []

    for node, data in _graph.nodes(data=True):
        if data.get("type") == "subgroup":
            title = data.get("title", "").lower()
            # All query words must be in the title
            if all(word in title for word in query_words):
                matches.append(
                    {
                        "code": node,
                        "title": data.get("title", ""),
                        "section": node[0] if len(node) >= 1 else "?",
                        "allocatable": data.get("is_allocatable", False),
                    }
                )

    return matches


@st.cache_data
def get_class_subgroups(_graph, class_code):
    """Get all subgroups for a class."""
    subgroups = []
    for sg in _graph.successors(class_code):
        data = _graph.nodes[sg]
        if data.get("type") == "subgroup":
            subgroups.append(
                {
                    "code": sg,
                    "title": data.get("title", "N/A"),
                    "level": data.get("level", 0),
                    "allocatable": data.get("is_allocatable", False),
                }
            )
    return sorted(subgroups, key=lambda x: x["code"])


@st.cache_data
def get_code_details(_graph, code):
    """Get details for a specific code."""
    if code not in _graph:
        return None

    data = _graph.nodes[code]

    # Parents
    parents = []
    for parent in _graph.predecessors(code):
        parent_data = _graph.nodes[parent]
        parents.append(
            {
                "code": parent,
                "title": parent_data.get("title", "N/A"),
                "relation": _graph.edges[parent, code].get("relation", "unknown"),
            }
        )

    # Children
    children = []
    for child in _graph.successors(code):
        child_data = _graph.nodes[child]
        children.append(
            {
                "code": child,
                "title": child_data.get("title", "N/A"),
                "relation": _graph.edges[code, child].get("relation", "unknown"),
            }
        )

    return {
        "code": code,
        "type": data.get("type", "N/A"),
        "title": data.get("title", "N/A"),
        "level": data.get("level", 0),
        "allocatable": data.get("is_allocatable", False),
        "parents": parents,
        "children": children,
    }


def load_embeddings():
    """Load embeddings data from file."""
    global _embeddings_data
    if _embeddings_data is None:
        if not os.path.exists(EMBEDDINGS_PATH):
            return None
        emb_data = np.load(EMBEDDINGS_PATH)
        _embeddings_data = {
            "symbols": emb_data["symbols"].tolist(),
            "embeddings": emb_data["embeddings"],
        }
    return _embeddings_data


def load_semantic_model():
    """Load sentence transformer model for semantic search."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def semantic_search(query, top_k=50):
    """Search using semantic embeddings and cosine similarity."""
    emb_data = load_embeddings()
    if emb_data is None:
        return None, "Embeddings file not found"

    model = load_semantic_model()

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, emb_data["embeddings"])[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        symbol = emb_data["symbols"][idx]
        score = float(similarities[idx])
        results.append(
            {
                "code": symbol,
                "similarity": score,
                "section": symbol[0] if len(symbol) >= 1 else "?",
            }
        )

    return results, None


def check_integrity(graph):
    """Validate graph integrity."""
    issues = []

    # Check orphans
    orphaned = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "subgroup":
            if graph.in_degree(node) == 0 and data.get("level", 0) > 0:
                orphaned.append(node)

    if orphaned:
        issues.append(f"{len(orphaned)} orphaned subgroups (no parent)")

    # Check empty classes
    empty_classes = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "class_3char":
            if graph.out_degree(node) == 0:
                empty_classes.append(node)

    if empty_classes:
        issues.append(f"{len(empty_classes)} empty classes")

    # Check embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        emb_data = np.load(EMBEDDINGS_PATH)
        emb_count = len(emb_data["symbols"])
        sg_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("type") == "subgroup"
        )

        if emb_count != sg_count:
            issues.append(
                f"Embedding mismatch: {sg_count} subgroups vs {emb_count} embeddings"
            )

    return issues


def main():
    st.title("📊 CPC Knowledge Graph Inspector")
    st.markdown("Explore the CPC classification knowledge graph")

    # Check if graph exists
    if not os.path.exists(GRAPH_PATH):
        st.error(f"Graph file not found: {GRAPH_PATH}")
        st.info("Please build the knowledge graph first.")
        return

    # Load graph
    with st.spinner("Loading graph..."):
        graph = load_graph()
        stats = get_graph_stats(graph)

    st.success(
        f"Loaded: {stats['total_nodes']:,} nodes, {stats['total_edges']:,} edges"
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "Overview",
            "Sections",
            "Classes",
            "Search",
            "Code Browser",
            "Integrity Check",
        ],
    )

    if page == "Overview":
        show_overview(stats)
    elif page == "Sections":
        show_sections(graph, stats)
    elif page == "Classes":
        show_classes(graph, stats)
    elif page == "Search":
        show_search(graph)
    elif page == "Code Browser":
        show_code_browser(graph)
    elif page == "Integrity Check":
        show_integrity(graph)


def show_overview(stats):
    st.header("📈 Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", f"{stats['total_nodes']:,}")
    col2.metric("Total Edges", f"{stats['total_edges']:,}")
    col3.metric("Subgroups", f"{stats['node_types'].get('subgroup', 0):,}")
    col4.metric("3-Char Classes", f"{stats['node_types'].get('class_3char', 0):,}")

    st.divider()

    # Node types
    st.subheader("Node Types")
    node_types_df = pd.DataFrame(
        [
            {
                "Type": k,
                "Count": v,
                "Percentage": f"{v / stats['total_nodes'] * 100:.1f}%",
            }
            for k, v in stats["node_types"].items()
        ]
    )
    st.dataframe(node_types_df, use_container_width=True, hide_index=True)

    # Relations
    st.subheader("Edge Relations")
    relations_df = pd.DataFrame(
        [{"Relation": k, "Count": v} for k, v in stats["relations"].items()]
    )
    st.dataframe(relations_df, use_container_width=True, hide_index=True)

    # Cache files
    st.subheader("Cache Files")
    files_data = []
    for name, path in [
        ("Graph", GRAPH_PATH),
        ("Embeddings", EMBEDDINGS_PATH),
        ("Metadata", META_PATH),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = os.path.getmtime(path)
            files_data.append(
                {
                    "File": name,
                    "Size (MB)": f"{size_mb:.1f}",
                    "Modified": pd.to_datetime(mtime, unit="s").strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
            )

    if files_data:
        st.dataframe(
            pd.DataFrame(files_data), use_container_width=True, hide_index=True
        )


def show_sections(graph, stats):
    st.header("📑 Sections Breakdown")

    # Create bar chart data
    sections_df = pd.DataFrame(
        [
            {
                "Section": k,
                "Subgroups": v,
                "Percentage": v / sum(stats["sections"].values()) * 100,
            }
            for k, v in stats["sections"].items()
        ]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(sections_df.set_index("Section")["Subgroups"])

    with col2:
        st.dataframe(
            sections_df.sort_values("Subgroups", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # Section selector
    st.divider()
    selected_section = st.selectbox(
        "Select Section to Explore",
        options=sorted(stats["sections"].keys()),
        format_func=lambda x: f"Section {x} ({stats['sections'][x]:,} subgroups)",
    )

    if selected_section:
        st.subheader(f"Section {selected_section} Classes")
        section_classes = [
            c for c in stats["classes"] if c["class"].startswith(selected_section)
        ]

        if section_classes:
            classes_df = pd.DataFrame(section_classes)
            st.dataframe(classes_df, use_container_width=True, hide_index=True)
        else:
            st.info("No classes found in this section")


def show_classes(graph, stats):
    st.header("🏛️ Classes Browser")

    # Top classes
    st.subheader("Top Classes by Subgroup Count")
    top_classes = stats["classes"][:50]
    classes_df = pd.DataFrame(top_classes)
    st.dataframe(classes_df, use_container_width=True, hide_index=True)

    st.divider()

    # Class explorer
    selected_class = st.selectbox(
        "Select Class to Explore",
        options=[c["class"] for c in stats["classes"]],
        format_func=lambda x: (
            f"{x} ({next(c['subgroups'] for c in stats['classes'] if c['class'] == x)} subgroups)"
        ),
    )

    if selected_class:
        st.subheader(f"Class: {selected_class}")
        class_data = next(
            (c for c in stats["classes"] if c["class"] == selected_class), None
        )
        if class_data:
            st.write(f"**Title:** {class_data['title']}")
            st.write(f"**Subgroups:** {class_data['subgroups']:,}")

        # Show subgroups
        with st.spinner("Loading subgroups..."):
            subgroups = get_class_subgroups(graph, selected_class)

        st.write(f"**Total Subgroups:** {len(subgroups)}")

        # Filter options
        show_all = st.checkbox(
            "Show all subgroups (may be slow for large classes)", value=False
        )

        if show_all or len(subgroups) <= 100:
            sg_df = pd.DataFrame(subgroups)
            st.dataframe(sg_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Showing first 100 of {len(subgroups)} subgroups")
            sg_df = pd.DataFrame(subgroups[:100])
            st.dataframe(sg_df, use_container_width=True, hide_index=True)


def show_search(graph):
    st.header("🔍 Search")

    # Search mode toggle
    search_mode = st.radio(
        "Search Mode",
        ["Keyword Search", "Semantic Search"],
        help="Keyword: exact word matching | Semantic: finds conceptually related codes",
    )

    if search_mode == "Keyword Search":
        st.info(
            "💡 **Keyword Search** (default): Finds titles containing ALL your search words. Fast but requires exact word matches."
        )
    else:
        st.info(
            "🧠 **Semantic Search**: Finds conceptually related codes using AI embeddings. Slower first load (~30s) but finds related concepts even without exact word matches."
        )

    query = st.text_input("Search text", placeholder="Enter search term...")

    if query:
        if search_mode == "Keyword Search":
            with st.spinner(f"Searching for '{query}'..."):
                matches = search_titles(graph, query)

            st.write(f"**Found {len(matches)} matches**")

            if matches:
                # Filter by section
                sections = sorted(set(m["section"] for m in matches))
                selected_sections = st.multiselect(
                    "Filter by section", options=sections, default=sections
                )

                filtered = [m for m in matches if m["section"] in selected_sections]

                # Pagination
                page_size = 50
                total_pages = (len(filtered) + page_size - 1) // page_size

                if total_pages > 1:
                    page_num = (
                        st.number_input(
                            "Page", min_value=1, max_value=total_pages, value=1
                        )
                        - 1
                    )
                else:
                    page_num = 0

                start = page_num * page_size
                end = min(start + page_size, len(filtered))
                page_matches = filtered[start:end]

                # Display
                for match in page_matches:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.code(match["code"])
                            st.caption(f"Section {match['section']}")
                            if match["allocatable"]:
                                st.success("✓ Allocatable")
                        with col2:
                            st.write(match["title"])
                        st.divider()
        else:
            # Semantic search
            if not os.path.exists(EMBEDDINGS_PATH):
                st.error("Embeddings file not found. Cannot perform semantic search.")
                st.info(f"Expected: {EMBEDDINGS_PATH}")
                return

            with st.spinner("Loading semantic model (first time only, ~30s)..."):
                results, error = semantic_search(query, top_k=50)

            if error:
                st.error(f"Semantic search error: {error}")
                return

            if results is None:
                st.error("No results returned from semantic search")
                return

            st.write(f"**Found {len(results)} top semantic matches**")

            if results:
                # Filter by section
                sections = sorted(set(r["section"] for r in results))
                selected_sections = st.multiselect(
                    "Filter by section",
                    options=sections,
                    default=sections,
                    key="semantic_sections",
                )

                filtered = [r for r in results if r["section"] in selected_sections]

                # Display with similarity scores
                for i, result in enumerate(filtered[:50]):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 1, 3])
                        with col1:
                            st.code(result["code"])
                            st.caption(f"Section {result['section']}")
                        with col2:
                            # Similarity score as progress bar
                            sim_pct = min(100, max(0, result["similarity"] * 100))
                            st.progress(sim_pct / 100)
                            st.caption(f"Similarity: {sim_pct:.1f}%")
                        with col3:
                            # Get title from graph
                            if result["code"] in graph:
                                title = graph.nodes[result["code"]].get("title", "N/A")
                                st.write(title)
                            else:
                                st.write("(Title not found in graph)")
                        st.divider()


def show_code_browser(graph):
    st.header("🔎 Code Browser")

    code = (
        st.text_input("Enter CPC Code", placeholder="e.g., G06F17/30").strip().upper()
    )

    if code:
        details = get_code_details(graph, code)

        if details:
            st.subheader(f"Code: {code}")

            # Basic info
            col1, col2, col3 = st.columns(3)
            col1.metric("Type", details["type"])
            col2.metric("Level", details["level"])
            col3.metric("Allocatable", "Yes" if details["allocatable"] else "No")

            st.write(f"**Title:** {details['title']}")

            # Parents
            if details["parents"]:
                st.subheader("Parents")
                for parent in details["parents"]:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.code(parent["code"])
                            st.caption(parent["relation"])
                        with col2:
                            st.write(parent["title"])

            # Children
            if details["children"]:
                st.subheader(f"Children ({len(details['children'])})")

                show_all = st.checkbox(
                    "Show all children", value=len(details["children"]) <= 20
                )

                children_to_show = (
                    details["children"] if show_all else details["children"][:20]
                )

                for child in children_to_show:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.code(child["code"])
                            st.caption(child["relation"])
                        with col2:
                            st.write(child["title"])

                if not show_all:
                    st.info(f"Showing 20 of {len(details['children'])} children")
        else:
            st.error(f"Code '{code}' not found in graph")

            # Suggest similar codes
            similar = [
                n
                for n in graph.nodes()
                if n.startswith(code[:3]) and graph.nodes[n].get("type") == "subgroup"
            ]
            if similar:
                st.subheader("Similar codes found:")
                for s in similar[:10]:
                    st.code(s)


def show_integrity(graph):
    st.header("✅ Integrity Check")

    with st.spinner("Running integrity checks..."):
        issues = check_integrity(graph)

    if issues:
        st.warning(f"Found {len(issues)} issues:")
        for issue in issues:
            st.error(f"• {issue}")
    else:
        st.success("✓ Graph integrity check passed - no issues found!")

    # Detailed checks
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orphaned Subgroups")
        orphaned = []
        for node, data in graph.nodes(data=True):
            if data.get("type") == "subgroup":
                if graph.in_degree(node) == 0 and data.get("level", 0) > 0:
                    orphaned.append(node)

        if orphaned:
            st.error(f"{len(orphaned)} orphaned subgroups")
            st.code("\n".join(orphaned[:20]))
        else:
            st.success("No orphaned subgroups")

    with col2:
        st.subheader("Empty Classes")
        empty = []
        for node, data in graph.nodes(data=True):
            if data.get("type") == "class_3char":
                if graph.out_degree(node) == 0:
                    empty.append(node)

        if empty:
            st.error(f"{len(empty)} empty classes")
            st.code("\n".join(empty))
        else:
            st.success("No empty classes")

    # Embeddings check
    st.subheader("Embeddings Coverage")
    if os.path.exists(EMBEDDINGS_PATH):
        emb_data = np.load(EMBEDDINGS_PATH)
        emb_count = len(emb_data["symbols"])
        sg_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("type") == "subgroup"
        )

        col1, col2 = st.columns(2)
        col1.metric("Subgroups", sg_count)
        col2.metric("Embeddings", emb_count)

        if emb_count == sg_count:
            st.success(f"✓ All {sg_count:,} subgroups have embeddings")
        else:
            diff = abs(emb_count - sg_count)
            st.warning(f"⚠ Mismatch: {diff} difference")
    else:
        st.error("Embeddings file not found")


if __name__ == "__main__":
    main()
