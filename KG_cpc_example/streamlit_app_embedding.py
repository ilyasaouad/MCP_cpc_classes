#!/usr/bin/env python3
"""
CPC Semantic Search with Embeddings
Uses Sentence-BERT to match patent concepts by meaning, not just keywords.
"""

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🔧 Must be the first Streamlit command
st.set_page_config(page_title="CPC Semantic Search", page_icon="🧠", layout="wide")

# 📦 Mini CPC Dataset (same structure as your original code)
CPC_DATA = [
    {"symbol": "G06F16/3329", "title": "Natural language dialog systems", "level": 4, "parent_chain": ["G06F", "G06F16", "G06F16/332"], "is_allocatable": True},
    {"symbol": "G06F16/332", "title": "Query processing", "level": 3, "parent_chain": ["G06F", "G06F16"], "is_allocatable": False},
    {"symbol": "G06F16/30", "title": "Information retrieval; Database structures", "level": 2, "parent_chain": ["G06F"], "is_allocatable": False},
    {"symbol": "G06F40/30", "title": "Semantic analysis; Natural language understanding", "level": 3, "parent_chain": ["G06F"], "is_allocatable": True},
    {"symbol": "G10L15/22", "title": "Procedures used during interactive speech recognition", "level": 3, "parent_chain": ["G10L"], "is_allocatable": True},
]

# 🛠️ Helper Functions
def build_cpc_graph(data):
    """Build NetworkX graph from CPC data."""
    G = nx.DiGraph()
    for item in data:
        symbol, title = item["symbol"], item["title"]
        G.add_node(symbol, type="subgroup", title=title, level=item["level"], is_allocatable=item["is_allocatable"])
        if item["parent_chain"]:
            parent = item["parent_chain"][-1]
            if not G.has_node(parent):
                G.add_node(parent, type="class", level=item["level"]-1)
            G.add_edge(parent, symbol, relation="parent_of")
    return G

def get_hierarchy_path(graph, symbol):
    """Trace from subgroup up to top-level class."""
    path = [symbol]
    current = symbol
    while True:
        parents = list(graph.predecessors(current))
        if not parents: break
        current = parents[0]
        path.insert(0, current)
    return path

# 🚀 Streamlit App
def main():
    st.title("🧠 CPC Semantic Search with Embeddings")
    st.caption("Uses `all-mpnet-base-v2` (Sentence-BERT) to find patent classes based on meaning, not just keywords.")

    # Initialize heavy objects once
    if "initialized" not in st.session_state:
        with st.spinner("🤖 Loading AI model & generating embeddings... (takes ~5-10s on first run)"):
            # Load model (cached across reruns)
            model = SentenceTransformer("all-mpnet-base-v2")
            
            # Build graph & generate embeddings
            G = build_cpc_graph(CPC_DATA)
            symbols = [n for n, d in G.nodes(data=True) if d.get("type") == "subgroup"]
            texts = [G.nodes[s]["title"] for s in symbols]
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.G = G
            st.session_state.symbols = symbols
            st.session_state.texts = texts
            st.session_state.embeddings = embeddings
            st.session_state.initialized = True

    G = st.session_state.G
    model = st.session_state.model
    symbols = st.session_state.symbols
    texts = st.session_state.texts
    embeddings = st.session_state.embeddings

    # 🔍 Search UI
    st.subheader("🔍 Semantic Search")
    query = st.text_input(
        "Enter a patent concept or technical query:", 
        placeholder="e.g., 'AI chatbot for customer support' or 'speech recognition dialog'",
        key="semantic_query"
    )

    if query.strip():
        # Encode query
        query_emb = model.encode([query.strip()], convert_to_numpy=True)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_emb, embeddings)[0]
        
        # Get top-5 results
        top_indices = np.argsort(similarities)[::-1][:5]
        results = []
        for idx in top_indices:
            results.append({
                "CPC Symbol": symbols[idx],
                "Title": texts[idx],
                "Similarity": f"{similarities[idx]:.2%}"
            })
        
        st.success(f"✅ Found {len(results)} semantically relevant CPC classes")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # Show hierarchy for top match
        if results:
            top_symbol = results[0]["CPC Symbol"]
            st.markdown("---")
            st.subheader(f"📍 Top Match: `{top_symbol}`")
            path = get_hierarchy_path(G, top_symbol)
            st.markdown(f"**Hierarchy Path:** `{' → '.join(path)}`")
            st.info(f"**Title:** {G.nodes[top_symbol]['title']} | **Allocatable:** {'✅' if G.nodes[top_symbol]['is_allocatable'] else '❌'}")

    # 🕸️ Graph Visualization
    st.markdown("---")
    st.subheader("🕸️ CPC Hierarchy Structure")
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    node_colors = ["skyblue" if G.nodes[n].get("type") == "subgroup" else "lightgreen" for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, 
            font_size=8, arrows=True, edge_color="gray", ax=ax)
    ax.set_title("Mini CPC Knowledge Graph")
    ax.axis("off")
    st.pyplot(fig)

if __name__ == "__main__":
    main()