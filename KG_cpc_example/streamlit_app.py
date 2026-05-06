import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 🔧 Must be the first Streamlit command
st.set_page_config(page_title="CPC KG Explorer", page_icon="🗂️", layout="wide")

# 📦 Sample CPC Data (identical to your mini_cpc_kg.py)
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

def keyword_search(graph, query, top_k=5):
    """Simple keyword matching on CPC titles."""
    query_words = query.lower().split()
    results = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "subgroup": continue
        title = data["title"].lower()
        score = sum(1 for w in query_words if w in title)
        if score > 0:
            results.append({"symbol": node, "title": data["title"], "score": score})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

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

# 🚀 Main App
def main():
    st.title("🗂️ Mini CPC Knowledge Graph Explorer")
    st.caption("Interactive demo of patent classification concepts • Built for beginners")

    # Build graph
    G = build_cpc_graph(CPC_DATA)

    # 📊 Sidebar Stats
    with st.sidebar:
        st.header("📊 Graph Stats")
        st.metric("Total Nodes", G.number_of_nodes())
        st.metric("Total Edges", G.number_of_edges())
        st.metric("Subgroups", sum(1 for _, d in G.nodes(data=True) if d.get("type") == "subgroup"))
        st.metric("Classes", sum(1 for _, d in G.nodes(data=True) if d.get("type") == "class"))
        
        st.divider()
        st.info("💡 **Tip:** Try searching for `dialog`, `query`, or `speech`")

    # 🎯 Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🌳 Hierarchy & Graph", "📋 Data Table"])

    # 🔍 TAB 1: Search
    with tab1:
        st.header("🔍 Keyword Search")
        query = st.text_input("Enter search terms (e.g., 'dialog natural language')", placeholder="Type here...")
        
        if query:
            results = keyword_search(G, query.strip(), top_k=5)
            if results:
                st.success(f"Found {len(results)} match(es)")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            else:
                st.warning("No matches found. Try different keywords!")

    # 🌳 TAB 2: Hierarchy & Visualization
    with tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📍 Explore Node")
            subgroups = [n for n, d in G.nodes(data=True) if d.get("type") == "subgroup"]
            selected = st.selectbox("Select a CPC subgroup:", subgroups)

            if selected:
                path = get_hierarchy_path(G, selected)
                st.markdown(f"**Hierarchy Path:**")
                st.code(" → ".join(path), language="text")
                
                nd = G.nodes[selected]
                st.markdown(f"**Title:** {nd['title']}")
                st.markdown(f"**Level:** {nd['level']}")
                st.markdown(f"**Allocatable:** {'✅ Yes' if nd['is_allocatable'] else '❌ No'}")

        with col2:
            st.subheader("🕸️ Graph Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            pos = nx.spring_layout(G, seed=42, k=0.6)
            
            # Styling
            node_colors = ["skyblue" if G.nodes[n].get("type") == "subgroup" else "lightgreen" for n in G.nodes()]
            current_sel = selected if selected else subgroups[0]
            node_sizes = [4000 if n == current_sel else 2200 for n in G.nodes()]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', edge_color="gray", ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
            
            edge_labels = nx.get_edge_attributes(G, 'relation')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

            ax.set_title("CPC Hierarchy Structure", fontsize=12)
            ax.axis("off")
            st.pyplot(fig)

    # 📋 TAB 3: Data Table
    with tab3:
        st.header("📋 CPC Data Overview")
        rows = []
        for n, d in G.nodes(data=True):
            if d.get("type") == "subgroup":
                parents = list(G.predecessors(n))
                parent = parents[0] if parents else "None"
                rows.append({
                    "Symbol": n, 
                    "Title": d["title"], 
                    "Parent Class": parent, 
                    "Level": d["level"], 
                    "Allocatable": d["is_allocatable"]
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()