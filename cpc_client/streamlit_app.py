import streamlit as st
import pandas as pd
from mcp_client import CPCRestClient

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="CPC Classifier - Phase 1 Test", page_icon="🔍", layout="wide"
)

client = CPCRestClient(base_url="http://localhost:3456")


# -----------------------------
# UI HEADER
# -----------------------------
st.title("🔍 Phase 1 Test — Technical Terms & CPC Class Extraction")
st.markdown("Paste patent text and optional claims → review Phase 1 extraction quality")

# -----------------------------
# INPUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area(
        "Patent Description / Technical Text",
        height=250,
        placeholder="Enter technical description here... (minimum 100 characters)",
    )

with col2:
    claims_input = st.text_area(
        "Claims (Optional)",
        height=250,
        placeholder="Paste claims here to prioritize claim terms in extraction...",
    )

uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")


# -----------------------------
# ACTION
# -----------------------------
if st.button("🚀 Extract Phase 1"):
    if not text_input.strip():
        st.warning("Please provide input text first.")
        st.stop()

    if len(text_input.strip()) < 100:
        st.warning(
            f"Text too short ({len(text_input.strip())} chars). "
            "Please provide at least 100 characters of patent text."
        )
        st.stop()

    # Issue 10 — meaningful spinner message
    with st.spinner(
        "Extracting technical terms and CPC classes — this may take 30–60 seconds..."
    ):
        # Pass claims separately if provided
        if claims_input.strip():
            result = client.classify_cpc_with_claims(text_input, claims_input)
        else:
            result = client.classify_cpc(text_input)

    # -----------------------------
    # ERROR HANDLING
    # -----------------------------
    if "error" in result:
        st.error(
            f"**Extraction failed.**\n\n"
            f"{result['error']}\n\n"
            "_Make sure both the MCP server (port 3456) and FastAPI backend (port 8000) are running._"
        )
        with st.expander("🔍 Full error details"):
            st.json(result)
        st.stop()

    # -----------------------------
    # PHASE 1 RESULTS ONLY
    # -----------------------------
    st.divider()
    st.subheader("🔍 Phase 1 Extraction Results")

    phase1 = result.get("phase1", {})

    if not phase1:
        st.warning("Phase 1 data not available.")
        st.stop()

    # Technical Object, Problem, and Core Function
    col_obj, col_prob = st.columns(2)
    with col_obj:
        st.markdown("**Technical Object of the Invention:**")
        st.info(phase1.get("technical_object", "N/A"))

    with col_prob:
        st.markdown("**Problem to be Solved:**")
        st.info(phase1.get("problem_solved", "N/A"))

    # Core Function - CRITICAL for function-first classification
    st.markdown("---")
    st.markdown("**🎯 Core Technical Function (Function-First Classification):**")
    core_function = phase1.get("core_function", "")
    if core_function:
        st.success(f"**{core_function}**")
        st.caption(
            "This function drives CPC classification. The model classifies by WHAT the invention DOES, not what it LOOKS LIKE."
        )
    else:
        st.warning(
            "No core function extracted. Classification may be form-based rather than function-based."
        )

    # System Context - CRITICAL for system-first classification
    st.markdown("---")
    st.markdown(
        "**🏭 System / Application Context (System-First Classification - MOST IMPORTANT):**"
    )
    system_context = phase1.get("system_context", "")
    if system_context:
        st.success(f"**{system_context}**")
        st.caption(
            "This is the STRONGEST signal for CPC classification. It identifies the overall technical system or industry domain."
        )
    else:
        st.warning(
            "No system context extracted. Classification may miss the correct application domain."
        )

    # CPC Classes & Sections
    st.markdown("---")
    col_classes, col_sections = st.columns(2)
    with col_classes:
        st.markdown("**Extracted CPC Classes:**")
        cpc_classes = phase1.get("cpc_classes", [])
        if cpc_classes:
            st.code(", ".join(cpc_classes))
        else:
            st.write("No classes extracted")

    with col_sections:
        st.markdown("**CPC Sections:**")
        sections = phase1.get("cpc_sections", [])
        if sections:
            st.code(", ".join(sections))
        else:
            st.write("No sections extracted")

    # Essential Terms Table
    st.markdown("---")
    st.markdown("**Essential Technical Terms (Ranked by Importance):**")

    terms = phase1.get("essential_terms", phase1.get("terms", []))
    if terms:
        df_terms = pd.DataFrame(terms)
        # Ensure expected columns exist
        if "importance" in df_terms.columns:
            df_terms = df_terms.sort_values("importance", ascending=False)

        # Display as a styled dataframe
        st.dataframe(
            df_terms,
            use_container_width=True,
            column_config={
                "term": st.column_config.TextColumn("Technical Term", width="medium"),
                "importance": st.column_config.NumberColumn(
                    "Importance (1-10)", width="small"
                ),
                "justification": st.column_config.TextColumn(
                    "Justification", width="large"
                ),
            },
        )

        # Bar chart of term importance
        if "importance" in df_terms.columns and "term" in df_terms.columns:
            st.subheader("📊 Term Importance Distribution")
            chart_df = df_terms.set_index("term")[["importance"]]
            st.bar_chart(chart_df)
    else:
        st.write("No terms extracted")

    # PHASE 2/3: CPC Results
    st.divider()
    st.subheader("📊 Phase 2 & 3 — CPC Subgroup Ranking (Top 7)")

    phase3 = result.get("phase3", [])
    if phase3:
        st.markdown(
            "CPC subgroups expanded from Phase 1 classes, ranked by **SYSTEM-FIRST and FUNCTION-FIRST keyword matching**. "
            "Subgroups matching the system context and core technical function are heavily prioritized."
        )

        df_p3 = pd.DataFrame(phase3)
        if "symbol" in df_p3.columns and "score" in df_p3.columns:
            df_p3["similarity_%"] = (df_p3["score"] * 100).round(1)

            # Select columns to display
            display_cols = ["symbol", "title", "score", "similarity_%"]
            if "level" in df_p3.columns:
                display_cols.append("level")

            st.dataframe(
                df_p3[display_cols],
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("CPC Code", width="small"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "score": st.column_config.NumberColumn("Score", width="small"),
                    "similarity_%": st.column_config.NumberColumn(
                        "Similarity %", width="small"
                    ),
                    "level": st.column_config.NumberColumn("Level", width="small"),
                },
            )

            st.subheader("📊 CPC Score Distribution")
            st.bar_chart(df_p3.set_index("symbol")[["score"]])
        else:
            st.json(phase3)
    else:
        st.warning("No Phase 3 results available.")

    # PHASE 4: Post-Ranking LLM Re-ranking
    phase4 = result.get("phase4", {})
    if phase4:
        st.divider()
        st.subheader("🎯 Phase 4 — LLM Re-Ranking & Best Code Selection")

        re_ranked = phase4.get("re_ranked", [])
        if re_ranked:
            st.markdown("**Re-Ranked by Patent Examiner LLM:**")
            for item in re_ranked[:5]:
                rank = item.get("rank", "?")
                symbol = item.get("symbol", "?")
                justification = item.get("justification", "")
                st.write(f"**#{rank}** `{symbol}` — {justification}")

        best = phase4.get("best_code", {})
        if best:
            st.markdown("---")
            st.markdown("### 🏆 Best CPC Code")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Code", best.get("symbol", "N/A"))
                st.metric("Confidence", best.get("confidence", "N/A").upper())
            with col2:
                st.markdown(f"**Title:** {best.get('title', 'N/A')}")
                st.markdown(f"**Reasoning:** {best.get('reasoning', 'N/A')}")

    # Raw Phase 1 JSON for debugging
    with st.expander("🔍 Raw Phase 1 JSON"):
        st.json(phase1)

    # Raw Phase 3 JSON for debugging
    with st.expander("🔍 Raw Phase 3 JSON"):
        st.json(phase3)

    # Raw Phase 4 JSON for debugging
    if phase4:
        with st.expander("🔍 Raw Phase 4 JSON"):
            st.json(phase4)
