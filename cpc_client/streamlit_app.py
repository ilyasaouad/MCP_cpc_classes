import streamlit as st
import pandas as pd
import urllib.request
import urllib.error
import json
from mcp_client import CPCRestClient

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="CPC Classifier - Phase 1 Test", page_icon="[DEBUG]", layout="wide"
)

client = CPCRestClient(base_url="http://localhost:3456")


# -----------------------------
# SYSTEM HEALTH CHECK
# -----------------------------
@st.cache_data(ttl=30)
def check_system_health():
    """Check full system health via backend API."""
    try:
        req = urllib.request.Request("http://localhost:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "error", "error": str(e)}


health = check_system_health()

# -----------------------------
# UI HEADER
# -----------------------------
st.title("[DEBUG] CPC Classification — Full Pipeline")
st.markdown("Paste patent text and optional claims → get structured CPC classification")

# Show system status banner
if health.get("status") == "healthy":
    st.success(
        f"[OK] **System Ready** — Ollama: {health['configured_model']} | "
        f"KG: {health['knowledge_graph']['nodes']} nodes"
    )
elif health.get("status") == "degraded":
    ollama = health.get("ollama", {})
    model = health.get("configured_model", "unknown")

    if not ollama.get("available"):
        st.error(
            "🚨 **Ollama Not Running** — The LLM server is not responding.\n\n"
            "**CPC classification CANNOT work without LLM.**\n\n"
            "**Fix this:**\n"
            "1. Start Ollama server:\n"
            "```bash\nollama serve\n```\n"
            "2. Pull a model (phi4 is 8GB, works on most systems):\n"
            "```bash\nollama pull phi4:latest\n```\n"
            "3. Or use a smaller model in `.env`:\n"
            "```bash\nLLM_MODEL=phi4:latest\n```"
        )
    elif not health.get("model_available"):
        st.error(
            f"🚨 **Model Not Found** — `{model}` is not downloaded.\n\n"
            f"**Fix this:**\n"
            f"```bash\nollama pull {model}\n```\n"
            f"Or change model in `.env` to one you have:\n"
            f"```bash\nLLM_MODEL=phi4:latest\n```"
        )
    else:
        st.warning(
            "[WARN] **System Degraded** — Some components are not working optimally.\n\n"
            f"**Recommendations:**\n"
            + "\n".join(f"- {r}" for r in health.get("recommendations", []))
        )
else:
    st.error(
        "🚨 **Backend Not Responding** — Cannot connect to FastAPI at `localhost:8000`.\n\n"
        "**Fix this:**\n"
        "1. Start the FastAPI backend:\n"
        "```bash\ncd patent_cpc_fastapi\npython -m uvicorn app.main:app --reload --port 8000\n```\n"
        "2. Or check if it's running on a different port"
    )

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
# MANUAL PHASE 1 (Bypass LLM)
# -----------------------------
st.divider()
use_manual = st.checkbox(
    "⚡ Use Manual Phase 1 (bypass LLM - for testing when LLM is unavailable)",
    value=False,
    help="When LLM keeps timing out, manually enter Phase 1 data to test the rest of the pipeline",
)

manual_phase1 = None
if use_manual:
    st.info(
        "[INFO] **Manual Mode:** Enter the semantic understanding directly. "
        "The pipeline will skip LLM extraction and use your inputs for Phase 2-5."
    )

    mp_col1, mp_col2 = st.columns(2)
    with mp_col1:
        tech_obj = st.text_area(
            "Technical Object",
            value="A method for quantizing trained large language models",
            height=80,
        )
        core_func = st.text_area(
            "Core Function",
            value="Model compression through weight clipping and quantization",
            height=80,
        )
        sys_ctx = st.text_area(
            "System Context",
            value="Neural network deployment systems",
            height=80,
        )
    with mp_col2:
        strategy = st.selectbox(
            "Classification Strategy",
            options=["function-first", "system-first", "hybrid"],
            index=0,
        )
        domains = st.text_area(
            "Domain Signals (one per line)",
            value="neural network quantization\nlarge language model\nmodel compression",
            height=80,
        )
        terms_text = st.text_area(
            "Technical Terms (format: term:importance)",
            value="quantization:10\nlarge language model:10\nweight clipping:10\ntransformer:9",
            height=80,
        )

    # Parse manual inputs
    domain_signals = []
    for d in domains.strip().split("\n"):
        if d.strip():
            domain_signals.append(
                {
                    "name": d.strip(),
                    "confidence": 0.9,
                    "evidence": "manual input",
                }
            )

    terms = []
    for t in terms_text.strip().split("\n"):
        if ":" in t:
            parts = t.split(":")
            terms.append(
                {
                    "term": parts[0].strip(),
                    "importance": int(parts[1].strip()),
                    "justification": "manual input",
                    "source_section": "manual",
                }
            )

    manual_phase1 = {
        "technical_object": tech_obj,
        "core_function": core_func,
        "system_context": sys_ctx,
        "classification_strategy": strategy,
        "domain_signals": domain_signals,
        "terms": terms,
        "negative_signals": [],
        "negative_domains": [],
        "negative_reasoning": "",
    }


# -----------------------------
# ACTION
# -----------------------------
button_label = "🚀 Run Classification" if use_manual else "🚀 Extract Phase 1"
if st.button(button_label):
    if not use_manual and not text_input.strip():
        st.warning("Please provide input text first.")
        st.stop()

    if not use_manual and len(text_input.strip()) < 100:
        st.warning(
            f"Text too short ({len(text_input.strip())} chars). "
            "Please provide at least 100 characters of patent text."
        )
        st.stop()

    # Issue 10 — meaningful spinner message
    with st.spinner(
        "Running CPC classification pipeline — this may take 30–60 seconds..."
    ):
        if use_manual and manual_phase1:
            # Send manual Phase 1 to backend
            result = client.classify_cpc("MANUAL_PHASE1:" + json.dumps(manual_phase1))
        elif claims_input.strip():
            result = client.classify_cpc_with_claims(text_input, claims_input)
        else:
            result = client.classify_cpc(text_input)

    # -----------------------------
    # ERROR HANDLING
    # -----------------------------
    if "error" in result:
        error_msg = result["error"]

        # Check if it's an LLM connection issue
        if (
            "Ollama" in error_msg
            or "LLM" in error_msg
            or "timed out" in error_msg.lower()
        ):
            st.error(
                f"🚨 **Phase 1 Extraction Failed — LLM Not Available**\n\n"
                f"{error_msg}\n\n"
                "**To fix this:**\n"
                "1. Start Ollama: `ollama serve`\n"
                "2. Pull a model: `ollama pull phi4:latest`\n"
                "3. Use a smaller model (8GB+ RAM needed for phi4, 48GB+ for 120B models)\n\n"
                "**Without LLM:** CPC classification cannot work. Phase 1 semantic extraction is required."
            )
        else:
            st.error(
                f"**Extraction failed.**\n\n"
                f"{error_msg}\n\n"
                "_Make sure both the MCP server (port 3456) and FastAPI backend (port 8000) are running._"
            )
        with st.expander("[DEBUG] Full error details"):
            st.json(result)
        st.stop()

    # -----------------------------
    # PHASE 1 RESULTS ONLY
    # -----------------------------
    st.divider()
    st.subheader("[DEBUG] Phase 1 Extraction Results")
    st.caption(
        "Method: LLM-based semantic extraction (technical object, core function, domain signals, terms)"
    )

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
    st.markdown("**[TARGET] Core Technical Function (Function-First Classification):**")
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
        "**[FACTORY] System / Application Context (System-First Classification - MOST IMPORTANT):**"
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

    # Domain Signals (NEW - replaces CPC Classes)
    st.markdown("---")
    st.markdown("**[TARGET] Domain Signals:**")
    domain_signals = phase1.get("domain_signals", [])
    if domain_signals:
        for ds in domain_signals[:5]:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                conf = ds.get("confidence", 0)
                st.write(f"- **{name}** (confidence: {conf:.2f})")
    else:
        st.write("No domain signals extracted")

    # Disambiguated Terms (NEW)
    st.markdown("---")
    st.markdown("**[SEARCH] Disambiguated Terms:**")
    disambiguated = phase1.get("disambiguated_terms", [])
    if disambiguated:
        for dt in disambiguated[:5]:
            if isinstance(dt, dict):
                term = dt.get("term", "")
                meaning = dt.get("meaning", "")
                domain = dt.get("domain", "")
                avoid = dt.get("avoid", [])
                st.write(
                    f"- `{term}` → **{meaning}** ({domain}) [avoid: {', '.join(avoid)}]"
                )
    else:
        st.write("No ambiguous terms detected")

    # Primary Domain (NEW)
    st.markdown("---")
    st.markdown("**[FACTORY] Primary Technical Domain:**")
    primary_domain = phase1.get("primary_domain", {})
    if primary_domain and isinstance(primary_domain, dict):
        pd_name = primary_domain.get("name", "N/A")
        pd_cpc = primary_domain.get("cpc_class", "N/A")
        pd_conf = primary_domain.get("confidence", 0)
        st.success(f"**{pd_name}** → `{pd_cpc}` (confidence: {pd_conf:.2f})")
        st.caption(f"Reasoning: {primary_domain.get('reasoning', 'N/A')}")
    else:
        st.warning("Primary domain not detected")

    # Classification Strategy
    st.markdown("---")
    strategy = phase1.get("classification_strategy", "")
    if strategy:
        st.markdown(f"**[CHART] Classification Strategy:** `{strategy}`")

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
            st.subheader("[CHART] Term Importance Distribution")
            chart_df = df_terms.set_index("term")[["importance"]]
            st.bar_chart(chart_df)
    else:
        st.write("No terms extracted")

    # PHASE 2: CPC Expansion Pipeline (2A, 2B, 2C)
    phase2 = result.get("phase2", {})

    # ── Phase 2A ──
    st.divider()
    st.subheader("[CHART] Phase 2A — CPC Family Router")
    st.caption(
        "Method: Domain taxonomy with purpose/tool distinction + hard constraints + co-occurrence rules"
    )
    if phase2 and phase2.get("phase2a_families"):
        st.markdown(
            "Routes the patent to relevant CPC families using **domain signals** (purpose vs tool distinction)."
        )
        col_fam, col_prim, col_mod = st.columns(3)
        with col_fam:
            st.markdown("**Selected Families:**")
            families = phase2.get("phase2a_families", [])
            for fam in families[:5]:
                st.code(fam)
        with col_prim:
            st.markdown("**Primary Family:**")
            st.success(f"**{phase2.get('phase2a_primary', 'N/A')}**")
        with col_mod:
            st.markdown("**Modality:**")
            st.info(phase2.get("phase2a_modality", "unknown"))
        st.caption(
            f"Source: {phase2.get('phase2a_source', 'unknown')} | "
            f"Reasoning: {phase2.get('phase2a_reasoning', '')}"
        )
    else:
        st.warning(
            "[WARN] **Phase 2A results not available.**\n\n"
            "CPC family routing did not complete. Classification may be inaccurate."
        )

    # ── Phase 2B ──
    st.markdown("---")
    st.subheader("[CHART] Phase 2B — Restricted XML Expansion")
    st.caption(
        "Method: CPC XML parser with family prefix filtering (98% search space reduction)"
    )
    if phase2:
        count_2b = phase2.get("phase2b_candidate_count", 0)
        families = phase2.get("phase2a_families", [])
        st.markdown(
            "Expands CPC subgroup definitions **only within Phase 2A families** to reduce search space."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expanded Candidates", count_2b)
        with col2:
            reduction = (
                f"~{((1 - count_2b / 250000) * 100):.1f}%" if count_2b > 0 else "N/A"
            )
            st.metric("Search Space Reduction", reduction)
        if families:
            st.caption(f"Restricted to families: {', '.join(families)}")
    else:
        st.warning("Phase 2B data not available.")

    # ── Phase 2C ──
    st.markdown("---")
    st.subheader("[CHART] Phase 2C — TF-IDF Scoring & Filtering")
    st.caption(
        "Method: TF-IDF with importance weighting + synonym expansion + cross-domain guardrails"
    )
    if phase2:
        count_2c = phase2.get("phase2c_final_count", 0)
        margin = phase2.get("score_margin", 0)
        confidence = phase2.get("confidence_level", "unknown")
        st.markdown(
            "Scores expanded candidates using **TF-IDF + domain boosting + false-friend penalty**."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Candidates", count_2c)
        with col2:
            st.metric("Score Margin", f"{margin:.3f}")
        with col3:
            conf_color = (
                "[HIGH]"
                if confidence == "high"
                else "[MED]"
                if confidence == "medium"
                else "[LOW]"
            )
            st.metric("Confidence", f"{conf_color} {confidence.upper()}")
        st.caption(
            "Higher margin = clearer separation between top candidates. "
            "Low margin suggests ambiguous classification."
        )
    else:
        st.warning("Phase 2C data not available.")

    # PHASE 3: CPC Results
    st.divider()
    st.subheader("[CHART] Phase 3 — CPC Subgroup Ranking (Top 10)")
    st.caption(
        "Method: Normalized TF-IDF ranking with system-first and function-first context boosts"
    )

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

            st.subheader("[CHART] CPC Score Distribution")
            st.bar_chart(df_p3.set_index("symbol")[["score"]])
        else:
            st.json(phase3)
    else:
        st.warning("No Phase 3 results available.")

    # PHASE 3.5: Decision Tree Constraint Layer
    phase35 = result.get("phase35", {})
    if phase35:
        st.divider()
        st.subheader("[TREE] Phase 3.5 — Decision Tree Constraint Layer")
        st.caption(
            "Method: Multi-step deterministic decision tree enforcing domain correctness, disambiguation, functional boosting, and invalid class filtering"
        )

        # Decision Tree Steps Display
        with st.expander("[INFO] Decision Tree Steps (Phase 3.5)", expanded=False):
            st.markdown("""
            **Step 1: Domain Detection**
            - Detect primary domain from Phase 1 signals
            - Map to CPC family (G06N, G06T, H04L, etc.)
            
            **Step 2: Domain Dominance**
            - Boost matching domain candidates (×2.0)
            - Penalize unrelated domains (×0.3)
            
            **Step 3: Object-Aware Disambiguation**
            - Resolve ambiguous terms using context
            - Example: "weight clipping" → G06N (not G06T)
            
            **Step 4: Functional Boosting**
            - Match core function to CPC subgroups
            - Boost specific subgroups
            
            **Step 5: Invalid Class Filtering**
            - Remove clearly wrong classes
            - Example: G06F2207 (BCD) for AI patents
            
            **Step 6: Normalization**
            - Re-scale scores after all adjustments
            """)

        domain = phase35.get("phase35_domain", "unknown")
        confidence = phase35.get("phase35_domain_confidence", 0)
        adjustments = phase35.get("phase35_adjustments", 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Domain", domain.upper())
        with col2:
            st.metric("Domain Confidence", f"{confidence:.2f}")
        with col3:
            st.metric("Rules Applied", adjustments)

        # Show adjusted candidates
        adjusted = phase35.get("phase35_candidates", [])
        if adjusted:
            st.markdown("**Adjusted Candidates (After Decision Tree):**")
            df_35 = pd.DataFrame(adjusted)
            if "symbol" in df_35.columns and "score" in df_35.columns:
                st.dataframe(
                    df_35[["symbol", "title", "score"]],
                    use_container_width=True,
                )

        # Show rules log
        rules_log = phase35.get("phase35_rules_log", [])
        if rules_log:
            with st.expander("[DEBUG] Decision Tree Rules Log"):
                st.markdown("**Applied Rules:**")
                for rule in rules_log[:20]:
                    st.write(
                        f"- **{rule.get('rule', '')}**: `{rule.get('symbol', '')}` "
                        f"{rule.get('score_before', 0):.3f} → {rule.get('score_after', 0):.3f} "
                        f"| {rule.get('reason', '')}"
                    )
    else:
        st.divider()
        st.subheader("[TREE] Phase 3.5 — Decision Tree Constraint Layer")
        st.warning("Phase 3.5 results not available.")

    # PHASE 3.6: Universal CPC Hierarchy Selection
    phase36 = result.get("phase36", {})
    if phase36:
        st.divider()
        st.subheader("[TREE] Phase 3.6 — Universal CPC Hierarchy Selection")
        st.caption(
            "Method: Contribution-type-first, domain-second, subclass-mapped using universal A-F hierarchy"
        )

        # Decision Tree Steps Display
        with st.expander("[INFO] Decision Tree Steps (Phase 3.6)", expanded=False):
            st.markdown("""
            **Step 1: Detect Contribution Types (Universal A-F)**
            - A: Parameter/Structure Optimization (priority 1)
            - B: Compression/Efficiency/Reduction (priority 2)
            - C: System Architecture/Design (priority 3)
            - E: Signal/Data Transformation (priority 4)
            - D: Operation/Execution/Inference (priority 5)
            - F: Abstract Modeling/Logic/Reasoning (priority 6)
            
            **Step 2: Select Primary Contribution**
            - Choose highest priority detected type
            - If A or B exists → NEVER select D or F as primary
            
            **Step 3: Map to CPC Patterns**
            - A → parameter optimization subclasses
            - B → compression/reduction subclasses
            - etc.
            
            **Step 4: Domain-Aware Refinement**
            - Apply domain-specific CPC prefix mapping
            - Example: B + AI domain → G06N3/063
            
            **Step 5: Score Adjustment**
            - Boost matching candidates (×2.5)
            - Penalize lower-priority matches (×0.15)
            """)

        primary_type = phase36.get("phase36_primary_type", "unknown")
        secondary_types = phase36.get("phase36_secondary_types", [])
        detected_types = phase36.get("phase36_types", {})
        adjustments = phase36.get("phase36_adjustments", 0)

        # Display contribution type hierarchy
        st.markdown("**Contribution Type Hierarchy:**")
        col_type, col_sec = st.columns(2)
        with col_type:
            st.success(f"**Primary:** {primary_type}")
        with col_sec:
            if secondary_types:
                st.info(f"**Secondary:** {', '.join(secondary_types)}")
            else:
                st.info("**Secondary:** None")

        # Show detected signals
        if detected_types:
            st.markdown("**Detected Signals by Type:**")
            for type_code, signals in detected_types.items():
                st.write(f"- **{type_code}**: {', '.join(signals)}")

        st.metric("Hierarchy Adjustments", adjustments)

        # Show refined candidates with contribution match badges
        refined = phase36.get("phase36_candidates", [])
        if refined:
            st.markdown("**Refined Candidates (After Hierarchy Selection):**")
            for c in refined[:5]:
                match_type = c.get("contribution_match", "neutral")
                badge = {
                    "primary": "[PRIMARY MATCH]",
                    "secondary": "[SECONDARY]",
                    "lower": "[LOWER PRIORITY]",
                    "neutral": "",
                }.get(match_type, "")
                st.write(f"`{c['symbol']}` — {c['score']:.4f} {badge}")
    else:
        st.divider()
        st.subheader("[TREE] Phase 3.6 — Universal CPC Hierarchy Selection")
        st.warning("Phase 3.6 results not available.")

    # PHASE 4: Hypothesis Consolidation
    phase4 = result.get("phase4", {})
    if phase4:
        st.divider()
        st.subheader("[PUZZLE] Phase 4 — Hypothesis Consolidation")
        st.caption(
            "Method: Jaccard clustering by CPC family with coherence scoring + normalized mean scores"
        )
        st.markdown(
            "Clusters Phase 3 candidates into max 2 hypotheses (PRIMARY + optional SECONDARY)."
        )

        hypotheses = phase4.get("phase4_hypotheses", [])
        primary_family = phase4.get("phase4_primary_family", "")
        confidence = phase4.get("phase4_confidence", "low")
        support_weight = phase4.get("phase4_support_weight", 0.0)

        if hypotheses:
            # Show hypotheses
            for hyp in hypotheses:
                role = hyp.get("role", "unknown").upper()
                family = hyp.get("family", "")
                score = hyp.get("normalized_score", 0)
                count = hyp.get("candidate_count", 0)
                coherence = hyp.get("coherence", 0)
                reasoning = hyp.get("reasoning", "")
                codes = hyp.get("supporting_codes", [])[:5]

                if role == "PRIMARY":
                    st.success(
                        f"**PRIMARY: {family}** — score={score:.3f}, "
                        f"candidates={count}, coherence={coherence:.2f}"
                    )
                else:
                    st.info(
                        f"**SECONDARY: {family}** — score={score:.3f}, "
                        f"candidates={count}, coherence={coherence:.2f}"
                    )

                st.caption(f"{reasoning}")
                if codes:
                    st.code(", ".join(codes))
                st.markdown("---")

            # Summary metrics
            col_sw, col_conf = st.columns(2)
            with col_sw:
                st.metric("Support Weight", f"{support_weight:.2%}")
            with col_conf:
                st.metric("Confidence", confidence.upper())

            st.caption(phase4.get("phase4_reasoning", ""))
        else:
            st.warning("No hypotheses formed from candidates.")
    else:
        st.divider()
        st.subheader("[PUZZLE] Phase 4 — Hypothesis Consolidation")
        st.warning("Phase 4 results not available.")

    # PHASE 5: Hypothesis Resolution (new deterministic format)
    phase5 = result.get("phase5", {})
    if phase5 and "primary" in phase5:
        st.divider()
        st.subheader("[TARGET] Phase 5 — Deterministic Hypothesis Resolution")
        st.caption(
            "Method: Weighted scoring (0.5*phase4 + 0.3*functional_alignment + 0.2*technical_coverage)"
        )
        st.markdown(
            "Resolves Phase 4 hypotheses into final CPC selection using deterministic scoring."
        )

        primary = phase5.get("primary", {})
        secondary = phase5.get("secondary", {})
        decision = phase5.get("decision_logic", {})

        # Primary
        if primary:
            st.markdown("### [BEST] Primary Family")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Family", primary.get("family", "N/A"))
                st.metric("Confidence", primary.get("confidence", "N/A").upper())
                st.metric("Final Score", f"{primary.get('final_score', 0):.3f}")
            with col2:
                st.markdown(f"**Phase 4 Score:** {primary.get('phase4_score', 0):.3f}")
                st.markdown(
                    f"**Functional Alignment:** {primary.get('functional_alignment', 0):.3f}"
                )
                st.markdown(
                    f"**Technical Coverage:** {primary.get('technical_coverage', 0):.3f}"
                )
                st.markdown(
                    f"**Specificity Match:** {primary.get('specificity_match', 0):.3f}"
                )
                st.markdown(f"**Reasoning:** {primary.get('reasoning', 'N/A')}")
                codes = primary.get("supporting_codes", [])
                if codes:
                    st.markdown(f"**Supporting Codes:** `{', '.join(codes[:5])}`")

        # Secondary
        if secondary:
            st.markdown("---")
            st.markdown("### [2ND] Secondary Family (accepted)")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Family", secondary.get("family", "N/A"))
                st.metric("Confidence", secondary.get("confidence", "N/A").upper())
                st.metric("Final Score", f"{secondary.get('final_score', 0):.3f}")
            with col2:
                st.markdown(
                    f"**Phase 4 Score:** {secondary.get('phase4_score', 0):.3f}"
                )
                st.markdown(
                    f"**Functional Alignment:** {secondary.get('functional_alignment', 0):.3f}"
                )
                st.markdown(
                    f"**Technical Coverage:** {secondary.get('technical_coverage', 0):.3f}"
                )
                st.markdown(
                    f"**Specificity Match:** {secondary.get('specificity_match', 0):.3f}"
                )
                st.markdown(f"**Reasoning:** {secondary.get('reasoning', 'N/A')}")
                codes = secondary.get("supporting_codes", [])
                if codes:
                    st.markdown(f"**Supporting Codes:** `{', '.join(codes[:5])}`")
        else:
            st.markdown("---")
            st.info("[INFO] No secondary family — gap was too large.")

        # Decision logic
        if decision:
            st.markdown("---")
            st.markdown("**Decision Logic:**")
            st.write(
                f"- Score gap: {decision.get('score_gap', 0):.3f} "
                f"(secondary accepted: {decision.get('secondary_accepted', False)})"
            )
            st.write(
                f"- Hypotheses evaluated: {decision.get('num_hypotheses_evaluated', 0)}"
            )
            st.write(f"- Method: {decision.get('selection_method', 'unknown')}")

    # Backward compatibility: old Phase 5 format
    elif phase5:
        st.divider()
        st.subheader("[TARGET] Phase 5 — Multi-Pass Validation & Best Code Selection")

        validated = phase5.get("validated_candidates", [])
        filtered = phase5.get("filtered_out", [])

        if validated:
            st.markdown("**[OK] Validated Candidates:**")
            for item in validated[:5]:
                symbol = item.get("symbol", "?")
                confidence = item.get("confidence", "medium")
                justification = item.get("justification", "")
                st.write(f"`{symbol}` — **{confidence.upper()}** — {justification}")

        if filtered:
            st.markdown("**❌ Filtered Out:**")
            for item in filtered[:3]:
                symbol = item.get("symbol", "?")
                reason = item.get("rejection_reason", "")
                st.write(f"`{symbol}` — {reason}")

        best = phase5.get("best_code", {})
        if best:
            st.markdown("---")
            st.markdown("### [BEST] Best CPC Code")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Code", best.get("symbol", "N/A"))
                st.metric("Confidence", best.get("confidence", "N/A").upper())
            with col2:
                st.markdown(f"**Title:** {best.get('title', 'N/A')}")
                st.markdown(f"**Reasoning:** {best.get('reasoning', 'N/A')}")

    # PREMIER: Single best validated CPC class
    premier = result.get("premier", {})
    if premier:
        st.divider()
        st.subheader("[BEST] Premier CPC Classification")
        st.caption("Method: Phase 7 consistency recommendation override (if coherent)")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Code", premier.get("symbol", "N/A"))
            st.metric("Confidence", premier.get("confidence", "N/A").upper())
        with col2:
            st.markdown(f"**Title:** {premier.get('title', 'N/A')}")
            st.markdown(f"**Reasoning:** {premier.get('reasoning', 'N/A')}")

    # PHASE 6: Per-Claim Classification
    per_claim = result.get("per_claim", [])
    if per_claim:
        st.divider()
        st.subheader("[LIST] Phase 6 — Per-Claim Classification")
        st.caption(
            "Method: LLM-based claim-level CPC assignment + reconciliation with validated codes"
        )
        for claim in per_claim[:10]:
            claim_num = claim.get("claim_number", "?")
            codes = claim.get("cpc_codes", [])
            codes_str = ", ".join(codes) if codes else "N/A"
            st.write(f"**Claim {claim_num}:** `{codes_str}`")
    else:
        st.divider()
        st.subheader("[LIST] Phase 6 — Per-Claim Classification")
        st.caption(
            "Method: LLM-based claim-level CPC assignment + reconciliation with validated codes"
        )
        st.warning(
            "**Phase 6 results not available.**\n\n"
            "Per-claim reconciliation did not complete. "
            "Individual claim classifications could not be determined."
        )

    # PHASE 7: Consistency Check
    phase7 = result.get("phase7", {})
    if phase7:
        st.divider()
        st.subheader("[OK] Phase 7 — Final Consistency Check")
        st.caption(
            "Method: LLM-based coherence check + conflict detection + final recommendation"
        )
        coherent = phase7.get("coherent", True)
        if coherent:
            st.success("[OK] Classifications are coherent")
        else:
            st.warning("[WARN] Inconsistencies detected")

        issues = phase7.get("issues", [])
        if issues:
            st.markdown("**Issues:**")
            for issue in issues:
                st.write(f"- {issue}")

        rec_primary = phase7.get("recommended_primary", "")
        if rec_primary:
            st.markdown(f"**Recommended Primary:** `{rec_primary}`")

        rec_secondary = phase7.get("recommended_secondary", [])
        if rec_secondary:
            st.markdown(f"**Recommended Secondary:** {', '.join(rec_secondary)}")

    # Raw JSON for debugging
    with st.expander("[DEBUG] Raw Phase 1 JSON"):
        st.json(phase1)

    with st.expander("[DEBUG] Raw Phase 3 JSON"):
        st.json(phase3)

    if phase4:
        with st.expander("[DEBUG] Raw Phase 4 JSON"):
            st.json(phase4)

    if phase5:
        with st.expander("[DEBUG] Raw Phase 5 JSON"):
            st.json(phase5)

    if phase7:
        with st.expander("[DEBUG] Raw Phase 7 JSON"):
            st.json(phase7)
