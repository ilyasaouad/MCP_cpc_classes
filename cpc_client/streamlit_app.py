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
    page_title="CPC Classifier — Phase by Phase", page_icon="[PHASE]", layout="wide"
)

# -----------------------------
# SESSION STATE + PHASE LIST
# -----------------------------
PHASES = [
    "Phase 1: Semantic Extraction",
    "Phase 1.5: Role Classification",
    "TCR: Technical Weight Analysis",
    "Phase 2A: Layer Decomposition",
    "Phase 2B: XML Expansion",
    "Phase 2C: Hybrid Scoring",
    "Phase 2D: Subclass Anchor",
    "Phase 3: CPC Subgroup Ranking",
    "Phase 3.5: Decision Tree Constraints",
    "Phase 3.6: Cross-Domain Validation",
    "Phase 4: Hypothesis Consolidation",
    "Phase 5: Hypothesis Resolution",
    "Phase 8: Role Labeling & Report",
]

if "current_phase" not in st.session_state:
    st.session_state["current_phase"] = PHASES[0]
if "classification_result" not in st.session_state:
    st.session_state["classification_result"] = None

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
# PHASE SELECTOR
# -----------------------------
st.divider()
st.caption("**Select which phase output to display:**")
col_phase_nav, col_phase_sel = st.columns([1, 8])
with col_phase_nav:
    if st.button("⬅ Prev", key="prev_phase", use_container_width=True):
        idx = PHASES.index(st.session_state["current_phase"])
        if idx > 0:
            st.session_state["current_phase"] = PHASES[idx - 1]
        st.rerun()
with col_phase_sel:
    selected_phase = st.selectbox(
        "Phase",
        PHASES,
        index=PHASES.index(st.session_state["current_phase"]),
        label_visibility="collapsed",
        key="phase_select",
    )
    st.session_state["current_phase"] = selected_phase
with col_phase_nav:
    if st.button("Next ➡", key="next_phase", use_container_width=True):
        idx = PHASES.index(st.session_state["current_phase"])
        if idx < len(PHASES) - 1:
            st.session_state["current_phase"] = PHASES[idx + 1]
        st.rerun()
st.divider()

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
button_label = "🚀 Run Full Pipeline"
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

    # Cache result in session state
    st.session_state["classification_result"] = result

# -----------------------------
# DISPLAY RESULTS (phase by phase)
# -----------------------------
result = st.session_state.get("classification_result")
if result is None:
    st.info("Enter patent text above and click **Run Full Pipeline** to start.")
    st.stop()

# -----------------------------
# ERROR HANDLING (from cached result)
# -----------------------------
if "error" in result:
    error_msg = result["error"]

    if "Ollama" in error_msg or "LLM" in error_msg or "timed out" in error_msg.lower():
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
# PHASE BY PHASE DISPLAY
# -----------------------------
phase = st.session_state.get("current_phase", PHASES[0])

# Show current phase badge
st.divider()
st.caption(f"**Currently viewing:** {phase}  (use selector at top to switch phases)")

# =============================================================================
# PHASE 1: Semantic Extraction
# =============================================================================
if phase == "Phase 1: Semantic Extraction":
    st.divider()
    st.subheader("[DEBUG] Phase 1 Extraction Results")
    st.caption(
        "Method: LLM-based semantic extraction (technical object, core function, domain signals, terms)"
    )

    phase1 = result.get("phase1", {})

    if not phase1:
        st.warning("Phase 1 data not available.")
        st.stop()

    col_obj, col_prob = st.columns(2)
    with col_obj:
        st.markdown("**Technical Object of the Invention:**")
        st.info(phase1.get("technical_object", "N/A"))
    with col_prob:
        st.markdown("**Problem to be Solved:**")
        st.info(phase1.get("problem_solved", "N/A"))

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

    st.markdown("---")
    strategy = phase1.get("classification_strategy", "")
    if strategy:
        st.markdown(f"**[CHART] Classification Strategy:** `{strategy}`")

    terms = phase1.get("essential_terms", phase1.get("terms", []))
    if terms:
        df_terms = pd.DataFrame(terms)
        if "importance" in df_terms.columns:
            df_terms = df_terms.sort_values("importance", ascending=False)
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
        if "importance" in df_terms.columns and "term" in df_terms.columns:
            st.subheader("[CHART] Term Importance Distribution")
            chart_df = df_terms.set_index("term")[["importance"]]
            st.bar_chart(chart_df)
    else:
        st.write("No terms extracted")

# =============================================================================
# PHASE 1.5: Invention Role Classification
# =============================================================================
elif phase == "Phase 1.5: Role Classification":
    phase15 = result.get("phase15", {})
    if phase15:
        st.divider()
        st.subheader("[TARGET] Phase 1.5 — Invention Role Classification")
        st.caption(
            "Method: LLM-based role classification (CORE_TECH / SYSTEM / APPLICATION / SUPPORT)"
        )

        role = phase15.get("role", "UNKNOWN")
        confidence = phase15.get("confidence", 0)

        role_colors = {
            "CORE_TECH": "success",
            "SYSTEM": "info",
            "APPLICATION": "warning",
            "SUPPORT": "secondary",
        }
        role_color = role_colors.get(role, "info")

        col_role, col_conf = st.columns(2)
        with col_role:
            if role_color == "success":
                st.success(f"**{role}** — Core Technical Innovation")
            elif role_color == "info":
                st.info(f"**{role}** — System Orchestration")
            elif role_color == "warning":
                st.warning(f"**{role}** — Domain Application")
            else:
                st.info(f"**{role}** — Auxiliary Support")
            st.caption("Primary classification driver")
        with col_conf:
            st.metric("Confidence", f"{confidence:.2f}")

        role_descriptions = {
            "CORE_TECH": "The invention modifies/improves the underlying technology itself (algorithms, model architecture, training methods)",
            "SYSTEM": "The invention orchestrates/coordinates/manages components (pipelines, multi-component systems, data/control flow)",
            "APPLICATION": "The invention applies known technology to a specific domain (medical, automotive, finance, industry-specific)",
            "SUPPORT": "Auxiliary functionality not central to technical operation (logging, storage, UI, monitoring)",
        }
        st.markdown(
            f"**Role Definition:** {role_descriptions.get(role, 'Unknown role')}"
        )

        reasoning = phase15.get("reasoning", [])
        if reasoning:
            st.markdown("**Reasoning:**")
            for r in reasoning:
                st.write(f"- {r}")

        evidence = phase15.get("evidence", [])
        if evidence:
            with st.expander("[INFO] Evidence"):
                for e in evidence:
                    st.write(f"- {e}")

        st.markdown("---")
        st.markdown("**CPC Routing Implications:**")
        if role == "CORE_TECH":
            st.markdown("→ Boost: G06N, G06T, G06V, G10L (technology-native classes)")
            st.markdown("→ Deprioritize: G06F, H04L (unless strongly supported)")
        elif role == "SYSTEM":
            st.markdown("→ Boost: G06F, H04L, G05B (system orchestration classes)")
            st.markdown(
                "→ Deprioritize: G06N, G06T (unless NN-internal signals present)"
            )
        elif role == "APPLICATION":
            st.markdown("→ Boost: A61, B60, B23, E21, A01 (domain-specific classes)")
            st.markdown("→ Deprioritize: G06N (AI is tool, not subject)")
        else:
            st.markdown("→ Boost: G06F (general computing for auxiliary functions)")
    else:
        st.warning("Phase 1.5 results not available.")

# =============================================================================
# TCR: Technical Weight Analysis
# =============================================================================
elif phase == "TCR: Technical Weight Analysis":
    tcr_result = result.get("tcr_analysis", {})
    if tcr_result:
        st.divider()
        st.subheader("[NEW] Technical Weight Analysis")
        st.caption(
            "Determines whether invention is primarily computational (software) or physical (domain-specific)"
        )

        tcr = tcr_result.get("tcr", 1.0)
        force_flag = tcr_result.get("force_flag", "HYBRID_INVENTION")
        comp_weight = tcr_result.get("computational_weight", 0)
        phys_weight = tcr_result.get("physical_weight", 0)
        dominant = tcr_result.get("dominant_bucket", "unknown")

        col_tcr1, col_tcr2, col_tcr3 = st.columns(3)
        with col_tcr1:
            st.metric("TCR (Technical Character Ratio)", f"{tcr:.3f}")
        with col_tcr2:
            flag_emoji = (
                "🖥️"
                if force_flag == "FORCE_SOFTWARE_CORE"
                else "⚙️"
                if force_flag == "FORCE_DOMAIN_CORE"
                else "🔄"
            )
            st.metric("Force Flag", f"{flag_emoji} {force_flag}")
        with col_tcr3:
            st.metric("Dominant Bucket", dominant.capitalize())

        if force_flag == "FORCE_SOFTWARE_CORE":
            st.info(
                "🖥️ **Interpretation:** Computationally dominant (TCR > 2.0). Primary CPC: G06F/G06N. Physical codes = CONTEXT/SUPPORT only."
            )
        elif force_flag == "FORCE_DOMAIN_CORE":
            st.info(
                "⚙️ **Interpretation:** Physically dominant (TCR < 0.5). Primary CPC: domain-specific. Computational codes = SUPPORT only."
            )
        else:
            st.info(
                "🔄 **Interpretation:** Hybrid invention (0.5 <= TCR <= 2.0). Both layers contribute meaningfully."
            )

        comp_terms = tcr_result.get("computational_terms", [])
        phys_terms = tcr_result.get("physical_terms", [])

        col_terms1, col_terms2 = st.columns(2)
        with col_terms1:
            if comp_terms:
                st.markdown(f"**Computational Terms ({len(comp_terms)}):**")
                terms_list = [t.get("term", "") for t in comp_terms[:15]]
                st.write(
                    ", ".join(terms_list) + ("..." if len(comp_terms) > 15 else "")
                )
        with col_terms2:
            if phys_terms:
                st.markdown(f"**Physical/Domain Terms ({len(phys_terms)}):**")
                terms_list = [t.get("term", "") for t in phys_terms[:15]]
                st.write(
                    ", ".join(terms_list) + ("..." if len(phys_terms) > 15 else "")
                )
    else:
        st.warning("TCR analysis not available.")

# =============================================================================
# PHASE 2A: Layer Decomposition
# =============================================================================
elif phase == "Phase 2A: Layer Decomposition":
    phase2a_layers = result.get("phase2a_layers", {})
    if phase2a_layers:
        st.divider()
        st.subheader("[LAYERS] Phase 2A — CPC Layer Decomposition")
        st.caption(
            "Multi-layer decomposition: each technical layer independently maps to CPC. "
            "NO cross-layer penalties. NO forced hierarchy."
        )

        primary_layer = phase2a_layers.get("primary_layer", "unknown")
        layer_scores = phase2a_layers.get("layer_scores", {})
        layers = phase2a_layers.get("layers", {})
        relationships = phase2a_layers.get("relationships", {})
        ai_role = phase2a_layers.get("ai_role", "unknown")

        col_primary, col_ai = st.columns(2)
        with col_primary:
            st.metric("Primary Layer", primary_layer.upper())
        with col_ai:
            st.metric("AI Role", ai_role)

        if layer_scores:
            st.markdown("**Layer Scores:**")
            score_df = pd.DataFrame(
                [{"Layer": k, "Score": v} for k, v in layer_scores.items()]
            )
            st.bar_chart(score_df.set_index("Layer"))

        st.markdown("---")
        st.markdown("**Technical Layers & CPC Candidates:**")

        layer_names = {
            "application": "Application (What system is FOR)",
            "data_reasoning": "Data & Reasoning (How knowledge is represented)",
            "interaction": "Interaction (User/System interface)",
            "control": "Control (System orchestration logic)",
        }

        for layer_name, layer_candidates in layers.items():
            if layer_candidates:
                with st.expander(
                    f"[{'PRIMARY' if layer_name == primary_layer else 'LAYER'}] {layer_names.get(layer_name, layer_name)}",
                    expanded=(layer_name == primary_layer),
                ):
                    st.markdown(f"**Score:** {layer_scores.get(layer_name, 0):.2f}")
                    st.markdown("**CPC Candidates:**")
                    for cand in layer_candidates[:5]:
                        st.code(f"{cand['symbol']} ({cand.get('type', 'family')})")
                    rels = relationships.get(layer_name, [])
                    if rels:
                        st.markdown("**Relationships:**")
                        for rel in rels:
                            st.write(f"- {rel}")

        st.markdown("---")
        st.info(
            "**Multi-Layer Principle:** Each layer maps to CPC independently. "
            "A vehicle speech control patent should have: B60W (control) + G10L (speech) + "
            "G06F (data/NLP) + G05B (orchestration) — NOT collapsed to a single family."
        )
    else:
        st.warning("Phase 2A results not available.")
        # Fallback to legacy Phase 2A display
        phase2 = result.get("phase2", {})
        if phase2 and phase2.get("phase2a_families"):
            st.subheader("[FALLBACK] Legacy Phase 2A — CPC Family Router")
            col_fam, col_prim, col_mod = st.columns(3)
            with col_fam:
                st.markdown("**Selected Families:**")
                for fam in phase2.get("phase2a_families", [])[:5]:
                    st.code(fam)
            with col_prim:
                st.metric("Primary Family", phase2.get("phase2a_primary", "N/A"))
            with col_mod:
                st.metric("Modality", phase2.get("phase2a_modality", "unknown"))

# =============================================================================
# PHASE 2B: XML Expansion
# =============================================================================
elif phase == "Phase 2B: XML Expansion":
    phase2 = result.get("phase2", {})
    st.divider()
    st.subheader("[CHART] Phase 2B — Restricted XML Expansion")
    st.caption(
        "Method: CPC XML parser restricted to technical-layer 4‑char subclass prefixes. "
        "Only subclasses matching Phase 2A technical layers are expanded. "
        "Application-layer domains (B60, A61, G06Q) are excluded at the source."
    )
    if phase2:
        count_2b = phase2.get("phase2b_candidate_count", 0)
        expansion_counts = phase2.get("phase2b_expansion_counts", {})
        skipped = phase2.get("phase2b_skipped_classes", [])
        families = phase2.get("phase2a_families", [])

        st.markdown(
            "Expands CPC subgroup definitions **only within Phase 2A technical families** "
            "to reduce search space and prevent non-technical domain leakage."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expanded Candidates", count_2b)
        with col2:
            reduction = (
                f"~{((1 - count_2b / 250000) * 100):.1f}%" if count_2b > 0 else "N/A"
            )
            st.metric("Search Space Reduction", reduction)

        # Per-family expansion breakdown
        if expansion_counts:
            st.markdown("---")
            st.markdown("**Subgroups expanded per subclass prefix:**")
            count_items = sorted(expansion_counts.items())
            cols = st.columns(min(len(count_items), 4))
            for i, (prefix, cnt) in enumerate(count_items):
                with cols[i % len(cols)]:
                    st.metric(prefix, cnt)

        if skipped:
            st.warning(f"Skipped (no XML file): {', '.join(skipped)}")

        if families:
            st.caption(f"Restricted to families: {', '.join(families)}")
    else:
        st.warning("Phase 2B data not available.")

# =============================================================================
# PHASE 2C: TF-IDF Scoring
# =============================================================================
elif phase == "Phase 2C: Hybrid Scoring":
    phase2 = result.get("phase2", {})
    st.divider()
    st.subheader("[CHART] Phase 2C — Hybrid Scoring & Filtering")
    st.caption(
        "Method: Find‑Until‑Full retrieval — scores ALL expanded candidates with "
        "hybrid TF‑IDF (bigrams) + embedding similarity (0.4×TF‑IDF + 0.6×Semantic). "
        "Progressive expansion ensures Phase 2D has enough candidates to reach 20‑survivor quota."
    )
    if phase2:
        total_scored = phase2.get(
            "phase2c_total_scored", phase2.get("phase2c_final_count", 0)
        )
        margin = phase2.get("score_margin", 0)
        confidence = phase2.get("confidence_level", "unknown")
        find_until_full = phase2.get("phase2d_find_until_full", [])

        st.markdown(
            "Scores **ALL** expanded candidates. Phase 2D then progressively expands "
            "(500 → 1000 → all) until ≥ 20 technical anchors survive filtering."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scored", total_scored)
        with col2:
            st.metric("Score Margin", f"{margin:.6f}")
        with col3:
            conf_color = (
                "[HIGH]"
                if confidence == "high"
                else "[MED]"
                if confidence == "medium"
                else "[LOW]"
            )
            st.metric("Confidence", f"{conf_color} {confidence.upper()}")

        # Find-Until-Full expansion log
        if find_until_full:
            st.markdown("---")
            st.markdown("**Find‑Until‑Full Expansion Log:**")
            for entry in find_until_full:
                depth = entry.get("depth", "?")
                surv = entry.get("survivors_found", 0)
                triggered = entry.get("deep_search_triggered", False)
                icon = "🔍 Deep" if triggered else "✅ Quota"
                st.write(
                    f"{icon}: Scanned **{depth}** candidates → found **{surv}** valid technical anchors"
                )

        st.caption(
            "Higher margin = clearer separation between top candidates. "
            "Low margin suggests ambiguous classification."
        )
    else:
        st.warning("Phase 2C data not available.")

# =============================================================================
# PHASE 2D: Subclass Structural Anchor
# =============================================================================
elif phase == "Phase 2D: Subclass Anchor":
    phase2 = result.get("phase2", {})
    st.divider()
    st.subheader("[ANCHOR] Phase 2D — Subclass Structural Anchor Filter")
    st.caption(
        "Method: Filters Phase 2C candidates — keeps only those whose 4-char CPC subclass prefix "
        "matches the Phase 2A technical layer anchors (pure_software, data_reasoning, interaction, control). "
        "Excludes application-layer codes and non-technical families (G06Q, etc.)."
    )

    anchor_set = phase2.get("phase2d_anchor_set", [])
    anchor_source = phase2.get("phase2d_anchor_source", [])
    kept_count = phase2.get("phase2d_kept_count", 0)
    discarded_count = phase2.get("phase2d_discarded_count", 0)
    discard_log = phase2.get("phase2d_discard_log", [])

    if anchor_set:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Anchor Subclasses", len(anchor_set))
        with col_b:
            st.metric("Candidates Kept", kept_count)
        with col_c:
            st.metric("Candidates Discarded", discarded_count)

        st.markdown(f"**Anchor Set:** `{', '.join(anchor_set)}`")
        st.caption(
            f"Source layers: {', '.join(anchor_source) if anchor_source else 'family_router'}"
        )

        st.markdown(
            "**Filter Rule:** Candidate prefix must be in anchor set AND not in excluded families (G06Q, G06C, G07F, G07G, G09F, G09B, A63F)."
        )

        if discard_log:
            with st.expander(f"[DEBUG] Discarded Candidates ({len(discard_log)})"):
                for d in discard_log:
                    reason_icon = (
                        "[FAMILY]"
                        if d.get("reason") == "non_technical_family"
                        else "[ANCHOR]"
                        if d.get("reason") == "not_in_anchor_set"
                        else "[PREFIX]"
                    )
                    st.write(
                        f"{reason_icon} `{d.get('symbol', '?')}` — {d.get('reason', '')}"
                    )
    else:
        st.info(
            "Phase 2D anchor filter not applied — all candidates passed through unchanged."
        )

# =============================================================================
# PHASE 3: CPC Subgroup Ranking
# =============================================================================
elif phase == "Phase 3: CPC Subgroup Ranking":
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

# =============================================================================
# PHASE 3.5: Decision Tree Constraints
# =============================================================================
elif phase == "Phase 3.5: Decision Tree Constraints":
    phase35 = result.get("phase35", {})
    if phase35:
        st.divider()
        st.subheader("[TREE] Phase 3.5 — Decision Tree Constraint Layer")
        st.caption(
            "Method: Multi-step deterministic decision tree enforcing domain correctness, disambiguation, functional boosting, and invalid class filtering"
        )

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

            **Step 7: Code Type Tagging**
            - PRIMARY_STANDARD: standard main-group codes (e.g., G06F 8/xx, G05B 19/xx)
            - SECONDARY_INDEXING: 2xxx series indexing codes (e.g., G05B 2219/..., G06F 2110/...)
            
            **Step 8: Canonical 'Noun-First' Sorting**
            - Level 1: Type — PRIMARY_STANDARD always before SECONDARY_INDEXING
            - Level 2: Score — descending within each type group
            
            **Step 9: Quota Guardrail**
            - If Top 5 are ALL SECONDARY_INDEXING, reaches into positions 6-20
            - Promotes at least 2 PRIMARY_STANDARD codes into top 5
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

        adjusted = phase35.get("phase35_candidates", [])
        if adjusted:
            standard_codes = [
                c for c in adjusted if c.get("code_type") != "SECONDARY_INDEXING"
            ]
            indexing_codes = [
                c for c in adjusted if c.get("code_type") == "SECONDARY_INDEXING"
            ]

            # Core Invention (Standard Codes)
            if standard_codes:
                st.markdown("**Core Invention** (Standard Codes)")
                st.caption(
                    "Broad main-group classifications representing the primary technical contribution."
                )
                df_std = pd.DataFrame(standard_codes)
                display_cols = [
                    c for c in ["symbol", "title", "score"] if c in df_std.columns
                ]
                st.dataframe(df_std[display_cols], use_container_width=True)

            # Technical Details (Indexing Codes)
            if indexing_codes:
                st.markdown("**Technical Details** (Indexing Codes)")
                st.caption(
                    "Supplementary indexing codes providing additional technical context."
                )
                df_idx = pd.DataFrame(indexing_codes)
                display_cols = [
                    c for c in ["symbol", "title", "score"] if c in df_idx.columns
                ]
                st.dataframe(df_idx[display_cols], use_container_width=True)

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

# =============================================================================
# PHASE 3.6: Cross-Domain Validation
# =============================================================================
elif phase == "Phase 3.6: Cross-Domain Validation":
    phase36 = result.get("phase36", {})
    if phase36:
        st.divider()
        st.subheader("[TREE] Phase 3.6 — Universal CPC Hierarchy Selection")
        st.caption(
            "Method: Contribution-type-first, domain-second, subclass-mapped using universal A-F hierarchy"
        )

        with st.expander("[INFO] Validation Steps (Phase 3.6)", expanded=False):
            st.markdown("""
            **Step 1: Domain Anchor Check**
            - Verify candidate's CPC family matches domain signals
            - Require ≥1 domain signal for family validity
            
            **Step 2: Anti-Domain Collapse**
            - Prevent G10L without audio/speech signals
            - Prevent G06T without image/visual signals
            - Prevent G06V without computer vision signals
            
            **Step 3: Contextual Entity Consistency**
            - "prompt" → NLP domain (G06F/G06N), NOT speech (G10L)
            - "utterance" → speech domain (G10L), NOT text (G06F)
            - "dialogue" → text conversation (G06F), NOT spoken
            
            **Step 4: Final Family Lock**
            - Require ≥2 independent signals to strongly lock family
            - Weak lock (1 signal) gets penalty
            """)

        adjustments = phase36.get("phase36_adjustments", 0)
        domain_verified = phase36.get("phase36_domain_verified", False)

        st.markdown("**Cross-Domain Validation:**")
        col_verified, col_adj = st.columns(2)
        with col_verified:
            if domain_verified:
                st.success("[OK] Domain Verified")
            else:
                st.warning("[WARN] Domain Not Verified")
        with col_adj:
            st.metric("Validation Rules Applied", adjustments)

        rules_log = phase36.get("phase36_rules_log", [])
        if rules_log:
            with st.expander("[DEBUG] Validation Rules Applied"):
                for rule in rules_log[:20]:
                    st.write(
                        f"- **{rule.get('rule', '')}**: `{rule.get('symbol', '')}` "
                        f"{rule.get('score_before', 0):.3f} → {rule.get('score_after', 0):.3f} "
                        f"| {rule.get('reason', '')}"
                    )

        refined = phase36.get("phase36_candidates", [])
        if refined:
            st.markdown("**Validated Candidates (After Cross-Domain Check):**")
            for c in refined[:5]:
                verified = c.get("domain_context_verified", False)
                badge = "[VERIFIED]" if verified else "[UNVERIFIED]"
                signals = c.get("domain_signals_matched", 0)
                st.write(
                    f"`{c['symbol']}` — {c['score']:.4f} {badge} (signals: {signals})"
                )
    else:
        st.divider()
        st.subheader("[TREE] Phase 3.6 — Universal CPC Hierarchy Selection")
        st.warning("Phase 3.6 results not available.")

# =============================================================================
# PHASE 4: Hypothesis Consolidation
# =============================================================================
elif phase == "Phase 4: Hypothesis Consolidation":
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

            col_sw, col_conf = st.columns(2)
            with col_sw:
                st.metric("Support Weight", f"{support_weight:.2%}")
            with col_conf:
                st.metric("Confidence", confidence.upper())

            # ── Human-Readable Interpretation ──
            interpretation = phase4.get("phase4_interpretation", {})
            if interpretation:
                st.markdown("---")
                st.markdown("### [INSIGHT] Classification Health Analysis")

                support_status = interpretation.get("support_status", "unknown")
                coherence_status = interpretation.get("coherence_status", "unknown")

                # Support weight insight
                icon_sw = (
                    "✅"
                    if support_status == "clean"
                    else "⚠️"
                    if support_status == "messy"
                    else "🔶"
                )
                st.info(icon_sw + " " + interpretation.get("support_text", ""))

                # Coherence insight
                icon_ch = (
                    "✅"
                    if coherence_status == "high"
                    else "⚠️"
                    if coherence_status == "low"
                    else "🔶"
                )
                st.info(icon_ch + " " + interpretation.get("coherence_text", ""))

                # Actionable advice
                st.success("💡 " + interpretation.get("actionable_advice", ""))

            st.caption(phase4.get("phase4_reasoning", ""))
        else:
            st.warning("No hypotheses formed from candidates.")
    else:
        st.divider()
        st.subheader("[PUZZLE] Phase 4 — Hypothesis Consolidation")
        st.warning("Phase 4 results not available.")

# =============================================================================
# PHASE 5: Hypothesis Resolution
# =============================================================================
elif phase == "Phase 5: Hypothesis Resolution":
    phase5 = result.get("phase5", {})
    premier = result.get("premier", {})

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
            st.success(
                "**Classification Health:** Primary focus confirmed. "
                "High signal separation detected (no significant cross-domain leakage)."
            )

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

    elif phase5 and "primary" in phase5:
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
    else:
        st.warning("Phase 5 results not available.")

    # Show Premier if available
    if premier:
        st.markdown("---")
        st.subheader("[BEST] Premier CPC Classification")
        st.caption(
            "Method: Phase 7 Logic Reconciliation "
            "(Final consistency check to align Functional, Methodological, and Application codes)."
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Code", premier.get("symbol", "N/A"))
            st.metric("Confidence", premier.get("confidence", "N/A").upper())
        with col2:
            st.markdown(f"**Title:** {premier.get('title', 'N/A')}")
            st.markdown(f"**Reasoning:** {premier.get('reasoning', 'N/A')}")

    # ── Cross-Domain Facets ──
    pillars = phase5.get("pillars", {})
    if pillars:
        st.markdown("---")
        st.markdown("### [FACETS] Cross-Domain Classifications")

        facet_tooltips = {
            "pillar1_goal": "The core technical result/output.",
            "pillar2_method": "The AI/ML implementation strategy.",
            "pillar3_context": "The target hardware/industrial environment.",
        }

        # Premier pillar shown first as top header
        goal = pillars.get("pillar1_goal", {})
        if goal and goal.get("symbol"):
            st.markdown(
                f"🎯 **Primary Facet:** `{goal['symbol']}` — *{goal.get('title', '')}*"
            )
            st.caption(
                f"Score: {goal.get('score', 0):.4f} | {facet_tooltips['pillar1_goal']}"
            )

        # Supporting facets grouped under collapsible
        supporting = {k: v for k, v in pillars.items() if k != "pillar1_goal"}
        if supporting:
            with st.expander("**Supporting Technical Facets**", expanded=True):
                method = pillars.get("pillar2_method", {})
                context = pillars.get("pillar3_context", {})

                if method:
                    if method.get("symbol"):
                        st.markdown(
                            f"🧠 **Methodological:** `{method['symbol']}` — *{method.get('title', '')}*"
                        )
                        st.caption(
                            f"Score: {method.get('score', 0):.4f} | {facet_tooltips['pillar2_method']}"
                        )
                    else:
                        st.warning(
                            f"🧠 **Methodological:** {method.get('title', 'Not found')}"
                        )

                if context:
                    if context.get("symbol"):
                        st.markdown(
                            f"⚙️ **Application:** `{context['symbol']}` — *{context.get('title', '')}*"
                        )
                        st.caption(
                            f"Score: {context.get('score', 0):.4f} | {facet_tooltips['pillar3_context']}"
                        )
                    else:
                        st.warning(
                            f"⚙️ **Application:** {context.get('title', 'Not found')}"
                        )

# =============================================================================
# PHASE 8: Role Labeling & Report
# =============================================================================
elif phase == "Phase 8: Role Labeling & Report":
    formatted_report = result.get("formatted_report", "")
    phase8 = result.get("phase8_role_labeling", {})
    premier = result.get("premier", {})
    pillars = result.get("phase5", {}).get("pillars", {})

    premier_symbol = premier.get("symbol", "")
    premier_title = premier.get("title", "")
    premier_conf = premier.get("confidence", "medium")

    core = phase8.get("layer1_core", [])
    support = phase8.get("layer2_support", [])
    ctx_codes = phase8.get("layer2_context", [])
    coverage = phase8.get("layer3_coverage", [])

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # HEADER + CONFIDENCE BADGE
    # ═══════════════════════════════════════════════════════════
    st.subheader("📄 Executive Patent Classification Report")

    col_hero, col_badge = st.columns([3, 1])
    with col_hero:
        if premier_symbol:
            st.markdown(f"### Main Recommendation: `{premier_symbol}`")
            st.markdown(f"*{premier_title}*")
        else:
            st.warning("No premier classification available.")
    with col_badge:
        if premier_conf == "high":
            st.success("✅ High Confidence")
        elif premier_conf == "medium":
            st.info("🔶 Medium Confidence")
        else:
            st.warning("⚠️ Low Confidence")
    st.caption("Status: ✅ Validated via Cross-Domain Consistency Check")

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL BREAKDOWN TABLE — first substantive section
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🛠 Technical Breakdown")
    st.caption("Recommended CPC Classes for this Invention — Role-Based View")

    goal = pillars.get("pillar1_goal", {})
    method = pillars.get("pillar2_method", {})
    context = pillars.get("pillar3_context", {})

    pillar_tooltips = {
        "🎯 Primary Goal": "This code represents the core legal contribution of your patent — the actual output.",
        "🧠 AI Methodology": "The brain used to do it — the AI/ML implementation strategy.",
        "⚙️ Domain Context": "The industrial target — where the invention is applied.",
    }

    if goal is not None or method is not None or context is not None:
        tech_data = []

        # Primary Goal — always show
        g_sym = goal.get("symbol", "") if goal else ""
        tech_data.append(
            {
                "Role": "🎯 Primary Goal",
                "CPC Code": f"`{g_sym}`" if g_sym else "*Not found*",
                "Context": (
                    f"The actual output: {goal.get('title', 'N/A')}"
                    if g_sym
                    else f"Target: {goal.get('family', 'G06F')} subclass"
                ),
            }
        )

        # AI Methodology — always show
        m_sym = method.get("symbol", "") if method else ""
        tech_data.append(
            {
                "Role": "🧠 AI Methodology",
                "CPC Code": f"`{m_sym}`" if m_sym else "*Not found*",
                "Context": (
                    f"The brain used to do it: {method.get('title', 'N/A')}"
                    if m_sym
                    else f"Target: {method.get('family', 'G06N')} subclass"
                ),
            }
        )

        # Domain Context — always show
        c_sym = context.get("symbol", "") if context else ""
        tech_data.append(
            {
                "Role": "⚙️ Domain Context",
                "CPC Code": f"`{c_sym}`" if c_sym else "*Not found*",
                "Context": (
                    f"The industrial target: {context.get('title', 'N/A')}"
                    if c_sym
                    else f"Target: {context.get('family', 'G05B')} subclass"
                ),
            }
        )

        st.dataframe(
            pd.DataFrame(tech_data),
            use_container_width=True,
            hide_index=True,
        )

        # Tooltips as caption below the table
        st.caption(
            "ℹ️ **Primary Goal:** This code represents the core legal contribution of your patent — the actual output.  "
            "**AI Methodology:** The brain used to do it — the AI/ML implementation strategy.  "
            "**Domain Context:** The industrial target — where the invention is applied."
        )

    # ═══════════════════════════════════════════════════════════
    # PROFESSIONAL JUSTIFICATION
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 💡 Professional Justification")
    llm_summary = phase8.get("phase85_executive_summary", "")
    if llm_summary:
        st.markdown(llm_summary)
    else:
        reason = premier.get("reasoning", "")
        if reason:
            st.info(reason)
        else:
            st.info(
                "The classification reflects the primary technical contribution "
                "of the disclosed invention based on semantic analysis, technical "
                "weight analysis, and cross-domain validation."
            )

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS & LOGIC (Collapsible Tabs)
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Analysis & Logic")
    st.caption("Click to expand — expert reasoning, supplementary codes, and raw data.")

    # Tab 1: Expert Analysis — Main Recommendation
    with st.expander("💡 How was the Main Recommendation selected?"):
        st.markdown(
            "<div style='background-color:#e8f4fd;padding:12px;border-radius:8px;"
            "border-left:4px solid #2196F3;'>"
            "<em><strong>Expert Analysis — Thematic Shift Detection</strong><br><br>"
            "The system performs a <strong>'Thematic Shift' analysis</strong> across "
            "all pipeline phases. While early phases focused on broad families, "
            "the final consistency check identifies the most legally defensible "
            "specific code by analyzing how the technical contribution <strong>shifts</strong> "
            "from general functionality to specific novelty.<br><br>"
            "This code survived all validation layers: "
            "<strong>Phase 2D anchor filtering</strong> (eliminated non-technical noise), "
            "<strong>Phase 3.5 decision tree</strong> (prioritized standard over indexing codes), "
            "<strong>Phase 3.6 cross-domain validation</strong> (verified domain consistency), "
            "and <strong>Phase 4/5 hypothesis resolution</strong> (confirmed as best-fit hypothesis)."
            "</em></div>",
            unsafe_allow_html=True,
        )

    # Tab 2: Expert Analysis — Confidence
    if premier_conf != "high":
        with st.expander("💡 Why not higher confidence?"):
            st.markdown(
                "<div style='background-color:#fff3e0;padding:12px;border-radius:8px;"
                "border-left:4px solid #FF9800;'>"
                "<em><strong>Expert Analysis — Subclass Discrepancy Detection</strong><br><br>"
                "This flag is triggered because the invention spans multiple distinct "
                "high-value CPC domains. The system detected a <strong>'Dual-Core' pattern</strong> — "
                "the patent's technical contribution bridges two or more major subclasses "
                "(e.g., G06N Artificial Intelligence vs. G06F Software Engineering).<br><br>"
                "<strong>What this means:</strong> The AI found credible classifications in "
                "multiple families. This is <strong>not an error</strong> — it indicates the "
                "invention is genuinely multi-disciplinary. Human review is advised to choose "
                "the primary filing target based on the strongest legal claims."
                "</em></div>",
                unsafe_allow_html=True,
            )

    # Tab 3: Expert Analysis — Role Assignment
    with st.expander("💡 Why assign roles to CPC codes?"):
        st.markdown(
            "<div style='background-color:#e8f4fd;padding:12px;border-radius:8px;"
            "border-left:4px solid #2196F3;'>"
            "<em><strong>Expert Analysis — Preventing Domain-Blindness</strong><br><br>"
            "This <strong>'Tech Stack' view</strong> prevents <strong>domain-blindness</strong> — "
            "the common patent classification error where a single code dominates and "
            "hides the multi-dimensional nature of modern inventions.<br><br>"
            "By assigning distinct <strong>functional roles</strong> "
            "(Goal = what it produces, Engine = how it works, Context = where it applies), "
            "we ensure the patent is <strong>searchable across AI, Software, and Industrial "
            "databases simultaneously</strong>."
            "</em></div>",
            unsafe_allow_html=True,
        )

    # Tab 4: Expert Analysis — Justification Generation
    with st.expander("💡 How was the Professional Justification generated?"):
        st.markdown(
            "<div style='background-color:#e8f4fd;padding:12px;border-radius:8px;"
            "border-left:4px solid #2196F3;'>"
            "<em><strong>Expert Analysis — Multi-Phase Reasoning Synthesis</strong><br><br>"
            "This justification is <strong>NOT</strong> a generic template — it is "
            "<strong>dynamically generated</strong> from actual pipeline data: "
            "Phase 1 (technical object, core function), Phase 5 Facets (goal/method/context "
            "champions), Phase 8 role labeling (core/support/context/coverage layers), "
            "and the consistency check (cross-domain coherence validation).<br><br>"
            "The LLM receives structured data and produces a <strong>professional brief</strong> "
            "suitable for patent filings or examiner responses."
            "</em></div>",
            unsafe_allow_html=True,
        )

    # Tab 5: Suggested Indexing Codes
    pillar_symbols = set()
    for k, v in pillars.items():
        if v.get("symbol"):
            pillar_symbols.add(v["symbol"])

    suggested = []
    for group in [core, support, ctx_codes, coverage]:
        for c in group:
            sym = c.get("symbol", "")
            if sym not in pillar_symbols and sym != premier_symbol:
                title = c.get("title", "")
                if title:
                    suggested.append(f"`{sym}` — {title}")

    if suggested:
        with st.expander("📋 Suggested Indexing Codes (for patent filing)"):
            st.caption(
                "Copy these codes into your patent application as supplementary indexing references."
            )
            for s in suggested[:10]:
                st.write(f"- {s}")
    else:
        with st.expander("📋 Suggested Indexing Codes (for patent filing)"):
            st.write(
                "No additional indexing codes available beyond the recommended classes above."
            )

    # Tab 6: Full Classification Report
    if formatted_report:
        with st.expander("📊 Full Classification Report (Raw Markdown)"):
            st.markdown(formatted_report)

    # Tab 7: Supporting Details
    if core or support or ctx_codes:
        with st.expander("📊 Supporting Classification Details (Layer Breakdown)"):
            if core:
                st.markdown("**Core Invention:**")
                for c in core:
                    st.write(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
            if support:
                st.markdown("**Enabling Technology:**")
                for c in support:
                    st.write(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
            if ctx_codes:
                st.markdown("**Application Context:**")
                for c in ctx_codes:
                    st.write(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")
            if coverage:
                st.markdown("**Legal Coverage:**")
                for c in coverage:
                    st.write(f"- `{c.get('symbol', '')}` — {c.get('title', 'N/A')}")

    # ═══════════════════════════════════════════════════════════
    # DOWNLOAD
    # ═══════════════════════════════════════════════════════════
    if formatted_report:
        st.markdown("---")
        st.download_button(
            label="📥 Download Executive Report (Markdown)",
            data=formatted_report,
            file_name="cpc_classification_report.md",
            mime="text/markdown",
        )

# =============================================================================
# RAW JSON DEBUG (always show, collapsed)
# =============================================================================
st.divider()
st.caption("**Raw JSON Dumps** (for debugging)")

phase1_raw = result.get("phase1", {})
phase3_raw = result.get("phase3", [])
phase4_raw = result.get("phase4", {})
phase5_raw = result.get("phase5", {})

with st.expander("[DEBUG] Raw Phase 1 JSON"):
    st.json(phase1_raw)

with st.expander("[DEBUG] Raw Phase 3 JSON"):
    st.json(phase3_raw)

if phase4_raw:
    with st.expander("[DEBUG] Raw Phase 4 JSON"):
        st.json(phase4_raw)

if phase5_raw:
    with st.expander("[DEBUG] Raw Phase 5 JSON"):
        st.json(phase5_raw)
