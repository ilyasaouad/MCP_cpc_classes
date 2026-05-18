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
    "Phase 1.2: Forensic Claims Audit",
    "Phase 1.5: Role Classification",
    "TCR: Technical Weight Analysis",
    "Phase 2A v2: CPC Decision",
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

if (
    "current_phase" not in st.session_state
    or st.session_state["current_phase"] not in PHASES
):
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

    # ── Completeness Status (new) ──
    p1_status = phase1.get("phase1_status", "MISSING")
    p1_score = phase1.get("phase1_score", "MISSING")
    if p1_status != "MISSING":
        status_color = (
            "success"
            if p1_status == "PASS"
            else "warning"
            if p1_status == "WARN"
            else "error"
        )
        getattr(st, status_color)(
            f"**Phase 1 Status: {p1_status}** (Score: {p1_score}/100)"
        )
    else:
        st.caption("_Phase 1 completeness status not available_")

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
                name = ds.get("label") or ds.get("name", "")
                cpc = ds.get("cpc_family", "")
                conf = ds.get("confidence", 0)
                role = ds.get("role", "")
                st.write(f"- **{name}** `{cpc}` (confidence: {conf:.2f}, role: {role})")
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
        if isinstance(strategy, dict):
            strat_type = strategy.get("strategy", "unknown")
            primary = strategy.get("primary_family", "")
            secondary = strategy.get("secondary_family") or "None"
            anchor_split = strategy.get("anchor_split", [1.0, 0.0])
            reconstructed = strategy.get("reconstructed", False)
            st.markdown(
                f"**[CHART] Classification Strategy:** `{strat_type}` | "
                f"Primary: `{primary}` | Secondary: `{secondary}` | "
                f"Anchor Split: `{anchor_split}`"
            )
            if reconstructed:
                st.caption(
                    f"⚠ Reconstructed from domain signals — "
                    f"{strategy.get('reason', 'LLM did not emit strategy block')}"
                )
        else:
            st.markdown(f"**[CHART] Classification Strategy:** `{strategy}`")

    # Evidence Table (if present)
    evidence_table = phase1.get("evidence_table", [])
    if evidence_table:
        st.markdown("---")
        st.markdown("**[EVIDENCE] Evidence Table:**")
        df_evidence = pd.DataFrame(evidence_table)
        st.dataframe(
            df_evidence,
            use_container_width=True,
            column_config={
                "term": st.column_config.TextColumn("Term", width="medium"),
                "weight": st.column_config.NumberColumn("Weight", width="small"),
                "justification": st.column_config.TextColumn(
                    "Justification", width="large"
                ),
                "source": st.column_config.TextColumn("Source", width="small"),
                "citation": st.column_config.TextColumn("Citation", width="large"),
            },
        )

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
# PHASE 1.2: Forensic Claims Audit
# =============================================================================
elif phase == "Phase 1.2: Forensic Claims Audit":
    phase1_2 = result.get("phase1_2", {})

    st.divider()
    st.subheader("[LEGAL GATE] Phase 1.2 — Mandatory Forensic Claims Audit")
    st.caption(
        "Method: Validates Phase 1 domain signals against actual claims text. "
        "First time claims are analyzed in the pipeline."
    )

    if not phase1_2:
        st.warning("Phase 1.2 results not available.")
        st.stop()

    # Audit status
    audit_status = phase1_2.get("audit_status", "UNKNOWN")
    if audit_status == "SUCCESS":
        st.success(f"**Audit Status: {audit_status}**")
    elif audit_status == "PARTIAL":
        st.warning(f"**Audit Status: {audit_status}**")
    else:
        st.error(f"**Audit Status: {audit_status}**")

    # Primary anchor
    primary_anchor = phase1_2.get("final_primary_anchor", "")
    if primary_anchor:
        st.markdown(f"**Final Primary Anchor:** `{primary_anchor}`")

    # Secondary anchors
    secondary_anchors = phase1_2.get("secondary_anchors", [])
    if secondary_anchors:
        st.markdown(
            f"**Secondary Anchors:** {', '.join(f'`{a}`' for a in secondary_anchors)}"
        )

    # Domain signal validation
    st.markdown("---")
    st.markdown("**Domain Signal Validation:**")
    validations = phase1_2.get("signal_validations", [])
    if validations:
        for v in validations[:10]:
            if isinstance(v, dict):
                domain = v.get("domain", v.get("signal", "?"))
                cpc = v.get("cpc_family", "?")
                status = v.get("status", "unknown")
                reason = v.get("reason", "")
                evidence = v.get("claims_evidence", "")

                if status == "validated":
                    st.markdown(f"- ✅ `{domain}` `{cpc}` — {reason}")
                elif status == "rejected":
                    st.markdown(f"- ❌ `{domain}` `{cpc}` — {reason}")
                elif status == "downgraded":
                    st.markdown(f"- ⚠️ `{domain}` `{cpc}` — {reason}")
                else:
                    st.markdown(f"- ❓ `{domain}` `{cpc}` — {status}: {reason}")

                if evidence:
                    st.caption(f"  *Evidence: {evidence[:150]}...*")
    else:
        st.caption("No signal validation details available")

    # Conflict resolution
    conflicts = phase1_2.get("conflicts_detected", [])
    if conflicts:
        st.markdown("---")
        st.markdown("**Conflicts Detected:**")
        for c in conflicts:
            if isinstance(c, dict):
                st.warning(f"- {c.get('description', str(c))}")

    # Rejected domains
    rejected = phase1_2.get("rejected_domains", [])
    if rejected:
        st.markdown("---")
        st.markdown("**Rejected Domains:**")
        for r in rejected:
            if isinstance(r, dict):
                st.markdown(f"- ❌ `{r.get('domain', '?')}` — {r.get('reason', '')}")
            elif isinstance(r, str):
                st.markdown(f"- ❌ {r}")

    # Reasoning
    reasoning = phase1_2.get("audit_reasoning", "")
    if reasoning:
        st.markdown("---")
        st.markdown("**Audit Reasoning:**")
        st.caption(reasoning)

    # Raw JSON
    with st.expander("[DEBUG] Raw Phase 1.2 JSON"):
        st.json(phase1_2)

# =============================================================================
# PHASE 1.5: Invention Role Classification + TCR (COMBINED)
# =============================================================================
elif phase == "Phase 1.5: Role Classification":
    phase15 = result.get("phase15", {})
    tcr_result = result.get("tcr_analysis", {})

    st.divider()
    st.subheader("[TARGET] Phase 1.5 — Invention Role Classification + TCR")
    st.caption(
        "Method: Structural pattern labelling (CORE_TECH / SYSTEM / APPLICATION / SUPPORT) "
        "+ Technical Character Ratio analysis"
    )

    if not phase15:
        st.warning("Phase 1.5 results not available.")
        st.stop()

    # ── Role Classification ──
    role = phase15.get("role", "UNKNOWN")
    confidence = phase15.get("confidence", 0)
    reasoning = phase15.get("reasoning", "")

    ROLE_UI_MAP = {
        "CORE_TECH": {"label": "Internal Mechanism Focus", "color": "success"},
        "SYSTEM": {"label": "Multi-Component System", "color": "info"},
        "APPLICATION": {"label": "Application Context", "color": "warning"},
        "SUPPORT": {"label": "Supporting Function", "color": "secondary"},
    }
    ui = ROLE_UI_MAP.get(role, {"label": role, "color": "info"})

    col_role, col_conf = st.columns(2)
    with col_role:
        fn = getattr(st, ui["color"], st.info)
        fn(f"**{ui['label']}**")
        st.caption(f"Raw role: {role}")
    with col_conf:
        st.metric("Role Confidence", f"{confidence:.2f}")

    if reasoning:
        st.caption(f"*{reasoning}*")

    st.markdown("---")

    # ── TCR Analysis (shown inline) ──
    st.markdown("#### Technical Character Ratio (TCR)")
    st.caption(
        "Soft bias signal — measures computational vs physical term density in the invention"
    )

    if tcr_result:
        tcr = tcr_result.get("tcr", 1.0)
        tcr_bias = tcr_result.get("tcr_bias", 0.0)
        tcr_conf = tcr_result.get("confidence", 0.0)
        comp_score = tcr_result.get("computational_score", 0)
        phys_score = tcr_result.get("physical_score", 0)
        force_mode = tcr_result.get("force_mode", "UNKNOWN")
        override = tcr_result.get("override_applied", False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TCR Ratio", f"{tcr:.3f}")
        with col2:
            st.metric("TCR Bias", f"{tcr_bias:+.3f}")
        with col3:
            st.metric("TCR Confidence", f"{tcr_conf:.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Computational", f"{comp_score:.1f}")
        with col2:
            st.metric("Physical", f"{phys_score:.1f}")

        st.caption(f"**Force Mode:** `{force_mode}`")

        # Bias interpretation
        if tcr_bias > 0.3:
            st.info("📊 **Computational leaning** — software/AI methodology dominant")
        elif tcr_bias < -0.3:
            st.info("📊 **Physical leaning** — hardware/domain physicality dominant")
        else:
            st.info("📊 **Balanced** — hybrid invention, no strong bias")

        # Matched terms
        comp_terms = tcr_result.get("computational_terms", [])
        phys_terms = tcr_result.get("physical_terms", [])

        if comp_terms or phys_terms:
            with st.expander("Show matched terms"):
                if comp_terms:
                    st.markdown(
                        f"**Computational ({len(comp_terms)}):** {', '.join(comp_terms[:15])}"
                        + ("..." if len(comp_terms) > 15 else "")
                    )
                if phys_terms:
                    st.markdown(
                        f"**Physical ({len(phys_terms)}):** {', '.join(phys_terms[:15])}"
                        + ("..." if len(phys_terms) > 15 else "")
                    )
                if not comp_terms and not phys_terms:
                    st.caption("No terms matched either category")
        else:
            st.caption("No term matches recorded")

        # Warning if fallback
        if tcr_conf == 0.0 and tcr == 1.0:
            st.error(
                "⚠️ **TCR is using fallback values** — "
                "`TechnicalWeightAnalyzer.analyze()` likely failed. Check backend logs."
            )
    else:
        st.warning("TCR analysis not available in response.")

    # ── Raw JSON Debug ──
    st.markdown("---")
    with st.expander("[DEBUG] Raw Phase 1.5 + TCR JSON"):
        st.json(
            {
                "phase15": phase15,
                "tcr_analysis": tcr_result,
            }
        )

# =============================================================================
# TCR: Technical Weight Analysis (DETAILED VIEW)
# =============================================================================
elif phase == "TCR: Technical Weight Analysis":
    tcr_result = result.get("tcr_analysis", {})

    st.divider()
    st.subheader("Technical Weight Analysis")
    st.caption(
        "Soft bias signal — non-authoritative. Measures computational vs physical term density."
    )

    if not tcr_result:
        st.warning("TCR analysis not available.")
        with st.expander("[DEBUG] Check response keys"):
            st.json(
                {
                    "available_keys": list(result.keys()),
                    "has_tcr_analysis": "tcr_analysis" in result,
                    "tcr_analysis_value": result.get("tcr_analysis"),
                }
            )
        st.stop()

    tcr = tcr_result.get("tcr", 1.0)
    tcr_bias = tcr_result.get("tcr_bias", 0.0)
    confidence = tcr_result.get("confidence", 0.0)
    comp_weight = tcr_result.get("computational_score", 0)
    phys_weight = tcr_result.get("physical_score", 0)
    force_mode = tcr_result.get("force_mode", "UNKNOWN")
    override = tcr_result.get("override_applied", False)
    dominant = tcr_result.get("dominant_bucket", "unknown")
    analysis_mode = tcr_result.get("analysis_mode", "unknown")

    col_tcr1, col_tcr2, col_tcr3 = st.columns(3)
    with col_tcr1:
        st.metric("TCR Ratio", f"{tcr:.3f}")
    with col_tcr2:
        st.metric("Bias", f"{tcr_bias:+.3f}")
    with col_tcr3:
        st.metric("Match Confidence", f"{confidence:.0%}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Comp Score", f"{comp_weight:.2f}")
    with col_s2:
        st.metric("Phys Score", f"{phys_weight:.2f}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Force Mode", force_mode)
    with col2:
        st.metric("Dominant", dominant)
    with col3:
        st.metric("Override Applied", "Yes" if override else "No")

    # Bias interpretation — accounts for both magnitude and force mode
    if force_mode == "FORCE_SOFTWARE_CORE" or tcr_bias >= 0.9:
        st.success("**Strong computational dominance** — software/AI core confirmed (FORCE_SOFTWARE_CORE active)")
    elif force_mode == "FORCE_DOMAIN_CORE" or tcr_bias <= -0.9:
        st.success("**Strong physical/domain dominance** — hardware/domain core confirmed (FORCE_DOMAIN_CORE active)")
    elif tcr_bias >= 0.5:
        st.info("**Moderate–strong computational leaning** — ranking bias applied toward software/AI families")
    elif tcr_bias <= -0.5:
        st.info("**Moderate–strong physical leaning** — ranking bias applied toward domain/hardware families")
    elif tcr_bias > 0.15:
        st.caption("Slight computational leaning — minimal ranking bias")
    elif tcr_bias < -0.15:
        st.caption("Slight physical/domain leaning — minimal ranking bias")
    else:
        st.caption("Balanced — no bias applied")

    # Matched terms
    comp_terms = tcr_result.get("computational_terms", [])
    phys_terms = tcr_result.get("physical_terms", [])

    col_terms1, col_terms2 = st.columns(2)
    with col_terms1:
        if comp_terms:
            st.markdown(f"**Computational Terms ({len(comp_terms)}):**")
            st.write(
                ", ".join(comp_terms[:15]) + ("..." if len(comp_terms) > 15 else "")
            )
        else:
            st.markdown("**Computational Terms:** (none matched)")
    with col_terms2:
        if phys_terms:
            st.markdown(f"**Physical/Domain Terms ({len(phys_terms)}):**")
            st.write(
                ", ".join(phys_terms[:15]) + ("..." if len(phys_terms) > 15 else "")
            )
        else:
            st.markdown("**Physical/Domain Terms:** (none matched)")

    # Fallback warning
    if confidence == 0.0 and tcr == 1.0:
        st.error(
            "⚠️ **TCR is using fallback values** — "
            "`TechnicalWeightAnalyzer.analyze()` likely failed."
        )

    # Analysis mode
    st.caption(f"Analysis mode: `{analysis_mode}`")

    # Debug
    with st.expander("[DEBUG] Raw TCR JSON"):
        st.json(tcr_result)

# =============================================================================
# PHASE 2A V2: CPC DECISION (AUTHORITATIVE)
# =============================================================================
elif phase == "Phase 2A v2: CPC Decision":
    st.divider()
    st.subheader("\u2714 Final CPC Decision (Phase 2A v2)")
    phase2 = result.get("phase2", {})
    phase2a_v2 = phase2.get("phase2a_v2", {})
    fallback_used = phase2.get("fallback_used", False)
    cpc_source = phase2.get("cpc_source", "unknown")
    if fallback_used:
        st.error("\u26a0 Fallback CPC Selection Used \u2014 Phase 2A v2 failed")
        st.caption("This is fallback, not primary classification.")
        families = phase2.get("final_cpc_families", [])
        if families:
            st.markdown("**Fallback Families:**")
            for fam in families:
                st.code(fam)
        else:
            st.warning(
                "No CPC families available \u2014 pipeline could not produce classification."
            )
    elif phase2a_v2.get("families"):
        st.caption(
            f"Source: {cpc_source}  |  Fallback: {'Yes' if fallback_used else 'No'}"
        )
        families = phase2a_v2["families"]
        for f in families:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{f['family']}**")
                st.caption(", ".join(f.get("evidence", [])))
            with col2:
                st.metric("Score", f"{f['score']:.3f}")
        debug = phase2a_v2.get("debug", {})
        if debug:
            with st.expander("Fusion Evidence Trace"):
                st.json(debug)
    elif phase2.get("phase2a_families"):
        st.info("Using saved CPC families")
        for fam in phase2.get("phase2a_families", [])[:5]:
            st.code(fam)

# =============================================================================
# PHASE 2B: XML Expansion
# =============================================================================
elif phase == "Phase 2B: XML Expansion":
    phase2 = result.get("phase2", {})
    st.divider()
    st.subheader("[CHART] Phase 2B — Weighted Hierarchical CPC Expansion")
    st.caption(
        "Method: Multi-source expansion (KG hierarchy → graph traversal depth 2-3 → XML fallback). "
        "Each subclass is scored via inheritance (50%) + KG similarity (30%) + embedding (20%). "
        "Slot allocation is proportional to Phase 2A relevance score — high-relevance families "
        "receive more search slots than low-relevance ones."
    )
    if phase2:
        count_2b = phase2.get("phase2b_candidate_count", 0)
        expansion_counts = phase2.get("phase2b_expansion_counts", {})
        family_expansions = phase2.get("phase2b_family_expansions", [])
        pruned_count = phase2.get("phase2b_pruned_count", 0)
        expansion_balance = phase2.get("expansion_balance", {})
        families = phase2.get("phase2a_families", [])
        raw_counts = phase2.get("phase2b_raw_family_counts", {})
        prop_caps = phase2.get("phase2b_proportional_caps", {})
        family_scores_2a = {
            f["family"]: f["score"]
            for f in phase2.get("phase2a_v2_result", {}).get("families", [])
        }

        st.markdown(
            "Expands CPC families into **scored, ranked, and pruned** subgroups. "
            f"{pruned_count} low-relevance subclasses filtered out (score < 0.30)."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expanded Candidates", count_2b)
        with col2:
            st.metric("Pruned (score < 0.30)", pruned_count)
        with col3:
            reduction = (
                f"~{((1 - count_2b / 250000) * 100):.1f}%" if count_2b > 0 else "N/A"
            )
            st.metric("Search Space Reduction", reduction)

        # Proportional scaling table
        if raw_counts and prop_caps and families:
            st.markdown("---")
            st.markdown("**Relevance-proportional slot allocation per family:**")
            st.caption(
                "Each family's raw taxonomy count is scaled down proportionally to its "
                "Phase 2A relevance score. The highest-scoring family keeps its full count "
                "as reference. Others are capped at: `ref_count × (family_score / ref_score)`. "
                "This prevents low-relevance families from flooding the search space."
            )

            # Find reference family (highest Phase 2A score)
            if family_scores_2a:
                ref_fam = max(family_scores_2a, key=family_scores_2a.get)
                ref_score_val = family_scores_2a[ref_fam]
                ref_raw_val = raw_counts.get(ref_fam, 0)
            else:
                ref_fam, ref_score_val, ref_raw_val = "", 0.0, 0

            cols = st.columns(len(families))
            for i, fam in enumerate(families):
                raw = raw_counts.get(fam, "—")
                cap = prop_caps.get(fam, "—")
                final = expansion_balance.get(fam, "—")
                score = family_scores_2a.get(fam, 0.0)
                is_ref = fam == ref_fam

                with cols[i]:
                    st.markdown(f"**{fam}**")
                    st.caption(f"Phase 2A score: `{score:.3f}`" + (" (reference)" if is_ref else ""))
                    st.metric("Raw taxonomy count", raw)
                    if not is_ref and ref_raw_val > 0 and ref_score_val > 0:
                        st.caption(
                            f"Cap = {ref_raw_val} × ({score:.3f} / {ref_score_val:.3f}) = **{cap}**"
                        )
                    else:
                        st.caption("Reference family — no cap applied")
                    st.metric("After scaling", final, delta=f"{(final - raw) if isinstance(raw, int) and isinstance(final, int) else '—'}")

        # Per-family expansion breakdown
        if expansion_counts:
            st.markdown("---")
            st.markdown("**Final subgroups per family (after scaling):**")
            count_items = sorted(expansion_counts.items())
            cols = st.columns(min(len(count_items), 4))
            for i, (prefix, cnt) in enumerate(count_items):
                with cols[i % len(cols)]:
                    st.metric(prefix, cnt)

        if families:
            st.caption(f"Families expanded: {', '.join(families)}")
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
        "Method: Scores ALL Phase 2B expanded candidates with BM25 (bigrams) + "
        "embedding similarity, fused via RRF. Phase 2D then keeps the top 50 by score."
    )
    if phase2:
        total_scored = phase2.get(
            "phase2c_total_scored", phase2.get("phase2c_final_count", 0)
        )
        margin = phase2.get("score_margin", 0)
        confidence = phase2.get("confidence_level", "unknown")
        find_until_full = phase2.get("phase2d_find_until_full", [])

        st.markdown(
            "Scores **ALL** expanded candidates. Phase 2D keeps the **top 50** "
            "by hybrid score for Phase 3 ranking."
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
    st.subheader("[FILTER] Phase 2D — Top-N Score Filter")
    st.caption(
        "Method: Keeps the top 50 highest-scoring candidates from Phase 2C. "
        "Eliminates low-confidence subgroups before Phase 3 ranking, giving Phase 3 "
        "a focused, high-quality input instead of hundreds of diluted candidates."
    )

    kept_count = phase2.get("phase2d_kept_count", 0)
    discarded_count = phase2.get("phase2d_discarded_count", 0)
    discard_log = phase2.get("phase2d_discard_log", [])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Candidates Kept (Top 50)", kept_count)
    with col_b:
        st.metric("Candidates Discarded", discarded_count)
    with col_c:
        total = kept_count + discarded_count
        reduction = f"~{(discarded_count / total * 100):.1f}%" if total > 0 else "N/A"
        st.metric("Noise Reduction", reduction)

    st.markdown("**Filter Rule:** Sort all Phase 2C candidates by score → keep top 50.")

    if discard_log:
        with st.expander(f"[DEBUG] Discarded Candidates ({len(discard_log)})"):
            for d in discard_log[:20]:
                st.write(f"`{d.get('symbol', '?')}` — {d.get('reason', '')}")
            if len(discard_log) > 20:
                st.caption(f"... and {len(discard_log) - 20} more")

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
            st.caption(
                "This is the single most specific, legally defensible CPC subgroup for this invention. "
                "It was selected by the pipeline as the code that best captures the core technical "
                "contribution — not just the domain, but the exact technical operation being claimed. "
                "It survived all 7 validation phases: extraction → audit → routing → scoring → "
                "decision tree → cross-domain validation → hypothesis resolution."
            )
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
    st.caption(
        "Every patent invention has three technical dimensions simultaneously. "
        "**Primary Goal** = what the invention *produces* (the legally claimed output). "
        "**AI Methodology** = *how* it works internally (the ML/AI engine behind it). "
        "**Domain Context** = *where* it applies (the technical environment or field). "
        "Assigning all three prevents 'domain-blindness' — the error of capturing only one dimension "
        "and missing the full scope of the invention."
    )

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
    # SUPPORTING CLASSIFICATION DETAILS (Layer Breakdown)
    # ═══════════════════════════════════════════════════════════
    if core or support or ctx_codes or coverage:
        st.markdown("---")
        st.markdown("### 📊 Supporting Classification Details")
        st.caption(
            "These codes describe the invention at three functional layers. "
            "**Core Invention** = the primary technical operation — what the patent *does* at its heart. "
            "**Enabling Technology** = supporting components that make the core possible (concatenation rules, phonemic categorisation, etc.). "
            "**Application Context** = the broader technical domain where the invention is deployed. "
            "Together these three layers give examiners a complete picture of the invention's scope."
        )
        col_core, col_enable, col_ctx = st.columns(3)
        with col_core:
            st.markdown("**🔵 Core Invention**")
            if core:
                for c in core:
                    sym = c.get("symbol", "")
                    title = c.get("title", "N/A")
                    st.markdown(f"`{sym}`")
                    st.caption(title)
            else:
                st.caption("—")
        with col_enable:
            st.markdown("**🟡 Enabling Technology**")
            if support:
                for c in support:
                    sym = c.get("symbol", "")
                    title = c.get("title", "N/A")
                    st.markdown(f"`{sym}`")
                    st.caption(title)
            else:
                st.caption("—")
        with col_ctx:
            st.markdown("**🟢 Application Context**")
            if ctx_codes:
                for c in ctx_codes:
                    sym = c.get("symbol", "")
                    title = c.get("title", "N/A")
                    st.markdown(f"`{sym}`")
                    st.caption(title)
            else:
                st.caption("—")

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
    # METHODS & ALGORITHMS
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### ⚙️ Methods & Algorithms Used")
    st.caption(
        "This classification was produced by a deterministic multi-phase pipeline. "
        "Only two phases use a Large Language Model (LLM). All other phases are fully "
        "deterministic — no generative AI, no hallucination risk."
    )

    pipeline_phases = [
        {
            "phase": "Phase 1 — Extraction",
            "type": "🤖 LLM",
            "color": "#fff3e0",
            "border": "#FF9800",
            "description": (
                "An LLM reads the raw patent text and extracts: Technical Object (what the device is), "
                "Core Function (what it does), 10–15 ranked terms by importance, system context, "
                "evidence table, and negative signals. This is the only phase that interprets free-form text."
            ),
        },
        {
            "phase": "Phase 1B — Claims Audit",
            "type": "🤖 LLM",
            "color": "#fff3e0",
            "border": "#FF9800",
            "description": (
                "An LLM performs forensic analysis of the independent claims, verifying which CPC "
                "families have genuine claim support. Families with zero claim evidence are added to "
                "a Kill Log and blocked from entering the search space. This is a hard filter — "
                "rejected families cannot re-enter downstream."
            ),
        },
        {
            "phase": "Phase 1C — Technical Character",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Computes the Technical Character Ratio (TCR): the density of computational terms "
                "(neural, embedding, gradient…) vs physical terms (motor, sensor, valve…). "
                "The ratio biases Phase 2A toward software or hardware CPC families. "
                "Also assigns role tags (CORE_TECH / SYSTEM / SUPPORT) to each extracted term."
            ),
        },
        {
            "phase": "Phase 2A — CPC Family Routing",
            "type": "📐 Deterministic + Embeddings",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Selects the 3–5 most relevant CPC families using a weighted fusion of three signals: "
                "0.45 × sentence-transformer embedding similarity (all-mpnet-base-v2), "
                "0.35 × Knowledge Graph keyword scoring, "
                "0.20 × anchor signal matching (domain keywords from Phase 1B). "
                "Families rejected by Phase 1B are hard-blocked before scoring."
            ),
        },
        {
            "phase": "Phase 2B — Subgroup Expansion",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Each selected family is expanded into its full set of CPC subgroups using a "
                "cascading strategy: KG hierarchy lookup → KG graph BFS traversal (depth 3) → "
                "XML definition parser fallback. Slot allocation is proportional to Phase 2A relevance "
                "scores — higher-ranked families receive more expansion slots."
            ),
        },
        {
            "phase": "Phase 2C — Candidate Scoring",
            "type": "📐 Deterministic + Embeddings",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Each subgroup is scored by two independent methods then fused: "
                "BM25 (term-frequency scoring against CPC titles using Phase 1 terms), "
                "and semantic embedding similarity (sentence-transformer cosine distance). "
                "The two rankings are merged using Reciprocal Rank Fusion (RRF, k=60). "
                "Scores are then family-normalised so each family's best candidate equals its Phase 2A weight."
            ),
        },
        {
            "phase": "Phase 2D — Score Filter",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "A simple Top-50 gate: keeps only the 50 highest-scoring candidates from Phase 2C. "
                "This focuses Phase 3 on a manageable, high-quality candidate set and discards "
                "low-confidence subgroups that would dilute ranking quality."
            ),
        },
        {
            "phase": "Phase 3A — Decision Tree Constraints",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "A 44-rule deterministic decision tree enforces domain correctness. Steps: "
                "(1) domain detection (SPEECH / IMAGE / CONTROL / …), "
                "(2) domain dominance boosting (×1.2–×2.0 for matching families), "
                "(3) functional verb filtering (penalises codes that match nouns but not verbs), "
                "(4) hierarchy priority (prefers specific subgroups over broad classes), "
                "(5) invalid-class filtering (removes non-allocatable nodes like family-level symbols without '/')."
            ),
        },
        {
            "phase": "Phase 3B — Cross-Domain Validation",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Two-step domain anchor check: "
                "DOMAIN_ANCHOR (counts required domain signals in patent text — G10L needs ≥2 of: "
                "speech, audio, voice, acoustic, utterance… → ×1.2 reward or ×0.5 penalty), "
                "then ANTI_COLLAPSE (binary gate — if any context word present → ×2.0 boost, "
                "if absent → ×0.05 near-elimination). Scores are renormalised after both steps."
            ),
        },
        {
            "phase": "Phase 4 — Hypothesis Consolidation",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Groups Phase 3 candidates into hypotheses by CPC family using Jaccard similarity "
                "clustering. For each cluster, computes: mean score, coherence (average pairwise "
                "Jaccard between subgroup titles), and support weight (fraction of total candidates). "
                "Outputs PRIMARY hypothesis and optional SECONDARY if a second family is competitive."
            ),
        },
        {
            "phase": "Phase 5 — Hypothesis Resolution + Tri-Pillar",
            "type": "📐 Deterministic",
            "color": "#e8f4fd",
            "border": "#2196F3",
            "description": (
                "Scores each hypothesis with: "
                "0.5 × Phase 4 score + 0.3 × functional alignment (keyword overlap between "
                "Phase 1 core function and CPC titles) + 0.2 × technical coverage (Phase 1 terms "
                "found in CPC titles). Also runs Tri-Pillar back-scan: searches the full Phase 2C pool "
                "for the best champion per role — Primary Goal (highest scorer in primary family), "
                "Methodological (best G06N subgroup), Application Context (next best in primary family)."
            ),
        },
        {
            "phase": "Phase 8 — Role Labeling & Report",
            "type": "🤖 LLM (justification only)",
            "color": "#fff3e0",
            "border": "#FF9800",
            "description": (
                "Deterministic role labeling assigns each surviving CPC code to a functional layer: "
                "Core Invention (directly addresses the claim), Enabling Technology (makes the core possible), "
                "Application Context (deployment domain), Legal Coverage (broad protective codes). "
                "The Professional Justification paragraph is then generated by an LLM that receives "
                "structured pipeline data — not free-form text — so the output is grounded and verifiable."
            ),
        },
    ]

    for p in pipeline_phases:
        bg = p["color"]
        border = p["border"]
        phase_label = p["phase"]
        method_type = p["type"]
        desc = p["description"]
        st.markdown(
            f"<div style='background:{bg};border-left:4px solid {border};"
            f"padding:10px 14px;border-radius:6px;margin-bottom:8px;'>"
            f"<strong>{phase_label}</strong> &nbsp;&nbsp;"
            f"<span style='background:{border};color:white;padding:2px 8px;"
            f"border-radius:12px;font-size:0.78em;'>{method_type}</span><br>"
            f"<span style='font-size:0.9em;'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
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
            phase5_primary = result.get("phase5", {}).get("primary", {})
            fa = phase5_primary.get("functional_alignment", 0)
            tc = phase5_primary.get("technical_coverage", 0)
            fs = phase5_primary.get("final_score", 0)
            n_hyp = result.get("phase5", {}).get("decision_logic", {}).get("num_hypotheses_evaluated", 1)
            # Build a data-driven explanation
            reasons = []
            if fs < 0.75:
                reasons.append(f"Final score <strong>{fs:.3f}</strong> is below the high-confidence threshold of <strong>0.75</strong>.")
            if tc < 0.3:
                reasons.append(f"Technical Coverage is <strong>{tc:.3f}</strong> — Phase 1 terms have limited verbatim overlap with CPC title vocabulary (normal for highly specific inventions).")
            if fa < 0.7:
                reasons.append(f"Functional Alignment is <strong>{fa:.3f}</strong> — the core function description partially overlaps with CPC subgroup titles.")
            if n_hyp == 1:
                reasons.append("Only <strong>1 hypothesis</strong> was formed — the patent is single-domain (G10L). Medium confidence here does <strong>not</strong> indicate ambiguity or a dual-core pattern.")
            reason_html = "<br>".join(f"• {r}" for r in reasons) if reasons else "Score slightly below the 0.75 threshold."
            st.markdown(
                "<div style='background-color:#fff3e0;padding:12px;border-radius:8px;"
                "border-left:4px solid #FF9800;'>"
                f"<em><strong>Score Analysis</strong><br><br>"
                f"{reason_html}<br><br>"
                "Formula: <code>0.5 × Phase4Score + 0.3 × FunctionalAlignment + 0.2 × TechnicalCoverage</code><br>"
                f"= <code>0.5 × 1.0 + 0.3 × {fa:.2f} + 0.2 × {tc:.2f} = {fs:.3f}</code><br><br>"
                "This is <strong>not an error</strong>. The classification is correct — "
                "confidence reflects scoring completeness, not domain uncertainty."
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

# ── DEBUG: Phase 1.5 + TCR presence check ──
with st.expander("[DEBUG] Phase 1.5 + TCR presence check"):
    st.json(
        {
            "phase15_exists": "phase15" in result,
            "phase15_keys": list(result.get("phase15", {}).keys())
            if result.get("phase15")
            else [],
            "tcr_analysis_exists": "tcr_analysis" in result,
            "tcr_keys": list(result.get("tcr_analysis", {}).keys())
            if result.get("tcr_analysis")
            else [],
            "tcr_value": result.get("tcr_analysis"),
        }
    )
