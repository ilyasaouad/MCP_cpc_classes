"""
streamlit_app_2.py — CPC Classifier client for patent_cpc_API (v2).

Targets:
  MCP server  → localhost:3456  (same as app 1)
  FastAPI v2  → localhost:8001/api/v2  (patent_cpc_API)

Do NOT modify streamlit_app.py — that targets patent_cpc_fastapi on port 8000.
"""

import json
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from mcp_client import CPCRestClient

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001/api/v2"
MCP_BASE = "http://localhost:3456"

st.set_page_config(
    page_title="CPC Classifier v2 — patent_cpc_API",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# PHASE LIST  (new 5-group naming)
# ─────────────────────────────────────────────────────────────
PHASES = [
    "1A — Signal Extraction",
    "1B — Claims Audit",
    "1C — Technical Character",
    "2A — Family Routing",
    "2B — Subgroup Expansion",
    "2C — Hybrid Scoring",
    "2D — Candidate Filter",
    "3A — Decision Tree",
    "3B — Cross-Domain Validation",
    "4A — Hypothesis Consolidation",
    "4B — Hypothesis Resolution",
    "5A — Consistency Check",
    "5B — Final Report",
]

PHASE_KEYS = {
    "1A — Signal Extraction":       "phase1a",
    "1B — Claims Audit":            "phase1b",
    "1C — Technical Character":     "phase1c",
    "2A — Family Routing":          "phase2a",
    "2B — Subgroup Expansion":      "phase2b",
    "2C — Hybrid Scoring":          "phase2c",
    "2D — Candidate Filter":        "phase2d",
    "3A — Decision Tree":           "phase3a",
    "3B — Cross-Domain Validation": "phase3b",
    "4A — Hypothesis Consolidation":"phase4a",
    "4B — Hypothesis Resolution":   "phase4b",
    "5A — Consistency Check":       "phase5a",
    "5B — Final Report":            "phase5b",
}

if "current_phase" not in st.session_state:
    st.session_state["current_phase"] = PHASES[0]
if "result" not in st.session_state:
    st.session_state["result"] = None

client = CPCRestClient(base_url=MCP_BASE)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def check_api_health():
    try:
        req = urllib.request.Request(f"{API_BASE}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def call_classify(query: str, claims: str, description: str, debug: bool) -> dict:
    payload = json.dumps(
        {"query": query, "claims": claims or None, "description": description or None}
    ).encode("utf-8")
    url = f"{API_BASE}/classify?debug={'true' if debug else 'false'}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"error": str(exc)}


def conf_badge(conf: str):
    conf = (conf or "").lower()
    if conf == "high":
        st.success("✅ High Confidence")
    elif conf == "medium":
        st.info("🔶 Medium Confidence")
    else:
        st.warning("⚠️ Low Confidence")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🔬 CPC Classifier v2 — patent_cpc_API")
st.caption(
    "13-phase pipeline · 5 logical groups · "
    "MCP `localhost:3456` · API `localhost:8001/api/v2`"
)

health = check_api_health()
if health.get("status") == "ok":
    kg = "✅ KG loaded" if health.get("kg_loaded") else "⚠️ KG not loaded"
    st.success(f"**API Ready** — {kg}")
else:
    err = health.get("error", "unknown")
    st.error(
        f"🚨 **API not responding** — `{err}`\n\n"
        "Start with:\n"
        "```bash\n"
        "uvicorn patent_cpc_API.main:app --reload --port 8001\n"
        "```"
    )

# ─────────────────────────────────────────────────────────────
# PHASE NAVIGATOR
# ─────────────────────────────────────────────────────────────
st.divider()
nav_l, nav_sel, nav_r = st.columns([1, 10, 1])
with nav_l:
    if st.button("◀", use_container_width=True):
        idx = PHASES.index(st.session_state["current_phase"])
        if idx > 0:
            st.session_state["current_phase"] = PHASES[idx - 1]
        st.rerun()
with nav_sel:
    sel = st.selectbox(
        "Phase",
        PHASES,
        index=PHASES.index(st.session_state["current_phase"]),
        label_visibility="collapsed",
    )
    st.session_state["current_phase"] = sel
with nav_r:
    if st.button("▶", use_container_width=True):
        idx = PHASES.index(st.session_state["current_phase"])
        if idx < len(PHASES) - 1:
            st.session_state["current_phase"] = PHASES[idx + 1]
        st.rerun()
st.divider()

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
col_in1, col_in2 = st.columns(2)
with col_in1:
    query_input = st.text_area(
        "Patent Abstract / Title",
        height=200,
        placeholder="Paste abstract or technical description here…",
    )
with col_in2:
    claims_input = st.text_area(
        "Claims (recommended)",
        height=200,
        placeholder="Claims text — used for Phase 1B forensic audit…",
    )

description_input = st.text_area(
    "Detailed Description (optional)",
    height=120,
    placeholder="Full description section — boosts Phase 1A term extraction…",
)

uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded:
    query_input = uploaded.read().decode("utf-8")

debug_mode = st.checkbox(
    "Include per-phase debug data in response",
    value=False,
    help="Adds raw phase dicts to the response — useful for debugging but slower to render.",
)

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Pipeline", type="primary"):
    if not query_input.strip():
        st.warning("Please provide at least an abstract or title.")
        st.stop()
    if len(query_input.strip()) < 50:
        st.warning(f"Text too short ({len(query_input.strip())} chars). Minimum 50.")
        st.stop()

    with st.spinner("Running 13-phase CPC pipeline — ~30–60 seconds…"):
        res = call_classify(
            query_input, claims_input, description_input, debug_mode
        )
    st.session_state["result"] = res

result = st.session_state.get("result")
if result is None:
    st.info("Enter patent text above and click **Run Pipeline** to start.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# ERROR
# ─────────────────────────────────────────────────────────────
if "error" in result:
    st.error(f"**Pipeline error:** {result['error']}")
    with st.expander("Full response"):
        st.json(result)
    st.stop()

# ─────────────────────────────────────────────────────────────
# PIPELINE WARNINGS / ERRORS BANNER
# ─────────────────────────────────────────────────────────────
if result.get("errors"):
    with st.expander(f"⚠️ {len(result['errors'])} pipeline error(s)"):
        for e in result["errors"]:
            st.error(e)

if result.get("warnings"):
    with st.expander(f"ℹ️ {len(result['warnings'])} warning(s)"):
        for w in result["warnings"]:
            st.warning(w)

# ─────────────────────────────────────────────────────────────
# RETRIEVE COMMON VALUES
# ─────────────────────────────────────────────────────────────
primary_cpc   = result.get("primary_cpc", "")
primary_title = result.get("primary_title", "")
confidence    = result.get("confidence", "LOW")
score         = result.get("score", 0.0)
pillars       = result.get("pillars", [])
justification = result.get("justification", "")
supporting    = result.get("supporting_codes", [])
elapsed_ms    = result.get("elapsed_ms", 0)
phase_details = result.get("phase_details", {}) or {}  # only present when debug=true

current_phase = st.session_state["current_phase"]
st.caption(f"**Viewing:** {current_phase}  ·  elapsed: {elapsed_ms} ms")
st.divider()


# ═══════════════════════════════════════════════════════════════════════
# 1A — SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════
if current_phase == "1A — Signal Extraction":
    st.subheader("Phase 1A — Signal Extraction")
    st.caption("LLM multi-pass extraction: terms, core function, domain signals, inventive step.")

    p = phase_details.get("phase1a", {})
    if not p:
        st.warning("Phase 1A data not available. Run with **Include debug data** enabled.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Technical Object**")
        st.info(p.get("technical_object", "N/A"))
        st.markdown("**Core Function**")
        st.success(p.get("core_function", "N/A"))
    with col_b:
        st.markdown("**Inventive Step**")
        st.info(p.get("inventive_step", "N/A"))
        st.markdown("**System Context**")
        st.info(p.get("system_context", "N/A"))

    st.markdown("---")
    st.markdown("**Domain Signals**")
    signals = p.get("domain_signals", [])
    if signals:
        for s in signals[:8]:
            if isinstance(s, dict):
                st.write(
                    f"- **{s.get('name', s.get('label', ''))}** "
                    f"`{s.get('cpc_family', '')}` "
                    f"(conf: {s.get('confidence', 0):.2f})"
                )
    else:
        st.write("None extracted")

    st.markdown("---")
    st.markdown("**Extracted Terms**")
    terms = p.get("terms", p.get("essential_terms", []))
    if terms:
        df = pd.DataFrame(terms)
        if "importance" in df.columns:
            df = df.sort_values("importance", ascending=False)
        st.dataframe(df, use_container_width=True)
        if "importance" in df.columns and "term" in df.columns:
            st.bar_chart(df.set_index("term")[["importance"]])
    else:
        st.write("No terms extracted")


# ═══════════════════════════════════════════════════════════════════════
# 1B — CLAIMS AUDIT
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "1B — Claims Audit":
    st.subheader("Phase 1B — Forensic Claims Audit")
    st.caption(
        "Validates which CPC families are actually evidenced in the claims text. "
        "Produces anchor_families (confirmed) and kill_log (rejected — cannot re-enter)."
    )

    p = phase_details.get("phase1b", {})
    if not p:
        st.warning("Phase 1B data not available. Enable debug mode.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Anchor Families** (claim-evidenced)")
        anchors = p.get("anchor_families", [])
        if anchors:
            for a in anchors:
                st.success(f"✅ `{a}`")
        else:
            st.write("None")

    with col_b:
        st.markdown("**Kill Log** (rejected — hard block)")
        killed = p.get("kill_log", [])
        if killed:
            for k in killed:
                st.error(f"❌ `{k}`")
        else:
            st.write("None")

    primary_anchor = p.get("final_primary_anchor", "")
    if primary_anchor:
        st.markdown(f"**Final Primary Anchor:** `{primary_anchor}`")

    dw = p.get("domain_weights", {})
    if dw:
        st.markdown("---")
        st.markdown("**Domain Weights**")
        st.json(dw)


# ═══════════════════════════════════════════════════════════════════════
# 1C — TECHNICAL CHARACTER
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "1C — Technical Character":
    st.subheader("Phase 1C — Technical Character Analysis")
    st.caption(
        "Computes the Technology Computation Ratio (TCR) and classifies the patent role. "
        "Drives Phase 2A routing toward software-heavy or hardware-heavy CPC families."
    )

    p = phase_details.get("phase1c", {})
    if not p:
        st.warning("Phase 1C data not available. Enable debug mode.")
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Role", p.get("role", "N/A"))
        st.metric("Role Confidence", f"{p.get('role_confidence', 0):.2f}")
    with col_b:
        st.metric("TCR Score", f"{p.get('tcr_score', 0):.3f}")
        st.metric("TCR Bias", f"{p.get('tcr_bias', 0):.3f}")
    with col_c:
        label = p.get("tcr_label", "neutral")
        if label == "software":
            st.success("Software Bias")
        elif label == "hardware":
            st.error("Hardware Bias")
        else:
            st.info("Neutral")

    with st.expander("TCR Details"):
        st.json(p.get("tcr_details", {}))


# ═══════════════════════════════════════════════════════════════════════
# 2A — FAMILY ROUTING
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "2A — Family Routing":
    st.subheader("Phase 2A — CPC Family Routing")
    st.caption(
        "Scores 4-char CPC families using embedding + KG proximity + hard anchors. "
        "These weights are used later to normalise Phase 2C scores."
    )

    p = phase_details.get("phase2a", {})
    if not p:
        st.warning("Phase 2A data not available. Enable debug mode.")
        st.stop()

    family_scores: dict = p.get("family_scores", {})
    if family_scores:
        df = pd.DataFrame(
            [{"Family": k, "Weight": v} for k, v in sorted(family_scores.items(), key=lambda x: x[1], reverse=True)]
        )
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Family")[["Weight"]])
    else:
        st.write("No family scores available")

    st.markdown(f"**Domain Confidence:** `{p.get('domain_confidence', 0):.2f}`")

    with st.expander("Layer Result"):
        st.json(p.get("layer_result", {}))


# ═══════════════════════════════════════════════════════════════════════
# 2B — SUBGROUP EXPANSION
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "2B — Subgroup Expansion":
    st.subheader("Phase 2B — Subgroup Expansion")
    st.caption(
        "Expands each Phase 2A family into individual subgroups via KG hierarchy + XML titles. "
        "Non-allocatable nodes (no '/' in symbol, cross-ref codes) are filtered here."
    )

    p = phase_details.get("phase2b", {})
    if not p:
        st.warning("Phase 2B data not available. Enable debug mode.")
        st.stop()

    candidates = p.get("expanded_candidates", [])
    st.metric("Expanded Candidates", len(candidates))

    if candidates:
        df = pd.DataFrame(candidates)
        cols = [c for c in ["symbol", "title", "family", "score"] if c in df.columns]
        st.dataframe(df[cols].head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 2C — HYBRID SCORING
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "2C — Hybrid Scoring":
    st.subheader("Phase 2C — Hybrid Scoring")
    st.caption(
        "BM25 lexical + semantic embedding scored via Reciprocal Rank Fusion (RRF). "
        "Family-level normalisation: best in family rescaled to its Phase 2A routing weight."
    )

    p = phase_details.get("phase2c", {})
    if not p:
        st.warning("Phase 2C data not available. Enable debug mode.")
        st.stop()

    scored = p.get("scored_candidates", [])
    st.metric("Scored Candidates", len(scored))

    if scored:
        df = pd.DataFrame(scored)
        cols = [c for c in ["symbol", "title", "score", "family"] if c in df.columns]
        st.dataframe(df[cols].head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 2D — CANDIDATE FILTER
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "2D — Candidate Filter":
    st.subheader("Phase 2D — Candidate Filter")
    st.caption(
        "Keeps top-N candidates after Phase 2C. "
        "Preserves the full raw pool (all_raw_candidates) for Phase 4B title lookup."
    )

    p = phase_details.get("phase2d", {})
    if not p:
        st.warning("Phase 2D data not available. Enable debug mode.")
        st.stop()

    candidates = p.get("candidates", [])
    raw = p.get("all_raw_candidates", [])
    col_a, col_b = st.columns(2)
    col_a.metric("Top-N Candidates", len(candidates))
    col_b.metric("Raw Pool Size", len(raw))

    if candidates:
        df = pd.DataFrame(candidates)
        cols = [c for c in ["symbol", "title", "score", "family"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 3A — DECISION TREE
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "3A — Decision Tree":
    st.subheader("Phase 3A — Decision Tree Constraints")
    st.caption(
        "Deterministic rule layer: domain boosts, invalid-class penalties, "
        "functional boosting. Always runs — ensures domain correctness before clustering."
    )

    p = phase_details.get("phase3a", {})
    if not p:
        st.warning("Phase 3A data not available. Enable debug mode.")
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Adjustments Applied", p.get("adjustments", 0))
    col_b.metric("Domain", p.get("domain", "unknown"))
    col_c.metric("Domain Confidence", f"{p.get('domain_confidence', 0):.2f}")

    st.markdown(f"**Layer Mode:** `{p.get('layer_mode', False)}`")

    candidates = p.get("candidates", [])
    if candidates:
        df = pd.DataFrame(candidates)
        cols = [c for c in ["symbol", "title", "score", "family"] if c in df.columns]
        st.dataframe(df[cols].head(20), use_container_width=True)

    with st.expander("Constraint Details"):
        st.json(p.get("constraint_details", {}))


# ═══════════════════════════════════════════════════════════════════════
# 3B — CROSS-DOMAIN VALIDATION
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "3B — Cross-Domain Validation":
    st.subheader("Phase 3B — Cross-Domain Validation")
    st.caption(
        "Anchor-based scoring: DOMAIN_ANCHOR_CONFIRMED boosts the primary family when "
        "claim signals are verified. ANTI_COLLAPSE_BOOST lifts secondary codes when "
        "domain context is present; ANTI_COLLAPSE_PENALTY suppresses them when absent."
    )

    p = phase_details.get("phase3b", {})
    if not p:
        st.warning("Phase 3B data not available. Enable debug mode.")
        st.stop()

    col_a, col_b = st.columns(2)
    verified = p.get("domain_verified", False)
    col_a.metric("Domain Verified", "✅ Yes" if verified else "❌ No")
    col_b.metric("Adjustments", p.get("adjustments", 0))

    rules = p.get("validation_rules", [])
    if rules:
        st.markdown("**Rules Applied:**")
        for r in rules:
            st.write(f"- {r}")

    candidates = p.get("candidates", [])
    if candidates:
        df = pd.DataFrame(candidates)
        cols = [c for c in ["symbol", "title", "score", "family"] if c in df.columns]
        st.dataframe(df[cols].head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 4A — HYPOTHESIS CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "4A — Hypothesis Consolidation":
    st.subheader("Phase 4A — Hypothesis Consolidation")
    st.caption(
        "Clusters candidates by CPC family into up to 3 hypotheses. "
        "Each hypothesis has: family, coherence score, and support_weight (% of candidate mass). "
        "Note: high support_weight + low coherence = sub-branch spread, not hallucination."
    )

    p = phase_details.get("phase4a", {})
    if not p:
        st.warning("Phase 4A data not available. Enable debug mode.")
        st.stop()

    hypotheses = p.get("hypotheses", [])
    st.metric("Hypotheses", len(hypotheses))

    for i, h in enumerate(hypotheses):
        fam = h.get("family", "?")
        coh = h.get("coherence", 0)
        sup = h.get("support_weight", 0)
        label = "PRIMARY" if i == 0 else f"#{i+1}"
        with st.expander(f"{label} — `{fam}`  coherence={coh:.2f}  support={sup:.2f}"):
            sc = h.get("candidates", [])
            if sc:
                df = pd.DataFrame(sc)
                cols = [c for c in ["symbol", "title", "score"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 4B — HYPOTHESIS RESOLUTION
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "4B — Hypothesis Resolution":
    st.subheader("Phase 4B — Hypothesis Resolution")
    st.caption(
        "Scores each hypothesis: 0.5 × phase4_evidence + 0.3 × functional_alignment "
        "+ 0.2 × technical_coverage. Picks winner and resolves the Tri-Pillar."
    )

    p = phase_details.get("phase4b", {})
    if not p:
        # Fall back to top-level response fields
        p = {
            "primary_cpc": primary_cpc,
            "primary_title": primary_title,
            "confidence": confidence,
            "score": score,
            "pillars": pillars,
        }

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Primary CPC", p.get("primary_cpc", primary_cpc))
    col_b.metric("Score", f"{p.get('score', score):.3f}")
    conf_badge(p.get("confidence", confidence))

    fa = p.get("functional_alignment", 0)
    tc = p.get("technical_coverage", 0)
    if fa or tc:
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Functional Alignment", f"{fa:.3f}")
        c2.metric("Technical Coverage", f"{tc:.3f}")

    # Tri-Pillar
    raw_pillars = p.get("pillars", pillars)
    if raw_pillars:
        st.markdown("---")
        st.markdown("### Tri-Pillar Breakdown")
        st.caption(
            "**Primary Goal** — the core legal claim (what the invention produces). "
            "**AI Methodology** — how it works internally (the ML engine). "
            "**Domain Context** — the technical environment it applies to."
        )
        role_icons = {
            "primary_goal":    "🎯 Primary Goal",
            "ai_methodology":  "🧠 AI Methodology",
            "domain_context":  "⚙️ Domain Context",
        }
        rows = []
        for pillar in raw_pillars:
            if isinstance(pillar, dict):
                role = pillar.get("role", "")
                rows.append({
                    "Role": role_icons.get(role, role),
                    "CPC Code": pillar.get("symbol", ""),
                    "Title": pillar.get("title", ""),
                    "Family": pillar.get("family", ""),
                })
        if rows:
            st.table(pd.DataFrame(rows))


# ═══════════════════════════════════════════════════════════════════════
# 5A — CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "5A — Consistency Check":
    st.subheader("Phase 5A — LLM Consistency Check")
    st.caption(
        "Final LLM sanity check: is the Phase 4B primary CPC self-consistent "
        "with the full candidate pool? Any reranking recommendations?"
    )

    p = phase_details.get("phase5a", {})
    if not p:
        st.warning("Phase 5A data not available. Enable debug mode.")
        st.stop()

    is_ok = p.get("is_consistent", True)
    if is_ok:
        st.success("✅ Classification is internally consistent")
    else:
        st.warning("⚠️ Consistency check flagged potential issues")

    adj = p.get("adjustments", [])
    if adj:
        st.markdown("**Adjustments Recommended:**")
        for a in adj:
            st.write(f"- {a}")

    with st.expander("Consistency Check Details"):
        st.json(p.get("consistency_check", {}))


# ═══════════════════════════════════════════════════════════════════════
# 5B — FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════
elif current_phase == "5B — Final Report":
    st.subheader("📄 Executive Patent Classification Report")
    st.caption(
        f"Pipeline completed in **{elapsed_ms} ms** across 13 phases."
    )

    # ── Hero: main recommendation ─────────────────────────────────────
    col_hero, col_badge = st.columns([3, 1])
    with col_hero:
        if primary_cpc:
            st.markdown(f"### Main Recommendation: `{primary_cpc}`")
            st.markdown(f"*{primary_title}*")
            st.caption(
                "This code survived all 13 pipeline phases: signal extraction → "
                "claims audit → TCR analysis → family routing → KG expansion → "
                "hybrid scoring → decision tree → cross-domain validation → "
                "hypothesis clustering → hypothesis resolution → consistency check."
            )
        else:
            st.warning("No primary classification produced.")
    with col_badge:
        conf_badge(confidence)
        st.metric("Score", f"{score:.3f}")

    # ── Technical Breakdown ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠 Technical Breakdown (Tri-Pillar)")
    st.caption(
        "Every patent has three simultaneous technical dimensions. "
        "**Primary Goal** = what it *produces* (the claimed output). "
        "**AI Methodology** = *how* it works (the ML engine). "
        "**Domain Context** = *where* it applies (the technical field)."
    )

    role_icons = {
        "primary_goal":   "🎯 Primary Goal",
        "ai_methodology": "🧠 AI Methodology",
        "domain_context": "⚙️ Domain Context",
    }
    if pillars:
        rows = []
        for p_item in pillars:
            if isinstance(p_item, dict):
                role = p_item.get("role", "")
                sym  = p_item.get("symbol", "")
                rows.append({
                    "Role": role_icons.get(role, role),
                    "CPC Code": f"`{sym}`" if sym else "*Not found*",
                    "Title": p_item.get("title", "N/A"),
                })
        if rows:
            st.table(pd.DataFrame(rows))
    else:
        st.info("Tri-Pillar not available (Phase 4B did not produce pillars).")

    # ── Supporting Classification Details ─────────────────────────────
    if supporting:
        st.markdown("---")
        st.markdown("### 📎 Supporting Classification Details")
        st.caption(
            "Additional CPC codes assigned by the pipeline to capture enabling technology, "
            "application context, and legal coverage beyond the primary claim."
        )
        core_codes    = [s for s in supporting if isinstance(s, dict) and s.get("role") in ("CORE", "core")]
        support_codes = [s for s in supporting if isinstance(s, dict) and s.get("role") in ("SUPPORT", "support")]
        context_codes = [s for s in supporting if isinstance(s, dict) and s.get("role") in ("CONTEXT", "context", "APPLICATION", "application")]

        col_c, col_s, col_x = st.columns(3)
        with col_c:
            st.markdown("**Core / Primary**")
            for c in core_codes or supporting[:2]:
                sym = c.get("symbol", "") if isinstance(c, dict) else str(c)
                ttl = c.get("title", "") if isinstance(c, dict) else ""
                st.markdown(f"`{sym}`")
                if ttl:
                    st.caption(ttl)
        with col_s:
            st.markdown("**Enabling Technology**")
            for c in support_codes:
                sym = c.get("symbol", "")
                ttl = c.get("title", "")
                st.markdown(f"`{sym}`")
                if ttl:
                    st.caption(ttl)
        with col_x:
            st.markdown("**Application Context**")
            for c in context_codes:
                sym = c.get("symbol", "")
                ttl = c.get("title", "")
                st.markdown(f"`{sym}`")
                if ttl:
                    st.caption(ttl)

    # ── Professional Justification ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Professional Justification")
    if justification:
        st.markdown(justification)
    else:
        st.info("Justification not available (Phase 5B LLM call may have failed).")

    # ── Methods & Algorithms ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Methods & Algorithms Used")
    st.caption(
        "Each pipeline phase and whether it uses an LLM or a deterministic algorithm."
    )

    _LLM   = "#FF9800"
    _DET   = "#2196F3"
    _BADGE_LLM = f"background:{_LLM};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;"
    _BADGE_DET = f"background:{_DET};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;"

    phase_cards = [
        ("1A — Signal Extraction",        "LLM",          "Multi-pass extraction: terms, core function, domain signals, inventive step."),
        ("1B — Claims Audit",             "LLM",          "Forensic audit of claim text → anchor_families, kill_log."),
        ("1C — Technical Character",      "LLM + Det.",   "LLM role classification + deterministic TCR score."),
        ("2A — Family Routing",           "Deterministic","Embedding + KG + anchor blend → family_scores."),
        ("2B — Subgroup Expansion",       "Deterministic","KG hierarchy BFS + XML title pre-loading per family."),
        ("2C — Hybrid Scoring",           "Deterministic","BM25 + semantic RRF + family-level normalisation."),
        ("2D — Candidate Filter",         "Deterministic","Top-N slice + anchor promotion + raw pool preservation."),
        ("3A — Decision Tree",            "Deterministic","Rule-based: domain boosts, invalid-class penalties, functional boosting."),
        ("3B — Cross-Domain Validation",  "Deterministic","Anchor boosts, anti-collapse rules, domain-verified flag."),
        ("4A — Hypothesis Consolidation", "Deterministic","Jaccard-based family clustering → coherence + support_weight."),
        ("4B — Hypothesis Resolution",    "Deterministic","0.5×phase4 + 0.3×FA + 0.2×TC → winner + Tri-Pillar."),
        ("5A — Consistency Check",        "LLM",          "LLM sanity check on Phase 4B primary vs full candidate pool."),
        ("5B — Final Report",             "LLM",          "Role labeling (CORE/SUPPORT/CONTEXT) + professional justification text."),
    ]

    cols = st.columns(3)
    for i, (name, method, desc) in enumerate(phase_cards):
        badge_style = _BADGE_LLM if "LLM" in method else _BADGE_DET
        badge_label = "🤖 LLM" if "LLM" in method else "📐 Deterministic"
        with cols[i % 3]:
            st.markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:10px;"
                f"margin-bottom:8px;'>"
                f"<strong style='font-size:0.85em'>{name}</strong><br>"
                f"<span style='{badge_style}'>{badge_label}</span><br>"
                f"<span style='font-size:0.82em;color:#555;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Analysis expanders ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Analysis & Logic")

    with st.expander("💡 How was the Main Recommendation selected?"):
        st.markdown(
            "The primary CPC was selected by Phase 4B after scoring all hypothesis clusters "
            "on three axes: **Phase 4A cluster evidence** (0.5), **Functional Alignment** — "
            "keyword overlap between the core function description and CPC titles (0.3), and "
            "**Technical Coverage** — how many Phase 1A terms appear in CPC titles (0.2). "
            "The winning hypothesis passed Phases 1B (claim-evidenced), 3A (domain-correct), "
            "and 3B (cross-domain consistent) before entering the scoring stage."
        )

    if (confidence or "").lower() != "high":
        with st.expander("💡 Why not higher confidence?"):
            p4b = phase_details.get("phase4b", {})
            fa = p4b.get("functional_alignment", 0)
            tc = p4b.get("technical_coverage", 0)
            reasons = []
            if score < 0.75:
                reasons.append(f"Final score **{score:.3f}** is below the high-confidence threshold of **0.75**.")
            if tc < 0.3:
                reasons.append(f"Technical Coverage **{tc:.3f}** — Phase 1 terms have limited overlap with short CPC title vocabulary (normal for specific inventions).")
            if fa < 0.7:
                reasons.append(f"Functional Alignment **{fa:.3f}** — core function description partially overlaps with CPC subgroup titles.")
            if not reasons:
                reasons.append("Score did not exceed the 0.75 high-confidence threshold.")
            for r in reasons:
                st.markdown(f"- {r}")
            st.caption(
                "Note: a single-domain patent with no competing hypotheses can still be "
                "medium confidence if term vocabulary differs from CPC official titles — "
                "it does not mean the classification is ambiguous."
            )

    if debug_mode and phase_details:
        with st.expander("🛠 Raw phase_details (debug)"):
            st.json(phase_details)
