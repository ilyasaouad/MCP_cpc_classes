# 🏗️ Patent CPC Classification System — Architecture Report

> **Generated:** 2026-04-30  
> **Author:** Antigravity AI  
> **Scope:** Three-repository ecosystem overview

---

## 1. System Overview

This ecosystem implements a **three-layer AI-powered patent classification pipeline** that takes raw patent text and produces Cooperative Patent Classification (CPC) codes. The three repositories form a clean separation of concerns:

| Layer | Repository | Technology | Role |
|---|---|---|---|
| **Frontend / Client** | `cpc_client` | Python · Streamlit | User interface & HTTP client |
| **Middleware / Bridge** | `MCP_patent_classification` | TypeScript · Node.js · Express | MCP gateway & REST proxy |
| **Backend / AI Engine** | `patent_cpc_fastapi` | Python · FastAPI · Ollama | LLM classification engine |

---

## 2. Request Flow (End-to-End)

```
┌─────────────────────────────┐
│  cpc_client                 │
│  (Streamlit UI)             │
│                             │
│  User pastes patent text    │
│  → clicks "Classify CPC"   │
│                             │
│  POST http://localhost:3456 │
│       /cpc/classify         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  MCP_patent_classification  │
│  (Node.js / Express)        │
│                             │
│  Receives request on :3456  │
│  Proxies to FastAPI         │
│                             │
│  POST http://localhost:8000 │
│       /classify             │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  patent_cpc_fastapi         │
│  (FastAPI / Python)         │
│                             │
│  Phase 1: LLM extraction    │
│  Phase 2: Final coding      │
│  Returns JSON with          │
│  phase1, phase2, cpc[]      │
└─────────────────────────────┘
             │
             ▼
     Response bubbles back
     up through MCP → Streamlit
```

---

## 3. Repository Deep Dive

### 3.1 `patent_cpc_fastapi` — The AI Engine (Port 8000)

**Stack:** Python · FastAPI · Ollama · EPO Linked Open Data API

This is the **brain** of the system. It exposes a single POST endpoint `/classify` that runs a two-phase LLM classification pipeline.

#### Key Files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI entry point, orchestrates the two-phase pipeline |
| `app/llm_engine.py` | Simple single-phase LLM wrapper (legacy/prototype) |
| `app/classifier.py` | Rule-based keyword fallback classifier (placeholder) |
| `app/cpc_classification/search_cpc.py` | **Main pipeline orchestrator** — `CPCClassifier` class |
| `app/cpc_classification/extracting_cpc.py` | Phase 1: LLM extracts technical terms + CPC class candidates |
| `app/cpc_classification/epo_client.py` | Fetches CPC hierarchy from live EPO Linked Open Data API |
| `app/cpc_classification/analysing_cpc.py` | Phase 2/3: Scores CPC candidates using Ollama embeddings (cosine similarity) |
| `app/cpc_classification/resources/` | Static CPC hint text files used in prompts |

#### Classification Pipeline (inside `main.py`)

```
Patent Text Input
      │
      ▼
Phase 1 — LLM Call (Ollama / gpt-oss:120b-cloud)
  Extract:
    - problem statement
    - method description
    - technical terms (with importance weights 1–5)
    - main CPC classes (e.g. G06, H04)
    - CPC sections (e.g. G, H)
      │
      ▼
Phase 2 — LLM Call (second call)
  Input: terms from Phase 1 + class candidates
  Output:
    - final CPC codes (e.g. G06N3-00)
    - reasoning explanation
      │
      ▼
Response JSON:
  {
    "phase1": { problem, method, terms[], cpc_classes[], cpc_sections[] },
    "phase2": { codes[], reasoning },
    "cpc":    [ { code, score } ]   ← normalized equal-weight scores
  }
```

> **Note:** There is also a more advanced `CPCClassifier` class in `cpc_classification/search_cpc.py` that adds a **Phase 3 EPO enrichment step** using live API data and embedding-based cosine scoring. This appears to be a newer pipeline that `main.py` does not yet call directly.

---

### 3.2 `MCP_patent_classification` — The MCP Gateway (Port 3456)

**Stack:** TypeScript · Node.js · Express · `@modelcontextprotocol/sdk`

This server acts as a **dual-mode bridge**:

1. **MCP Protocol mode** — Exposes the `classify_patent` tool via the Model Context Protocol (MCP) JSON-RPC over HTTP, allowing any MCP-compatible AI agent or IDE plugin to call the classifier as a tool.
2. **REST proxy mode** — Also exposes a plain REST endpoint (`POST /cpc/classify`) that the Streamlit client uses directly, which internally proxies to FastAPI on port 8000.

#### Key Files

| File | Purpose |
|---|---|
| `src/server/index.ts` | **Main server** — Express + MCP SDK, both modes |
| `src/server/raw-server.ts` | Minimal raw Node.js HTTP MCP prototype (dev/debug only) |
| `src/test-client.ts` | MCP protocol test harness (initialize → call tool) |
| `package.json` | Node.js config — `@modelcontextprotocol/sdk`, `express`, `zod` |
| `.env` | Environment config (`LLM_MODEL`, `PORT`, `SERPAPI_API_KEY`) |

#### Why MCP?

The Model Context Protocol (MCP) is an open standard that lets AI agents (like Claude, Copilot, etc.) discover and call tools defined by a server. By wrapping the FastAPI classifier in an MCP server, the classification engine becomes callable by any MCP-compatible agent — not just the Streamlit UI.

#### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check |
| `/mcp` | POST | MCP JSON-RPC endpoint (for AI agents) |
| `/cpc/classify` | POST | Plain REST proxy (for Streamlit client) |

---

### 3.3 `cpc_client` — The Streamlit UI (calls Port 3456)

**Stack:** Python · Streamlit · pandas · requests

This is the **user-facing interface**. It provides a clean web UI where a user can:
- Paste patent text or upload a `.txt` file
- Trigger the full classification pipeline with one click
- View the pipeline steps (Phase 1 extraction, Phase 2 coding)
- See CPC results as a table + bar chart
- Inspect the raw JSON response

#### Key Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit UI — layout, input, results rendering |
| `mcp_client.py` | Thin HTTP client — wraps `requests.post` to the MCP server |
| `requirements.txt` | Python dependencies |

#### `MCPClient` class (`mcp_client.py`)

```python
class MCPClient:
    def __init__(self, base_url="http://localhost:3456"):
        ...

    def classify_cpc(self, text: str) -> Dict[str, Any]:
        # POST to /cpc/classify on the MCP server
        ...
```

> **Note:** Despite being called `MCPClient`, it does **not** use the MCP JSON-RPC protocol. It calls the plain REST endpoint `/cpc/classify` on the MCP server. This is by design — the MCP server exposes both modes simultaneously.

---

## 4. Port Map

```
localhost:8000  →  patent_cpc_fastapi   (FastAPI / Python)
localhost:3456  →  MCP_patent_classification  (Node.js / Express / MCP)
localhost:8501  →  cpc_client           (Streamlit, default port)
```

---

## 5. Data Model

The canonical JSON response flowing through all layers:

```json
{
  "phase1": {
    "problem": "Short description of the technical problem",
    "method":  "Description of the solution method",
    "cpc_classes": ["G06N", "H04L"],
    "cpc_sections": ["G", "H"],
    "terms": [
      { "term": "neural network", "importance": 5 },
      { "term": "image recognition", "importance": 4 }
    ]
  },
  "phase2": {
    "codes": ["G06N3-00", "H04L12-00"],
    "reasoning": "The patent describes..."
  },
  "cpc": [
    { "code": "G06N3-00", "score": 0.5 },
    { "code": "H04L12-00", "score": 0.5 }
  ]
}
```

> **Scoring:** Currently, the FastAPI backend distributes equal weight (`1 / n_codes`) across all returned codes. The `CPCAnalyzer` module in `cpc_classification/` implements proper cosine-similarity scoring via Ollama embeddings for a more advanced version.

---

## 6. LLM Integration

| Component | LLM Used | How |
|---|---|---|
| `patent_cpc_fastapi` | `gpt-oss:120b-cloud` | Via `ollama.chat()` — runs locally through Ollama |
| `cpc_classification/analysing_cpc.py` | Ollama embedding model | Via `OllamaClient.embeddings()` for cosine similarity scoring |

> **Ollama** acts as the local LLM runtime. The model tag `gpt-oss:120b-cloud` suggests a large model pulled from a cloud-backed Ollama registry.

---

## 7. How to Start the Full Stack

Run these three commands in separate terminals:

```bash
# Terminal 1 — FastAPI backend (port 8000)
cd patent_cpc_fastapi
uvicorn app.main:app --reload --port 8000

# Terminal 2 — MCP Node.js server (port 3456)
cd MCP_patent_classification
npm run dev

# Terminal 3 — Streamlit UI
cd cpc_client
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## 8. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                 │
│                    http://localhost:8501                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cpc_client  (Streamlit · Python)                                    │
│                                                                      │
│  streamlit_app.py ─── mcp_client.py                                  │
│  [UI + file upload]    [HTTP POST /cpc/classify]                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  POST http://localhost:3456/cpc/classify
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MCP_patent_classification  (Express · TypeScript · MCP SDK)         │
│                                                                      │
│  src/server/index.ts                                                 │
│  ├── POST /mcp          ← MCP JSON-RPC (for AI agents)               │
│  ├── POST /cpc/classify ← REST proxy (for Streamlit)                 │
│  └── GET  /             ← Health check                               │
│                                                                      │
│  MCP Tool: classify_patent                                           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  POST http://localhost:8000/classify
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  patent_cpc_fastapi  (FastAPI · Python · Ollama)                     │
│                                                                      │
│  app/main.py                                                         │
│  ├── Phase 1: LLM extracts terms + CPC class candidates              │
│  ├── Phase 2: LLM maps candidates → final codes + reasoning          │
│  └── Returns: { phase1, phase2, cpc[] }                              │
│                                                                      │
│  app/cpc_classification/                                             │
│  ├── extracting_cpc.py  (LLM extraction)                             │
│  ├── epo_client.py      (EPO Linked Open Data API)                   │
│  └── analysing_cpc.py  (embedding cosine scoring)                   │
│                                                                      │
│  LLM Runtime: Ollama (local)  ← gpt-oss:120b-cloud                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Observations & Notes

### ✅ What Works Well
- Clean separation of concerns across three layers
- MCP gateway enables the classifier to be called by any MCP-compatible agent (Claude, Copilot, etc.) — not just the Streamlit UI
- The `cpc_classification/` subpackage in `patent_cpc_fastapi` contains a more sophisticated 3-phase pipeline (LLM → EPO API → embedding scoring) that is modular and reusable

### ⚠️ Things to Be Aware Of
- **Port mismatch risk:** `mcp_client.py` defaults to port `3000` but `streamlit_app.py` overrides to `3456`. The MCP server runs on `3456`. Make sure no hard-coded `3000` references slip through.
- **Two pipelines coexist:** `app/main.py` implements its own inline two-phase pipeline, while `cpc_classification/search_cpc.py` has a separate `CPCClassifier` class. These are **not currently connected** — `main.py` does not import `CPCClassifier`.
- **Equal-weight scoring:** The current scoring in `main.py` distributes equal scores across all returned codes (`1/n`). The embedding-based `CPCAnalyzer` exists but is only used inside the advanced `CPCClassifier` path.
- **Ollama dependency:** The entire backend requires a running Ollama instance with the `gpt-oss:120b-cloud` model pulled. The `.env` sets `LLM_MODEL=llama3` (for MCP) but `main.py` hard-codes `gpt-oss:120b-cloud`.
- **`raw-server.ts`** is a stub/debug server with a placeholder tool response. It should not be used in production.
