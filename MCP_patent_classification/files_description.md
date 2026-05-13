# MCP Patent Classification — File Descriptions

## Overview

This is a **TypeScript/Node.js MCP (Model Context Protocol) server** that acts as a proxy/gateway between LLM agents (Claude Desktop, Cursor, etc.) and the Python FastAPI patent classification backend. It does NOT perform classification itself — it exposes the `classify_patent` MCP tool which forwards requests to `localhost:8000/classify`.

**Last Updated:** 2026-05-12

---

## Directory Structure

```
MCP_patent_classification/
├── .env                          # Runtime environment variables
├── .env.example                  # Template for .env
├── Dockerfile                    # Container build instructions
├── package.json                  # NPM project manifest + scripts
├── package-lock.json             # Locked dependency tree
├── tsconfig.json                 # TypeScript compiler configuration
├── debug-test.ts                 # Automated debug test script
├── src/
│   ├── server/
│   │   ├── index.ts              # ** PRODUCTION ** MCP server (Express + FastAPI proxy)
│   │   └── raw-server.ts         # ** DEBUG/DEV ** Bare-bones stub MCP server
│   └── test-client.ts            # Manual MCP client test
├── .vscode/
│   └── settings.json             # VS Code workspace settings
└── node_modules/                 # Installed npm dependencies (auto-generated)
```

---

## Source Files

### `src/server/index.ts` — Production MCP Server

**Purpose:** Primary MCP server that proxies classification requests to the FastAPI backend.

**Architecture:**
- **Express HTTP server** on port `PORT` (default 3456, from `.env`)
- Two endpoints:
  - `POST /mcp` — MCP protocol endpoint for LLM agents (Claude, Cursor, etc.)
  - `POST /cpc/classify` — Direct REST endpoint for Streamlit clients (bypasses MCP protocol)
- `GET /` — Health-check returning `{status: "MCP server running", tool: "classify_patent", port: PORT}`

**Key Implementation Details:**
- **Per-request isolation**: Creates a new `McpServer` instance + `StreamableHTTPServerTransport` on each request to prevent cross-session state leaking.
- **Global fetch timeout tuning**: Uses undici's `Agent`/`setGlobalDispatcher` to set `headersTimeout`/`bodyTimeout` to 600s (10 min) and `connectTimeout` to 60s — prevents `HeadersTimeoutError` when the FastAPI backend is slow (large patent texts can take 30–120s).
- **Tool: `classify_patent`** — Accepts `text: z.string()`, forwards via `fetch("http://localhost:8000/classify")`, returns JSON as MCP text content.

**Dependencies:**
- `@modelcontextprotocol/sdk` — MCP server framework (`McpServer`, `StreamableHTTPServerTransport`)
- `express` — HTTP framework
- `dotenv` — Loads `.env` into `process.env`
- `zod` — Runtime schema validation for tool parameters
- `undici` — Global fetch agent tuning for timeouts

---

### `src/server/raw-server.ts` — Debug/Dev Stub Server

**Purpose:** Minimal MCP server used during early development to verify SDK transport handshakes.

**Differences from `index.ts`:**
- Uses raw `http.createServer()` instead of Express
- The `classify_patent` tool returns a **hard-coded placeholder** string — does NOT call FastAPI
- Runs on **port 3457** (separate from production port 3456)

**⚠️ DO NOT USE IN PRODUCTION** — This is a smoke-test stub only.

---

### `src/test-client.ts` — Manual MCP Client Test

**Purpose:** Integration test for the full MCP protocol flow.

**Flow:**
1. Sends `initialize` request with `protocolVersion: "2024-11-05"`
2. Captures `mcp-session-id` from response headers
3. Sends `notifications/initialized` to acknowledge initialization
4. Calls `tools/call` with `name: "classify_patent"` and sample text
5. Parses SSE-format responses (looking for `data:` lines), falls back to plain JSON

**Usage:** `npx tsx src/test-client.ts` (requires server running on port 3456)

---

### `debug-test.ts` — Automated Debug Tester

**Purpose:** Quick one-shot test: spawn server, send one request, capture output.

**Flow:**
1. Spawns `src/server/index.ts` as a child process
2. Captures stdout/stderr
3. Waits 3 seconds, then sends an `initialize` JSON-RPC request
4. Logs the response, then kills the server

**Note:** Uses port 3002 (from an older config) — may need updating to match current `.env`.

---

## Configuration Files

### `package.json`

| Field | Value |
|-------|-------|
| **Name** | `mcp_patent_classification` v1.0.0 |
| **Type** | ESM module (`"type": "module"`) |
| **Script `dev`** | `tsx src/server/index.ts` |
| **Script `build`** | `tsc` → compiles to `dist/` |
| **Script `start`** | `node dist/server/index.js` |

**Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| `@modelcontextprotocol/sdk` | ^1.29.0 | MCP server framework |
| `dotenv` | — | Load `.env` at startup |
| `express` | ^5.2.1 | HTTP server framework |
| `zod` | ^3.25.0 | Schema validation for tool inputs |
| `undici` | — | Global fetch timeouts |

**Dev Dependencies:** `typescript` ^6.0.3, `tsx`, `ts-node`, `@types/express`, `@types/node`

---

### `tsconfig.json`

| Setting | Value |
|---------|-------|
| **Target** | ES2022 |
| **Module** | NodeNext / NodeNext resolution |
| **Root** | `./src` |
| **Output** | `./dist` |
| **Strict** | Enabled + extra checks |
| **Source maps** | Generated |
| **Declarations** | Generated |

---

### `.env`

```
SERPAPI_API_KEY=your_key_here
LLM_MODEL=gpt-oss:120b-cloud
PORT=3456
```

| Variable | Purpose |
|----------|---------|
| `SERPAPI_API_KEY` | SerpAPI key (for web search, if used) |
| `LLM_MODEL` | Shared LLM model name (must match FastAPI backend `.env`) |
| `PORT` | Server listen port (default 3456) |

---

### `.env.example`

Template with placeholder values for `SERPAPI_API_KEY`, `LLM_MODEL`, and `PORT`. Copy to `.env` for local setup.

---

### `Dockerfile`

- **Base image:** `node:18-alpine`
- Installs production dependencies only (`npm ci --only=production`)
- Builds TypeScript (`npm run build`)
- Exposes port **3456**
- Starts with `npm start`

---

## Architecture Flow

```
┌──────────────────────────────────────────────────────────┐
│  LLM Agents (Claude Desktop, Cursor, etc.)               │
│  ─────────────────────────────────────────────────────── │
│  Call tool: classify_patent(patent_text)                 │
└────────────────────────┬─────────────────────────────────┘
                         │ MCP Protocol (JSON-RPC over HTTP)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  MCP_patent_classification (THIS PROJECT)                │
│  ─────────────────────────────────────────────────────── │
│  src/server/index.ts                                     │
│  • Express HTTP server on port 3456                      │
│  • POST /mcp → MCP endpoint (for LLM agents)             │
│  • POST /cpc/classify → REST endpoint (for Streamlit)    │
│  • Tool: classify_patent(text) → fetch to FastAPI        │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP POST /classify
                         ▼
┌──────────────────────────────────────────────────────────┐
│  patent_cpc_fastapi (Python FastAPI backend)             │
│  ─────────────────────────────────────────────────────── │
│  app/main.py                                             │
│  • POST /classify → CPCClassifier.classify()             │
│  • Runs full Phase 1–8 pipeline                          │
│  • Returns JSON with layers, facets, report              │
└──────────────────────────────────────────────────────────┘
```

### Streamlit Client Path

```
Streamlit UI (cpc_client/streamlit_app.py)
    │
    │ POST /cpc/classify (REST, bypassing MCP)
    ▼
MCP Server (index.ts) → proxies to FastAPI
```

---

## Running the Server

```bash
# Development (with hot-reload)
npm run dev

# Production (after build)
npm run build
npm start

# Debug stub (standalone, no FastAPI needed)
npx tsx src/server/raw-server.ts

# Test client (requires server running)
npx tsx src/test-client.ts
```

---

## Log Files

These are residual runtime artifacts from previous runs, not source code:

| File | Content |
|------|---------|
| `server.log` | One line: startup on port 3002 (old config) |
| `raw_server.log` | Three lines: startup, one request, response |
| `fresh_server.log` | Timestamped startup on port 3003 |
| `server_console.log` | Residual console output |
| `server_out.log` | Residual output |

---

## Notes

- This project contains **no Python files** — all classification logic lives in the separate `patent_cpc_fastapi` project.
- The MCP server is a **thin proxy** — it adds MCP protocol support on top of the existing REST API without duplicating classification logic.
- Per-request `McpServer` instances prevent session state from leaking between concurrent users.
- The 10-minute fetch timeout is intentional — some patent classification runs can take 30–120 seconds with the full pipeline and LLM calls.
- The `raw-server.ts` stub is useful for quickly verifying MCP SDK transport works before integrating Express.
