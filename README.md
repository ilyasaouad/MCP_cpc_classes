# MPC Patent Classification System

Patent CPC (Cooperative Patent Classification) classification pipeline using a three-phase approach with LLM extraction, XML-based subgroup expansion, and TF-IDF scoring.

## System Architecture

```
+-----------------+     +------------------+     +-----------------+
|   Streamlit     |────>|  FastAPI         |────>|  Ollama/Local   |
|   Client        |     |  Backend         |     |  LLM            |
|   (Port 8502)   |     |  (Port 8000)     |     |                 |
+-----------------+     +------------------+     +-----------------+
         |                       |
         |              +--------v--------+
         |              |  MCP Server     |
         +─────────────>|  (Port 3456)    |
                        +-----------------+
```

## Prerequisites

- **Node.js** 18+ (for MCP server)
- **Python** 3.11+ (for FastAPI + Streamlit)
- **Ollama** installed locally with your preferred model (e.g., `llama3.2`)
- **Git**
- **~2GB disk space** for CPC XML scheme files (not included in repo)

## Project Structure

| Service | Folder | Port | Tech Stack |
|---------|--------|------|------------|
| **MCP Server** | `MCP_patent_classification/` | 3456 | TypeScript, Express, MCP SDK |
| **FastAPI Backend** | `patent_cpc_fastapi/` | 8000 | Python, FastAPI, Ollama |
| **Streamlit Client** | `cpc_client/` | 8502 | Python, Streamlit |

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/ilyasaouad/MPC_patent_classification.git
cd MPC_patent_classification
```

### Step 2: Download CPC XML Scheme Files

**These files are NOT included in the repository** (239MB, exceeds GitHub limits).

1. Download from EPO (European Patent Office):
   - Go to: https://www.epo.org/searching-for-patents/helpful-resources/cpc.html
   - Download latest CPC scheme XML files

2. Place files in:
   ```
   patent_cpc_fastapi/app/cpc_classification/resources/cpc_scheme_2026/
   ```

3. On first run, the system will automatically parse XML files into cached JSON.

### Step 3: Configure Environment

**Copy example environment files:**

```bash
# MCP Server
cp MCP_patent_classification/.env.example MCP_patent_classification/.env

# FastAPI Backend
cp patent_cpc_fastapi/.env.example patent_cpc_fastapi/.env

# Streamlit Client (optional - uses defaults)
cp cpc_client/.env.example cpc_client/.env
```

**Edit each `.env` file with your values:**

#### MCP_patent_classification/.env
```env
SERPAPI_API_KEY=your_key_here
LLM_MODEL=llama3.2
PORT=3456
```

#### patent_cpc_fastapi/.env
```env
OPENAI_API_KEY=your_api_key_here  # Optional fallback
LLM_MODEL=llama3.2
```

#### cpc_client/.env (optional)
```env
API_URL=http://localhost:8000
MCP_URL=http://localhost:3456
```

### Step 4: Install Dependencies

**MCP Server (TypeScript):**
```bash
cd MCP_patent_classification
npm install
```

**FastAPI Backend (Python):**
```bash
cd patent_cpc_fastapi
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Streamlit Client (Python):**
```bash
cd cpc_client
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Running the System

### Terminal 1: Start Ollama
```bash
ollama serve
# Ensure your model is pulled:
# ollama pull llama3.2
```

### Terminal 2: Start MCP Server
```bash
cd MCP_patent_classification
npm run dev
# Server runs on http://localhost:3456
```

### Terminal 3: Start FastAPI Backend
```bash
cd patent_cpc_fastapi
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Terminal 4: Start Streamlit Client
```bash
cd cpc_client
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8502
# Open: http://localhost:8502
```

## API Endpoints

### POST /classify
Classify patent text into CPC codes.

**Request:**
```json
{
  "text": "Patent description text here...",
  "claims": "Optional claims text here..."
}
```

**Response:**
```json
{
  "phase1": {
    "technical_object": "...",
    "system_context": "...",
    "core_function": "...",
    "classification_strategy": "system-first",
    "cpc_classes": ["E21B", "F16J"],
    "essential_terms": [...],
    "negative_signals": [...]
  },
  "phase2": {
    "codes": [...],
    "reasoning": "..."
  },
  "phase3": [
    {"symbol": "E21B33/04", "title": "...", "score": 0.95}
  ],
  "phase4": {
    "best_code": {"symbol": "...", "confidence": "high"},
    "re_ranked": [...]
  }
}
```

## Classification Strategy

The system uses a **3-way strategy**:

- **system-first**: Domain-specific equipment (e.g., wellhead -> E21B)
- **function-first**: Generic components (e.g., seal -> F16J)
- **hybrid**: Both domain and function are equally novel

## Key Features

- **Claims-aware**: Extracts terms separately from description and independent claims (claims get 2x weight)
- **Negative signals**: Automatically generates terms/domains to exclude
- **Post-ranking LLM**: Re-ranks top 7 codes and selects the best one
- **No hardcoded blacklists**: Adapts to any domain (oil/gas, biotech, software, etc.)
- **Local LLM support**: Works with Ollama (no API costs)

## Docker Deployment (Optional)

If you prefer Docker, a `docker-compose.yml` is provided:

```bash
docker-compose up --build
```

This starts all 3 services with a single command.

**Note:** You still need to download CPC XML files manually before running Docker.

## Troubleshooting

### Issue: "Could not reach Ollama"
**Fix:** Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`)

### Issue: "Phase 1 returns empty"
**Fix:** The patent text might be too short (min 100 chars) or the LLM response is truncated. Increase `max_tokens` in `extracting_cpc.py`.

### Issue: "No CPC XML files found"
**Fix:** Download CPC scheme files from EPO and place in `resources/cpc_scheme_2026/`. The system will auto-parse them on first run.

### Issue: "Streamlit port already in use"
**Fix:** Change port: `streamlit run streamlit_app.py --server.port 8503`

## License

ISC
# MCP_cpc_classes
