import os
import sys

# Ensure app/ is on path so absolute imports like 'cpc_classification' resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from cpc_classification.search_cpc import CPCClassifier
from cpc_classification.knowledge_graph import CPCKnowledgeGraph

# Load .env from the project root (parent of app/)
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    load_dotenv()  # Fallback to current directory
    print("⚠️  No .env file found, using system environment variables")

app = FastAPI(title="Patent CPC Classification API")

# Read model from .env — shared with MCP server (issue 3)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")

# ── Load Knowledge Graph (Optional) ───────────────────────────────
# Set SKIP_KG=1 to disable knowledge graph for faster startup
# Set KG_SECTIONS=G,H to build only specific sections
graph_cache_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cpc_classification", "resources"
)
cpc_cache_dir = os.path.join(graph_cache_dir, "cpc_scheme_2026")

knowledge_graph = None

# Debug: Show environment variables
skip_kg_rebuild = os.getenv("SKIP_KG_REBUILD", "NOT SET")
skip_kg = os.getenv("SKIP_KG", "NOT SET")
print(f"DEBUG: SKIP_KG_REBUILD = '{skip_kg_rebuild}'")
print(f"DEBUG: SKIP_KG = '{skip_kg}'")

if os.getenv("SKIP_KG") == "1":
    print("⚠️  Knowledge graph disabled (SKIP_KG=1)")
    print("   Using LLM-only classification (no semantic search)")
else:
    knowledge_graph = CPCKnowledgeGraph(cache_dir=graph_cache_dir)

    if knowledge_graph.load():
        print("✅ Knowledge graph loaded from cache")
        # Check if source files changed (unless disabled)
        if os.getenv("SKIP_KG_REBUILD") != "1" and knowledge_graph.check_source_changed(
            cpc_cache_dir
        ):
            print("📝 CPC source files changed, rebuilding graph...")
            print("   Tip: Set SKIP_KG_REBUILD=1 to skip automatic rebuilds")
            knowledge_graph.build_from_cache(cpc_cache_dir)
            knowledge_graph.save()
            print("✅ Knowledge graph rebuilt and saved")
        elif os.getenv("SKIP_KG_REBUILD") == "1":
            print("   ⏩ Skipping rebuild check (SKIP_KG_REBUILD=1)")
    else:
        # Check if we have JSON cache files or XML files
        json_files = [
            f
            for f in os.listdir(cpc_cache_dir)
            if f.startswith("cpc-cache-") and f.endswith(".json")
        ]
        xml_files = [
            f
            for f in os.listdir(cpc_cache_dir)
            if f.startswith("cpc-scheme-") and f.endswith(".xml")
        ]

        kg_sections = os.getenv("KG_SECTIONS", "")
        if kg_sections:
            sections = [s.strip() for s in kg_sections.split(",") if s.strip()]
            print(f"   Building sections: {sections} (fast mode)")
        else:
            sections = None
            print("   Building all sections A-Z (full mode)")

        if json_files:
            print("🔨 Building knowledge graph from CPC JSON cache files...")
            print("   This may take 10-15 minutes on first run...")
            print("   Tip: Copy pre-built cache files from Colab to skip this step")
            print("   Or: Set KG_SECTIONS=G,H in .env for faster build")
            print("   Or: Set SKIP_KG=1 to disable knowledge graph entirely")
            knowledge_graph.build_from_cache(cpc_cache_dir, sections=sections)
        elif xml_files:
            print("🔨 Building knowledge graph from CPC XML scheme files...")
            print("   This may take 30-60 minutes depending on sections...")
            print(
                "   Tip: Set KG_SECTIONS=G,H in .env for faster build (computing/telecom only)"
            )
            print("   Or: Set SKIP_KG=1 to disable knowledge graph entirely")
            knowledge_graph.build_from_xml(cpc_cache_dir, sections=sections)
        else:
            print("❌ No CPC data files found in", cpc_cache_dir)
            print("   Expected: cpc-cache-*.json or cpc-scheme-*.xml files")
            print("   Continuing without knowledge graph...")
            knowledge_graph = None

        if knowledge_graph:
            knowledge_graph.save()
            print("✅ Knowledge graph built and saved")

# Instantiate classifier with or without knowledge graph
classifier = CPCClassifier(model_name=LLM_MODEL, knowledge_graph=knowledge_graph)


# ── Input schema with validation (issue 9) ──────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str
    claims: str | None = None  # Optional claims field

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 100:
            raise ValueError(
                f"Patent text too short ({len(stripped)} chars). "
                "Provide at least 100 characters."
            )
        if len(stripped) > 50_000:
            raise ValueError(
                f"Patent text too long ({len(stripped)} chars). "
                "Maximum is 50,000 characters."
            )
        return stripped


# ── Root route ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Patent CPC Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "classify": {
                "path": "/classify",
                "method": "POST",
                "description": "Classify patent text into CPC codes",
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Check system health (Ollama, KG, etc.)",
            },
        },
        "status": "running",
    }


# ── Health check endpoint ──────────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Check system health: Ollama LLM, Knowledge Graph, and model availability.
    """
    import urllib.request
    import urllib.error
    import json

    # Check Ollama
    ollama_status = {"available": False, "models": [], "error": None}
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                models = body.get("models", [])
                ollama_status = {
                    "available": True,
                    "models": [m.get("name", "") for m in models],
                    "error": None,
                }
    except Exception as e:
        ollama_status["error"] = str(e)

    # Check if configured model is available
    model_available = LLM_MODEL in ollama_status.get("models", [])

    # Check Knowledge Graph
    kg_status = {
        "loaded": knowledge_graph is not None,
        "embeddings": (
            knowledge_graph.embeddings is not None if knowledge_graph else False
        ),
        "nodes": (knowledge_graph.graph.number_of_nodes() if knowledge_graph else 0),
    }

    overall_healthy = ollama_status["available"] and model_available

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "ollama": ollama_status,
        "configured_model": LLM_MODEL,
        "model_available": model_available,
        "knowledge_graph": kg_status,
        "recommendations": []
        if overall_healthy
        else [
            "Start Ollama: ollama serve",
            f"Pull model: ollama pull {LLM_MODEL}",
            "Or set LLM_MODEL in .env to an available model",
            "Or set SKIP_KG=1 if you don't need knowledge graph",
        ],
    }


# ── Main endpoint (issues 1, 2, 8) ──────────────────────────────────────────
@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify patent text into CPC codes.

    Supports two modes:
    1. Normal mode: text = patent description (uses LLM for Phase 1)
    2. Manual Phase 1 mode: text starts with "MANUAL_PHASE1:" followed by JSON
       (bypasses LLM, uses provided Phase 1 data directly)
    """
    try:
        # Check for manual Phase 1 mode
        if req.text.startswith("MANUAL_PHASE1:"):
            import json

            manual_json = req.text[len("MANUAL_PHASE1:") :]
            manual_phase1 = json.loads(manual_json)

            # Run pipeline starting from Phase 2 with manual Phase 1
            result = classifier.classify_from_phase1(manual_phase1)
        else:
            # Combine description and claims if claims are provided
            full_text = req.text
            if req.claims:
                full_text = f"DESCRIPTION:\n{req.text}\n\nCLAIMS:\n{req.claims}"

            result = classifier.classify(full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
