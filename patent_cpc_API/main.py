"""
main.py — Patent CPC API entry point.

Startup sequence:
  1. Load .env (API keys, model name, paths)
  2. Initialise the Knowledge Graph singleton
  3. Instantiate CPCPipelineEngine with all dependencies
  4. Register the API router
  5. Serve via uvicorn

Run:
    uvicorn patent_cpc_API.main:app --reload --port 8001
"""

import logging
import os
import sys

# ── sys.path: add patent_cpc_fastapi/app/ so phase modules can do
#             `from cpc_classification.X import Y` (same as old project) ──────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_FASTAPI_APP = os.path.normpath(
    os.path.join(_ROOT, "..", "patent_cpc_fastapi", "app")
)
if _FASTAPI_APP not in sys.path:
    sys.path.insert(0, _FASTAPI_APP)

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env — check patent_cpc_API/ first, then parent MCP_cpc_classes/
_ENV = os.path.join(_ROOT, ".env")
if not os.path.exists(_ENV):
    _ENV = os.path.join(os.path.dirname(_ROOT), "patent_cpc_fastapi", ".env")
load_dotenv(_ENV if os.path.exists(_ENV) else None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Patent CPC Classification API",
    description=(
        "13-phase CPC patent classification pipeline. "
        "Combines LLM signal extraction with BM25 + embedding retrieval, "
        "rule-based constraint layers, and hypothesis resolution."
    ),
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    # ── Knowledge Graph ────────────────────────────────────────────────────
    from .core.knowledge_graph import KnowledgeGraphSingleton

    kg = None
    if os.getenv("SKIP_KG") != "1":
        kg_cache = os.getenv(
            "KG_CACHE_DIR",
            os.path.join(_ROOT, "..", "patent_cpc_fastapi", "app",
                         "cpc_classification", "resources"),
        )
        kg = KnowledgeGraphSingleton.get(cache_dir=kg_cache)
        if kg:
            logger.info("KG loaded: %d nodes", len(kg.graph.nodes))
        else:
            logger.warning("KG not available — running LLM-only mode")
    else:
        logger.info("SKIP_KG=1 — knowledge graph disabled")

    # ── XML Parser (optional) ──────────────────────────────────────────────
    xml_parser = None
    try:
        from cpc_classification.cpc_xml_parser import CPCXMLParser
        xml_dir = os.getenv(
            "XML_DIR",
            os.path.normpath(os.path.join(_ROOT, "..", "patent_cpc_fastapi", "app",
                             "cpc_classification", "resources", "cpc_scheme_2026")),
        )
        if os.path.isdir(xml_dir):
            xml_parser = CPCXMLParser(xml_dir)
            logger.info("XML parser ready: %s", xml_dir)
    except Exception as exc:
        logger.warning("XML parser unavailable: %s", exc)

    # ── LLM Client ────────────────────────────────────────────────────────
    llm = None
    try:
        from search_core.ollama_client import OllamaClient
        llm = OllamaClient()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s — LLM phases will fail gracefully", exc)

    # ── Pipeline Engine ───────────────────────────────────────────────────
    from .core.engine import CPCPipelineEngine
    from .api import set_engine, set_kg

    engine = CPCPipelineEngine(kg=kg, xml_parser=xml_parser, llm_client=llm)
    set_engine(engine)
    set_kg(kg)
    logger.info("CPCPipelineEngine registered")


# ── Router ─────────────────────────────────────────────────────────────────
from .api.router import router

app.include_router(router, prefix="/api/v2")
