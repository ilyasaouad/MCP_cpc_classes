import os
import sys

# Ensure app/ is on path so absolute imports like 'cpc_classification' resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from cpc_classification.search_cpc import CPCClassifier
from cpc_classification.knowledge_graph import CPCKnowledgeGraph

load_dotenv()

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

if os.getenv("SKIP_KG") == "1":
    print("⚠️  Knowledge graph disabled (SKIP_KG=1)")
    print("   Using LLM-only classification (no semantic search)")
else:
    knowledge_graph = CPCKnowledgeGraph(cache_dir=graph_cache_dir)

    if knowledge_graph.load():
        print("✅ Knowledge graph loaded from cache")
        # Check if source files changed
        if knowledge_graph.check_source_changed(cpc_cache_dir):
            print("📝 CPC source files changed, rebuilding graph...")
            knowledge_graph.build_from_cache(cpc_cache_dir)
            knowledge_graph.save()
            print("✅ Knowledge graph rebuilt and saved")
    else:
        print("🔨 Building knowledge graph from CPC cache files...")
        print("   This may take 10-15 minutes on first run...")
        print("   Tip: Copy pre-built cache files from Colab to skip this step")
        print("   Or: Set KG_SECTIONS=G,H in .env for faster build")
        print("   Or: Set SKIP_KG=1 to disable knowledge graph entirely")

        kg_sections = os.getenv("KG_SECTIONS", "")
        if kg_sections:
            sections = [s.strip() for s in kg_sections.split(",") if s.strip()]
            print(f"   Building sections: {sections} (fast mode)")
        else:
            sections = None
            print("   Building all sections A-Z (full mode)")

        knowledge_graph.build_from_cache(cpc_cache_dir, sections=sections)
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


# ── Main endpoint (issues 1, 2, 8) ──────────────────────────────────────────
@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify patent text into CPC codes.

    Improved Pipeline (addresses weaknesses W1-W8):
        Phase 1  — LLM extraction with section-aware term weighting,
                   claim type analysis (method vs apparatus), and
                   soft class hypotheses with confidence scores
        Phase 1b — Probabilistic domain inference (replaces hardcoded injection)
        Phase 2  — XML expansion + improved TF-IDF scoring with:
                   - Section-aware term weights
                   - Probabilistic domain boosting (default 1.2x for unknown)
                   - Term-density guard for specificity bonus
                   - Score margin calculation for confidence level
        Phase 3  — Ranking by composite score
        Phase 5  — Multi-pass validation (one candidate per prompt):
                   - Function alignment check
                   - Context alignment check
                   - Visual bias detection
                   - Method vs apparatus verification
                   - Score margin awareness
        Phase 6  — Per-claim reconciliation (removes rejected codes,
                   replaces with validated alternatives)
        Phase 7  — Final consistency check (coherence of selected codes)

    Output includes:
        - premier: Single best validated CPC class
        - per_claim: Claim-by-claim classification mapping
        - phase5: Validation results with filtered candidates and rejection reasons
        - phase7: Consistency assessment

    Optional claims field allows prioritizing claim terms in Phase 1.
    """
    try:
        # Combine description and claims if claims are provided
        full_text = req.text
        if req.claims:
            full_text = f"DESCRIPTION:\n{req.text}\n\nCLAIMS:\n{req.claims}"

        result = classifier.classify(full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
