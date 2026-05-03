import os
import sys

# Ensure app/ is on path so absolute imports like 'cpc_classification' resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from cpc_classification.search_cpc import CPCClassifier

load_dotenv()

app = FastAPI(title="Patent CPC Classification API")

# Read model from .env — shared with MCP server (issue 3)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")

# Instantiate once at startup (classifier is stateless per request)
classifier = CPCClassifier(model_name=LLM_MODEL)


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

    Pipeline:
        Phase 1 — LLM extraction of terms + candidate classes
        Phase 2 — EPO Linked Open Data enrichment (with fallback)
        Phase 3 — Embedding cosine scoring + ranked cpc[] output

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
