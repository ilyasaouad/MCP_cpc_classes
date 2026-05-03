import requests
from typing import Dict, Any


class CPCRestClient:
    """
    HTTP client for the MCP patent classification server.

    Calls the plain REST endpoint POST /cpc/classify on the MCP server
    (which in turn proxies to the FastAPI backend on port 8000).

    Note: Despite the historical name 'MCPClient', this client does NOT
    speak the MCP JSON-RPC protocol — it uses the direct REST bridge
    exposed by the Node.js MCP server for Streamlit compatibility.
    """

    def __init__(self, base_url: str = "http://localhost:3456"):
        self.base_url = base_url.rstrip("/")

    def classify_cpc(self, text: str) -> Dict[str, Any]:
        """
        Send patent text to the MCP server and get CPC classification.

        Returns a dict with keys: phase1, phase2, phase3, cpc[]
        On connection failure, returns {"error": "...", "cpc": []}.
        """
        url = f"{self.base_url}/cpc/classify"
        payload = {"text": text}

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {
                "error": (
                    f"Cannot reach MCP server at {self.base_url}. "
                    "Is the Node.js MCP server running? (npm run dev)"
                ),
                "cpc": [],
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out after 120 seconds. The LLM may still be processing.",
                "cpc": [],
            }
        except requests.RequestException as e:
            return {"error": str(e), "cpc": []}

    def classify_cpc_with_claims(self, text: str, claims: str) -> Dict[str, Any]:
        """
        Send patent text with separate claims to the MCP server.

        The claims field allows the Phase 1 extractor to prioritize
        terms that appear in claims when extracting essential terms.
        """
        url = f"{self.base_url}/cpc/classify"
        payload = {"text": text, "claims": claims}

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {
                "error": (
                    f"Cannot reach MCP server at {self.base_url}. "
                    "Is the Node.js MCP server running? (npm run dev)"
                ),
                "cpc": [],
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out after 120 seconds. The LLM may still be processing.",
                "cpc": [],
            }
        except requests.RequestException as e:
            return {"error": str(e), "cpc": []}


# Backward-compatible alias — kept so existing imports don't break
MCPClient = CPCRestClient
