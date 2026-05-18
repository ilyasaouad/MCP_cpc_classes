"""
run.py — Start patent_cpc_API from inside its own folder.

Usage (from inside patent_cpc_API/):
    .\.venv\Scripts\python.exe run.py

This script adds the parent directory to sys.path so Python can find
'patent_cpc_API' as a package, then launches uvicorn programmatically.
"""

import os
import sys

# Make the parent folder (MCP_cpc_classes/) importable so both
# 'patent_cpc_API' and 'patent_cpc_fastapi' resolve as packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "patent_cpc_API.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[_HERE],
    )
