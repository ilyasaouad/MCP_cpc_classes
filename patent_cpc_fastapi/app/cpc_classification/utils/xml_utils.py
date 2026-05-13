"""xml_utils.py — XML directory resolution for CPC parser."""

import os


def resolve_xml_dir() -> str:
    """Resolve the CPC XML directory relative to the cpc_classification package."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "resources", "cpc_scheme_2026")
