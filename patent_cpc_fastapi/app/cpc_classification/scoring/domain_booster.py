"""domain_booster.py — Domain-specific term mapping and synonym expansion."""

from typing import List, Set


# Synonym mapping for CPC technical terms
CPC_SYNONYMS = {
    "venting": ["deaeration", "degassing", "air removal", "bleeding", "vent"],
    "bleeding": ["deaeration", "degassing", "venting", "air removal"],
    "deaeration": ["venting", "degassing", "air removal", "bleeding"],
    "degassing": ["deaeration", "venting", "air removal"],
    "air removal": ["deaeration", "venting", "degassing"],
    "cooling": [
        "temperature control",
        "heat removal",
        "thermal management",
        "coolant",
        "radiator",
    ],
    "thermal management": ["cooling", "heat removal", "temperature control"],
    "sealing": ["gasketing", "packing", "jointing", "seal"],
    "seal": ["sealing", "gasket", "packing"],
    "valve": ["tap", "cock", "vent", "shut-off"],
    "battery": ["accumulator", "cell", "electrochemical storage"],
    "wellhead": [
        "well head",
        "blowout preventer",
        "christmas tree",
        "wellhead assembly",
    ],
    "drilling": ["boring", "earth drilling", "well drilling"],
    "explosive": ["charge", "detonation", "blast", "explosion"],
    "cutter": ["cutting tool", "perforator", "radial cutter", "pipe cutter"],
    "completion": ["wellbore completion", "tubing completion"],
    "device": ["apparatus", "equipment", "unit"],
    "apparatus": ["device", "equipment", "unit"],
}


def get_synonyms(word: str) -> List[str]:
    """Get CPC synonyms for a word."""
    return CPC_SYNONYMS.get(word.lower(), [])


def get_expanded_terms(term: str) -> Set[str]:
    """Get all variants of a term including synonyms."""
    terms = {term.lower()}
    words = term.lower().split()
    for word in words:
        syns = get_synonyms(word)
        for syn in syns:
            for w in words:
                variant = term.lower().replace(w, syn)
                terms.add(variant)
            terms.add(syn)
    return terms
