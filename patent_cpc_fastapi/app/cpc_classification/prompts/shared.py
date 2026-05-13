"""
shared.py — Shared utilities for all prompt modules.

Contents:
  - label_claims()         Parse claims text and tag [INDEPENDENT]/[DEPENDENT]
  - detect_sections()      Split patent text into sections (abstract, claims, etc.)
  - UNIFIED_IMPORTANCE_RUBRIC  Term weighting rules used across all phases
"""

import re


# ---------------------------------------------------------------------------
# CLAIM LABELING
# ---------------------------------------------------------------------------


def label_claims(raw_claims_text: str) -> str:
    """
    Parse raw claims text and prefix each claim with [INDEPENDENT] or
    [DEPENDENT: ref claim N].
    """
    dependency_pattern = re.compile(
        r"\baccording to claim[s]?\s+([\d, ]+(?:or\s+\d+)?)"
        r"|\bof claim[s]?\s+([\d, ]+(?:or\s+\d+)?)",
        re.IGNORECASE,
    )

    lines = raw_claims_text.strip().splitlines()
    labeled_lines = []
    current_claim_lines = []
    current_claim_num = None

    def flush(claim_lines, claim_num):
        if not claim_lines:
            return
        block = "\n".join(claim_lines)
        match = dependency_pattern.search(block)
        if match:
            ref = (match.group(1) or match.group(2)).strip()
            labeled_lines.append(f"[DEPENDENT: ref claim {ref}]")
        else:
            labeled_lines.append("[INDEPENDENT]")
        labeled_lines.extend(claim_lines)
        labeled_lines.append("")

    claim_start = re.compile(r"^\s*(\d+)\.\s")

    for line in lines:
        m = claim_start.match(line)
        if m:
            flush(current_claim_lines, current_claim_num)
            current_claim_num = int(m.group(1))
            current_claim_lines = [line]
        else:
            current_claim_lines.append(line)

    flush(current_claim_lines, current_claim_num)
    return "\n".join(labeled_lines)


# ---------------------------------------------------------------------------
# SECTION DETECTION
# ---------------------------------------------------------------------------


def detect_sections(text: str) -> dict:
    """
    Detect patent sections and return a mapping with full text of each section.
    """
    section_markers = {
        "abstract": r"(?i)^\s*(abstract|summary of the invention|brief summary)",
        "background": r"(?i)^\s*(background|prior art|related art|field of the invention)",
        "summary": r"(?i)^\s*(summary|summary of invention)",
        "claims": r"(?i)^\s*(claims?|we claim|what is claimed)",
        "detailed": r"(?i)^\s*(detailed description|description of.*embodiment|detailed description of.*invention)",
        "drawings": r"(?i)^\s*(brief description of (the )?drawings?|description of (the )?figures?)",
    }

    sections = {"full_text": text}
    lines = text.splitlines()

    for section_name, pattern in section_markers.items():
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                sections[section_name] = "\n".join(lines[i:])
                break

    return sections


# ---------------------------------------------------------------------------
# UNIFIED IMPORTANCE & SECTION WEIGHT REFERENCE
# ---------------------------------------------------------------------------

UNIFIED_IMPORTANCE_RUBRIC = """
=== TERM IMPORTANCE CALIBRATION (unified — apply once) ===

Step A — Assign a BASE score using the content-type scale:
  10  Term is the core inventive feature AND appears in an independent claim
   9  Term appears in an independent claim and is a key structural/functional element
   8  Term is in the summary or detailed description AND directly enables the solution
   7  Term is in the description and important context for the solution
   5  Supporting or enabling term
   3  Generic domain vocabulary (retain only if no better term available)
   1  Too generic to contribute — exclude

Step B — Multiply base by the section weight where the term was found:
  Claims (independent)   × 1.2  (cap at 10)
  Claims (dependent)     × 1.0
  Summary of Invention   × 1.0
  Detailed Description   × 0.9
  Drawing Descriptions   × 0.95 (high precision — figure captions name exact components)
  Abstract               × 0.6
  Background / Prior Art × 0.2  (max effective score = 3 unless reinforced in claims)

Step C — Apply figure trust adjustment (drawing descriptions only):
  Figure labeled "prior art" or "conventional"  → treat as background (× 0.2)
  Figure labeled "according to the invention"   → treat as detailed description (× 0.95)
  Figure with no label                           → default to × 0.7

Step D — Round to nearest integer, cap at 10.

NEVER assign score 10 to background-only terms.
NEVER assign below 7 to independent-claim terms unless purely generic.
"""
