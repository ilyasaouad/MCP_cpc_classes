"""
phase_2a_v2 — Phase 2A v2 Multi-Source Fusion Router.

Replaces legacy keyword-based CPC family routing with:
  - Semantic embedding similarity (45%)
  - CPC graph structure awareness (35%)
  - Deterministic anchor-to-CPC mapping (20%)

No LLM dependency. No static domain lists. No legacy fallback.
"""

from .phase_2a_v2_router import Phase2AV2Router
