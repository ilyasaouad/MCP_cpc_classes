"""
cpc_role_labeling.py - CPC Role Labeling & 3-Layer Explanation Model

PURPOSE:
Transform flat CPC classification into a structured reasoning graph.

CORE PRINCIPLE:
CPC classification is not "correct vs incorrect" — it is "what role does
each CPC code play?"

Every CPC output must be tagged with:
- CORE: Highest technical contribution, strongest novelty, tightest match
- SUPPORT: Supporting infrastructure, framework, enabling technology
- CONTEXT: System environment, background domain, general usage
- LEGAL_COVERAGE: Broader parent classes, overlapping domains, fallback safety nets

DESIGN:
- Layer 1 (CORE): 1-2 CPCs representing the invention's essence
- Layer 2 (SUPPORT/CONTEXT): 1-3 CPCs explaining the system environment
- Layer 3 (LEGAL_COVERAGE): Optional broader classes for legal defensibility
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# System prompt for Phase 8.5 LLM-based Technical Reasoning
EXAMINER_REPORT_SYSTEM_PROMPT = """You are a Patent Examiner. Based on the selected CPC codes and the extracted technical object, write a 3-4 sentence "Technical Reasoning Graph" summary. Explain how the CORE codes interact with the SUPPORT tools within the given CONTEXT. Focus on the orchestration of components rather than just listing features."""


EXAMINER_REPORT_USER_PROMPT = """TECHNICAL OBJECT:
{technical_object}

CORE FUNCTION:
{core_function}

SYSTEM CONTEXT:
{system_context}

SELECTED CPC CODES:

1. PRIMARY CLASSIFICATION (CORE):
{core_codes}

2. SUPPORTING & ENABLING TECHNOLOGY:
{support_codes}

3. APPLICATION CONTEXT:
{context_codes}

4. LEGAL COVERAGE:
{coverage_codes}

Please provide:
1. A 3-4 sentence Technical Reasoning Graph summary explaining how CORE interacts with SUPPORT within CONTEXT
2. A one-sentence reasoning for each CPC code, linking it specifically to the patent description

Format your response as JSON:
{{
    "executive_summary": "3-4 sentence summary of technical reasoning",
    "code_reasoning": [
        {{"symbol": "G10L15/22", "reasoning": "Specific reasoning linking code to patent description"}},
        ...
    ]
}}"""


# Role definitions with selection criteria
ROLE_DEFINITIONS = {
    "CORE": {
        "description": "Highest technical contribution, strongest novelty, tightest match",
        "max_count": 2,
        "criteria": [
            "Matches primary contribution type (A or B)",
            "Highest score from Phase 3.6",
            "Directly implements the core function",
            "Represents the novel aspect of the invention",
        ],
    },
    "SUPPORT": {
        "description": "Supporting infrastructure, framework, enabling technology",
        "max_count": 2,
        "criteria": [
            "Matches secondary contribution type",
            "Enables the core function to work",
            "Provides necessary infrastructure",
            "Different technical domain from CORE",
        ],
    },
    "CONTEXT": {
        "description": "System environment, background domain, general usage",
        "max_count": 2,
        "criteria": [
            "Describes the system context from Phase 1",
            "General domain classification",
            "Broader than CORE but still relevant",
            "Explains WHERE the invention operates",
        ],
    },
    "LEGAL_COVERAGE": {
        "description": "Broader parent classes, overlapping domains, fallback safety nets",
        "max_count": 2,
        "criteria": [
            "Parent class of CORE or SUPPORT",
            "Overlapping domain that could be claimed",
            "Bureaucratic coverage for legal safety",
            "Intentionally redundant with CORE",
        ],
    },
}


class CPCRoleLabeling:
    """
    Assigns structural roles to CPC candidates producing a reasoning graph.
    """

    def __init__(self, llm=None, tcr_result: Optional[Dict[str, Any]] = None):
        self.rules_log: List[Dict[str, Any]] = []
        self.llm = llm
        self.tcr_result = tcr_result or {}

    def label_roles(
        self,
        candidates: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
        phase36_result: Dict[str, Any],
        tcr_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assign roles to CPC candidates and build 3-layer explanation graph.

        Args:
            candidates: CPC candidates from Phase 3
            phase1_data: Phase 1 output
            phase36_result: Phase 3.6 validation result
            tcr_result: Optional Technical Weight Analysis result
                        If provided, guides role assignment:
                        - FORCE_SOFTWARE_CORE: Prioritize G06F as CORE
                        - FORCE_DOMAIN_CORE: Prioritize domain codes as CORE
                        - HYBRID_INVENTION: Balanced role assignment

        Returns:
            Dict with layered CPC model and reasoning traces.
        """
        if tcr_result:
            self.tcr_result = tcr_result

        if not candidates:
            return self._empty_result()

        self.rules_log = []

        # Step 1: Identify CORE candidates (with TCR guidance)
        core = self._select_core(candidates, phase36_result)

        # Step 2: Identify SUPPORT candidates
        support = self._select_support(candidates, core, phase36_result)

        # Step 3: Identify CONTEXT candidates
        context = self._select_context(candidates, core, support, phase1_data)

        # Step 4: Identify LEGAL_COVERAGE candidates
        coverage = self._select_coverage(candidates, core, support, phase36_result)

        # Build structured output
        all_codes = core + support + context + coverage

        # Phase 8.5: Generate LLM-based Technical Reasoning (if LLM available)
        llm_reasoning = {}
        if self.llm:
            try:
                llm_reasoning = self._generate_llm_reasoning(
                    core, support, context, coverage, phase1_data
                )
            except Exception as e:
                logger.warning("Phase 8.5 LLM reasoning failed: %s", e)

        return {
            "layer1_core": core,
            "layer2_support": support,
            "layer2_context": context,
            "layer3_coverage": coverage,
            "all_codes": all_codes,
            "role_summary": self._build_summary(core, support, context, coverage),
            "detailed_report": self._build_detailed_report(
                core, support, context, coverage, phase1_data
            ),
            "reasoning_graph": self._build_reasoning_graph(
                core, support, context, coverage, phase1_data
            ),
            "phase85_executive_summary": llm_reasoning.get("executive_summary", ""),
            "phase85_code_reasoning": llm_reasoning.get("code_reasoning", []),
            "tcr_value": self.tcr_result.get("tcr"),
            "tcr_bias": self.tcr_result.get("tcr_bias"),
            "rules_log": self.rules_log,
        }

    def _select_core(
        self, candidates: List[Dict[str, Any]], phase36_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select 1-2 CORE CPCs."""
        core = []
        primary_type = phase36_result.get("phase36_primary_code", "")

        # Candidates with "primary" contribution match
        primary_candidates = [
            c for c in candidates if c.get("contribution_match") == "primary"
        ]

        # Sort by score
        primary_candidates = sorted(
            primary_candidates, key=lambda x: x.get("score", 0), reverse=True
        )

        # Take top 1-2
        for c in primary_candidates[: ROLE_DEFINITIONS["CORE"]["max_count"]]:
            core.append(self._enrich_with_role(c, "CORE"))
            self._log_rule(
                "CORE_SELECTED",
                c.get("symbol", ""),
                "Primary contribution match",
            )

        # If no primary matches, take highest overall (with TCR guidance)
        if not core and candidates:
            best = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[0]
            # For HYBRID_INVENTION or when no TCR guidance, just take best
            core.append(self._enrich_with_role(best, "CORE"))
            self._log_rule(
                "CORE_SELECTED", best.get("symbol", ""), "Highest overall score"
            )

        return core

    def _select_support(
        self,
        candidates: List[Dict[str, Any]],
        core: List[Dict[str, Any]],
        phase36_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Select 1-2 SUPPORT CPCs."""
        support = []
        core_symbols = {c["symbol"] for c in core}

        # First: Candidates with "secondary" contribution match
        secondary_candidates = [
            c
            for c in candidates
            if c.get("contribution_match") == "secondary"
            and c["symbol"] not in core_symbols
        ]

        secondary_candidates = sorted(
            secondary_candidates, key=lambda x: x.get("score", 0), reverse=True
        )

        for c in secondary_candidates[: ROLE_DEFINITIONS["SUPPORT"]["max_count"]]:
            support.append(self._enrich_with_role(c, "SUPPORT"))
            self._log_rule(
                "SUPPORT_SELECTED", c.get("symbol", ""), "Secondary contribution match"
            )

        # Second: Add AI/NLP related candidates if SUPPORT is empty
        # This ensures LLM, speech processing, NLP codes are included
        if len(support) < ROLE_DEFINITIONS["SUPPORT"]["max_count"]:
            ai_nlp_families = ["G06N", "G10L", "G06F40", "G06K"]
            for c in candidates:
                if c["symbol"] in core_symbols:
                    continue
                if any(c["symbol"].startswith(f) for f in ai_nlp_families):
                    if not any(s["symbol"] == c["symbol"] for s in support):
                        support.append(self._enrich_with_role(c, "SUPPORT"))
                        self._log_rule(
                            "SUPPORT_SELECTED",
                            c.get("symbol", ""),
                            "AI/NLP enabling technology",
                        )
                        if len(support) >= ROLE_DEFINITIONS["SUPPORT"]["max_count"]:
                            break

        return support

    def _select_context(
        self,
        candidates: List[Dict[str, Any]],
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Select 1-2 CONTEXT CPCs.

        CONTEXT = system environment, background domain, general usage.
        Only select candidates that are:
        1. In a DIFFERENT domain from CORE/SUPPORT (or broader version)
        2. Actually relevant to the invention's domain
        3. NOT just random CPCs that happen to match some word

        CRITICAL: For pure software patents, CONTEXT is often EMPTY.
        A01K (agriculture) is NOT context for LLM code generation.
        """
        context = []
        used_symbols = {c["symbol"] for c in core + support}

        # Get core/support family prefixes to avoid
        core_support_families = set()
        for c in core + support:
            sym = c.get("symbol", "")
            if sym:
                # Add 3-char and 4-char prefixes
                core_support_families.add(sym[:3])
                core_support_families.add(sym[:4])

        # Check system_context quality
        system_context = phase1_data.get("system_context", "").strip().lower()

        # If system_context is empty or too generic, skip CONTEXT selection
        if len(system_context) < 10:
            self._log_rule(
                "CONTEXT_SKIPPED",
                "",
                "system_context too short/generic - no reliable context domain",
            )
            return context

        # Get meaningful terms from system_context (at least 2 chars, not common words)
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "for",
            "in",
            "on",
            "to",
            "of",
            "and",
            "or",
            "with",
            "by",
        }
        context_terms = [
            t for t in system_context.split() if len(t) > 3 and t not in stop_words
        ]

        if len(context_terms) < 2:
            self._log_rule(
                "CONTEXT_SKIPPED",
                "",
                "system_context has no meaningful terms for matching",
            )
            return context

        # Only match if CPC title contains AT LEAST 2 context terms
        # (single-term matches are too noisy)
        for c in candidates:
            if c["symbol"] in used_symbols:
                continue

            title = c.get("title", "").lower()

            # Count how many context terms appear in the title
            matched_terms = [t for t in context_terms if t in title]

            if len(matched_terms) >= 2:
                context.append(self._enrich_with_role(c, "CONTEXT"))
                self._log_rule(
                    "CONTEXT_SELECTED",
                    c.get("symbol", ""),
                    f"Matches system context with terms: {matched_terms[:3]}",
                )
                if len(context) >= ROLE_DEFINITIONS["CONTEXT"]["max_count"]:
                    break

        # If no good matches, CONTEXT stays empty (better than wrong matches)
        if not context:
            self._log_rule(
                "CONTEXT_EMPTY",
                "",
                "No CPCs match system_context well - CONTEXT intentionally empty",
            )

        return context

    def _select_coverage(
        self,
        candidates: List[Dict[str, Any]],
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        phase36_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Select 1-2 LEGAL_COVERAGE CPCs (broader parent classes)."""
        coverage = []
        used_symbols = {c["symbol"] for c in core + support}

        # Find broader classes (shorter symbols = broader)
        for c in candidates:
            if c["symbol"] in used_symbols:
                continue

            symbol = c.get("symbol", "")
            # Broader classes have shorter symbol length
            if len(symbol) <= 7:  # e.g., "G06N3/", "G06F"
                coverage.append(self._enrich_with_role(c, "LEGAL_COVERAGE"))
                self._log_rule("COVERAGE_SELECTED", symbol, "Broader parent class")
                if len(coverage) >= ROLE_DEFINITIONS["LEGAL_COVERAGE"]["max_count"]:
                    break

        return coverage

    def _enrich_with_role(self, candidate: Dict[str, Any], role: str) -> Dict[str, Any]:
        """Add role metadata to candidate."""
        enriched = dict(candidate)
        enriched["role"] = role
        enriched["role_description"] = ROLE_DEFINITIONS[role]["description"]
        return enriched

    def _build_summary(
        self,
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        coverage: List[Dict[str, Any]],
    ) -> str:
        """Build human-readable summary."""
        parts = []
        if core:
            parts.append(f"CORE ({len(core)}): {', '.join(c['symbol'] for c in core)}")
        if support:
            parts.append(
                f"SUPPORT ({len(support)}): {', '.join(c['symbol'] for c in support)}"
            )
        if context:
            parts.append(
                f"CONTEXT ({len(context)}): {', '.join(c['symbol'] for c in context)}"
            )
        if coverage:
            parts.append(
                f"COVERAGE ({len(coverage)}): {', '.join(c['symbol'] for c in coverage)}"
            )
        return " | ".join(parts)

    def _build_detailed_report(
        self,
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        coverage: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> str:
        """
        Build detailed report in the format the user requested.

        Format:
        1. Primary Classification (CORE)
        2. Supporting & Enabling Technology (SUPPORT)
        3. Application Context (CONTEXT)
        4. Technical Reasoning Graph (Executive Summary)
        5. Legal Coverage (Search Safety Net)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CPC Classification & Technical Reasoning Report")
        lines.append("=" * 80)

        # Patent info
        patent_title = phase1_data.get(
            "technical_object", "Patent Classification Report"
        )
        lines.append(f"\nPatent Title: {patent_title}")

        # Section 1: Primary Classification
        lines.append("\n" + "-" * 80)
        lines.append("1. PRIMARY CLASSIFICATION (THE CORE INVENTION)")
        lines.append("-" * 80)
        lines.append(
            "These codes represent the technical essence and novelty of the disclosure."
        )
        lines.append("")
        lines.append(f"{'CPC Symbol':<15} {'Role':<10} {'Technical Definition':<45}")
        lines.append("-" * 70)

        for c in core:
            symbol = c.get("symbol", "")
            title = c.get("title", "")[:45]
            reasoning = c.get("reasoning", c.get("role_description", ""))[:50]
            lines.append(f"{symbol:<15} {'CORE':<10} {title}")

        # Add core reasoning
        if core:
            core_func = phase1_data.get("core_function", "")
            lines.append("")
            lines.append(f"Core Function Reasoning: {core_func[:100]}...")

        # Section 2: Supporting Technology
        if support:
            lines.append("\n" + "-" * 80)
            lines.append("2. SUPPORTING & ENABLING TECHNOLOGY")
            lines.append("-" * 80)
            lines.append(
                "These codes represent the underlying infrastructure and data structures."
            )
            lines.append("")
            lines.append(
                f"{'CPC Symbol':<15} {'Role':<12} {'Technical Definition':<45}"
            )
            lines.append("-" * 70)

            for c in support:
                symbol = c.get("symbol", "")
                title = c.get("title", "")[:45]
                lines.append(f"{symbol:<15} {'SUPPORT':<12} {title}")

        # Section 3: Application Context
        if context:
            lines.append("\n" + "-" * 80)
            lines.append("3. APPLICATION CONTEXT")
            lines.append("-" * 80)
            lines.append(
                "The industry domain and system environment where the invention is applied."
            )
            lines.append("")
            lines.append(
                f"{'CPC Symbol':<15} {'Role':<12} {'Technical Definition':<45}"
            )
            lines.append("-" * 70)

            for c in context:
                symbol = c.get("symbol", "")
                title = c.get("title", "")[:45]
                lines.append(f"{symbol:<15} {'CONTEXT':<12} {title}")

        # Section 4: Technical Reasoning Graph
        lines.append("\n" + "-" * 80)
        lines.append("4. TECHNICAL REASONING GRAPH (EXECUTIVE SUMMARY)")
        lines.append("-" * 80)

        core_symbols = [c.get("symbol", "") for c in core]
        support_symbols = [c.get("symbol", "") for c in support]
        context_symbols = [c.get("symbol", "") for c in context]

        if core:
            lines.append(f"\nSummary: The claimed invention focuses on")
            if core_symbols:
                lines.append(f"  - Primary technical contribution in {core_symbols[0]}")
            if support:
                lines.append(f"  - Supported by {support_symbols[0]} infrastructure")
            if context:
                lines.append(f"  - Applied within {context_symbols[0]} domain")

            core_func = phase1_data.get("core_function", "")
            if core_func:
                lines.append(f"\nCore Function: {core_func[:80]}...")

        # Section 5: Legal Coverage
        if coverage:
            lines.append("\n" + "-" * 80)
            lines.append("5. LEGAL COVERAGE (SEARCH SAFETY NET)")
            lines.append("-" * 80)
            lines.append(
                "Broader classifications recommended for comprehensive prior-art search."
            )
            lines.append("")

            for c in coverage:
                symbol = c.get("symbol", "")
                title = c.get("title", "")
                lines.append(f"  {symbol}: {title}")

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _build_reasoning_graph(
        self,
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        coverage: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build structured reasoning graph."""
        return {
            "core_reasoning": f"Represents the primary technical contribution: {phase1_data.get('core_function', '')}",
            "support_reasoning": "Enabling technology that makes the core function possible",
            "context_reasoning": f"System environment where the invention operates: {phase1_data.get('system_context', '')}",
            "coverage_reasoning": "Broader classes for legal defensibility and examiner alignment",
            "layer1_count": len(core),
            "layer2_count": len(support) + len(context),
            "layer3_count": len(coverage),
            "total_codes": len(core) + len(support) + len(context) + len(coverage),
        }

    def _log_rule(self, rule: str, symbol: str, reason: str) -> None:
        """Log a role assignment."""
        self.rules_log.append(
            {
                "rule": rule,
                "symbol": symbol,
                "reason": reason,
            }
        )

    def _generate_llm_reasoning(
        self,
        core: List[Dict[str, Any]],
        support: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        coverage: List[Dict[str, Any]],
        phase1_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Phase 8.5: Generate LLM-based Technical Reasoning Summary.

        Uses the LLM to generate examiner-style reasoning for each CPC code.
        """
        if not self.llm:
            return {}

        # Format CPC codes for prompt
        def format_codes(codes: List[Dict], layer_name: str) -> str:
            if not codes:
                return f"No {layer_name} codes selected"
            lines = []
            for c in codes:
                lines.append(f"  - {c.get('symbol', '')}: {c.get('title', 'N/A')}")
            return "\n".join(lines)

        core_str = format_codes(core, "CORE")
        support_str = format_codes(support, "SUPPORT")
        context_str = format_codes(context, "CONTEXT")
        coverage_str = format_codes(coverage, "COVERAGE")

        user_prompt = EXAMINER_REPORT_USER_PROMPT.format(
            technical_object=phase1_data.get("technical_object", "N/A"),
            core_function=phase1_data.get("core_function", "N/A"),
            system_context=phase1_data.get("system_context", "N/A"),
            core_codes=core_str,
            support_codes=support_str,
            context_codes=context_str,
            coverage_codes=coverage_str,
        )

        try:
            response = self.llm.chat(
                system_prompt=EXAMINER_REPORT_SYSTEM_PROMPT,
                user_message=user_prompt,
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse JSON response
            import json
            import re

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.info("Phase 8.5: LLM reasoning generated successfully")
                return result
            else:
                logger.warning("Phase 8.5: Could not parse LLM response as JSON")
                return {"executive_summary": response, "code_reasoning": []}

        except Exception as e:
            logger.warning("Phase 8.5: LLM reasoning failed: %s", e)
            return {}

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "layer1_core": [],
            "layer2_support": [],
            "layer2_context": [],
            "layer3_coverage": [],
            "all_codes": [],
            "role_summary": "No candidates to label",
            "reasoning_graph": {},
            "rules_log": [],
        }


def apply_role_labeling(
    candidates: List[Dict[str, Any]],
    phase1_data: Dict[str, Any],
    phase36_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Quick function to apply CPC role labeling."""
    labeler = CPCRoleLabeling()
    return labeler.label_roles(candidates, phase1_data, phase36_result)
