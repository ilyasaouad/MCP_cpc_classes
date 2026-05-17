"""tfidf_scorer.py — BM25 scoring for Phase 2C (replacing TF-IDF)."""

import logging
import re
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

from ..utils.text_utils import normalize_word

def tokenize_with_bigrams(text: str) -> Tuple[Set[str], Set[str]]:
    """Tokenize into normalised unigrams AND bigrams."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    normalised = [normalize_word(w) for w in words if len(w) > 2]
    unigrams = set(normalised)
    bigrams = set()
    for i in range(len(normalised) - 1):
        bigrams.add(f"{normalised[i]}_{normalised[i + 1]}")
    return unigrams, bigrams

def make_term_bigrams(term: str) -> List[str]:
    """Generate bigrams from a multi-word term."""
    words = [normalize_word(w) for w in term.lower().split() if len(w) > 2]
    if len(words) < 2:
        return []
    return [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]

def _tokenize_bm25(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    text = text.lower()
    tokens = []
    for word in text.split():
        # Remove non-alphanumeric characters
        cleaned = "".join(c for c in word if c.isalnum())
        if len(cleaned) > 2:  # Skip very short tokens
            tokens.append(cleaned)
    return tokens

def score_candidates(
    all_subgroups: List[Dict[str, Any]],
    term_importance: Dict[str, int],
    system_context: str,
    core_function: str,
    strategy: str,
    phase2a_result: Dict[str, Any],
    negative_signals: List[Dict[str, Any]],
    cpc_terms: List[str] = None,
    core_function_precise: str = "",
    core_function_generalized: List[str] = None,
    evidence_table: List[Dict[str, Any]] = None,
) -> List[Tuple[float, Dict[str, Any], int]]:
    """
    Score candidates using BM25.
    
    Query is built from:
    1. cpc_terms (1.5x weight)
    2. core_function_precise
    3. core_function_generalized
    4. evidence_table (weight >= 6)
    
    Returns list of (normalized_bm25_score, subgroup_dict, matching_terms_count).
    """
    if not all_subgroups:
        return []

    # 1. Build weighted query string
    query_parts = []
    
    unique_cpc = set(cpc_terms) if cpc_terms else set()
    for t in unique_cpc:
        # Add 3 times for 1.5x weight relative to 2x for others
        query_parts.extend([t] * 3)
        
    others = set()
    if core_function_precise:
        others.add(core_function_precise)
    if core_function_generalized:
        others.update(core_function_generalized)
    if evidence_table:
        for row in evidence_table:
            if row.get("weight", 0) >= 6:
                others.add(row.get("term", ""))
                
    # Subtract cpc_terms from others to ensure correct ratio
    others -= unique_cpc
    for t in others:
        query_parts.extend([t] * 2)
        
    query_string = " ".join(query_parts)
    tokenized_query = _tokenize_bm25(query_string)
    
    if not tokenized_query:
        logger.warning("[Phase 2C] BM25 query is empty after tokenization.")
        return [(0.0, sg, 0) for sg in all_subgroups]

    # 2. Build local BM25 index for the batch
    corpus = []
    for sg in all_subgroups:
        text = sg.get("full_context", sg.get("title", "")).lower()
        corpus.append(_tokenize_bm25(text))
        
    if not any(corpus):
        logger.warning("[Phase 2C] Tokenized corpus is completely empty. Bypassing BM25 scoring.")
        return [(0.0, sg, 0) for sg in all_subgroups]
        
    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(tokenized_query)
    
    # 3. Min-Max Normalization to [0, 1]
    min_s = float(np.min(raw_scores))
    max_s = float(np.max(raw_scores))
    
    if max_s > min_s:
        normalized_scores = [(float(s) - min_s) / (max_s - min_s) for s in raw_scores]
    else:
        normalized_scores = [1.0 if s > 0 else 0.0 for s in raw_scores]

    # 4. Calculate matching terms count (for compatibility with existing logic)
    # We use unigrams from the query for this count
    unique_query_unigrams = set(tokenized_query)
    
    scored: List[Tuple[float, Dict[str, Any], int]] = []
    for i, sg in enumerate(all_subgroups):
        score = normalized_scores[i]
        
        # Heuristic: apply negative signal penalties (kept for quality)
        for neg in negative_signals:
            term = neg.get("term", "").lower() if isinstance(neg, dict) else str(neg).lower()
            conf = neg.get("confidence", 0.5) if isinstance(neg, dict) else 0.5
            if term and (term in sg.get("full_context", "").lower()):
                score -= 0.5 * conf # Scaled to normalized range
        
        # Count overlapping unigrams
        match_count = len(unique_query_unigrams & set(corpus[i]))
        
        scored.append((max(0.0, score), sg, match_count))

    # Sort descending by score
    scored.sort(key=lambda x: -x[0])
    return scored
