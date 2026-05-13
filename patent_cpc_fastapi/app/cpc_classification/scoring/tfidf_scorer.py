"""tfidf_scorer.py — TF-IDF scoring with bigram support for Phase 2C."""

import math
import re
from typing import Dict, Any, Set, List, Tuple, Optional
from collections import Counter

from ..utils.text_utils import normalize_word


def tokenize(text: str) -> Set[str]:
    """Tokenize text into normalised words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {normalize_word(w) for w in words if len(w) > 2}


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


def score_candidates(
    all_subgroups: List[Dict[str, Any]],
    term_importance: Dict[str, int],
    system_context: str,
    core_function: str,
    strategy: str,
    phase2a_result: Dict[str, Any],
    negative_signals: List[Dict[str, Any]],
) -> List[Tuple[float, Dict[str, Any], int]]:
    """Score candidates using TF-IDF with bigram support.

    Returns list of (score, subgroup_dict, matching_terms_count) sorted descending.
    """
    scored: List[Tuple[float, Dict[str, Any], int]] = []
    title_count = len(all_subgroups)
    doc_freq: Counter = Counter()
    all_context_bigrams: List[Set[str]] = []

    for sg in all_subgroups:
        context = sg.get("full_context", sg["title"]).lower()
        unigrams, bigrams = tokenize_with_bigrams(context)
        all_context_bigrams.append(bigrams)
        for token in unigrams:
            doc_freq[token] += 1
        for bigram in bigrams:
            doc_freq[bigram] += 1

    for idx, sg in enumerate(all_subgroups):
        score = 0.0
        matching_terms = 0
        context = sg.get("full_context", sg["title"]).lower()
        context_tokens = {
            normalize_word(w) for w in re.findall(r"[a-zA-Z]+", context) if len(w) > 2
        }
        context_bigrams = all_context_bigrams[idx]
        title_lower = sg["title"].lower()

        # Negative signal penalties
        for neg in negative_signals:
            if isinstance(neg, dict):
                term = neg.get("term", "").lower()
                conf = neg.get("confidence", 0.5)
                if term and (term in context or term in title_lower):
                    score -= 5.0 * conf

        # Term matching
        for term, importance in term_importance.items():
            term_score = 0.0
            importance_weight = importance / 10.0
            term_tokens = {
                normalize_word(t) for t in term.lower().split() if len(t) > 2
            }

            # Multi-word phrase match
            if len(term.split()) >= 2 and term in context:
                avg_df = sum(doc_freq.get(t, 1) for t in term_tokens) / len(term_tokens)
                idf = math.log(title_count / max(avg_df, 1))
                term_score += idf * importance_weight * 8
                matching_terms += 1

            # Bigram matching
            term_bigrams = make_term_bigrams(term)
            if term_bigrams and context_bigrams:
                for tb in term_bigrams:
                    if tb in context_bigrams:
                        avg_df = doc_freq.get(tb, 1)
                        idf = math.log(title_count / max(avg_df, 1))
                        term_score += idf * (importance / 5.0) * 6
                        matching_terms += 1

            # Word overlap
            overlap = term_tokens & context_tokens
            if overlap:
                avg_df = sum(doc_freq.get(t, 1) for t in overlap) / len(overlap)
                idf = math.log(title_count / max(avg_df, 1))
                term_score += idf * (importance / 5.0) * 3

            # Single-word match
            if len(term.split()) == 1 and term in context:
                idf = math.log(title_count / max(doc_freq.get(term, 1), 1))
                term_score += idf * importance_weight * 5
                matching_terms += 1

            score += term_score

        scored.append((score, sg, matching_terms))

    scored.sort(key=lambda x: -x[0])
    return scored
