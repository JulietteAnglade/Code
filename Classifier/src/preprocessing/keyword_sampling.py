# src/preprocessing/keyword_sampling.py
"""
Fonctions utilitaires pour trouver mots-clés et proposer positions à échantillonner.
"""

import random
from typing import List, Tuple
import regex as re

def find_keyword_spans(text: str, keywords: List[str]) -> List[Tuple[int, int]]:
    """Retourne spans (start, end) de toutes les occurrences (insensible à la casse)."""
    spans = []
    if not keywords:
        return spans
    lower = text.lower()
    for kw in keywords:
        if not kw:
            continue
        pattern = re.compile(re.escape(kw.lower()))
        for m in pattern.finditer(lower):
            spans.append((m.start(), m.end()))
    return spans

def sample_positions_by_keywords(
    text: str,
    keywords: List[str],
    n_samples: int = 10,
    chunk_size: int = 500,
    window: int = 50,
    bias_to_end_beta: Tuple[float, float] = (2, 5)
) -> List[int]:
    """
    Propose positions de départ à utiliser pour chunking.
    Mélange : occurrences autour des keywords + positions aléatoires biaisées vers la fin.
    """
    text_len = len(text)
    starts = []

    # positions autour des keywords (prioritaires)
    spans = find_keyword_spans(text, keywords)
    for s, e in spans:
        center = (s + e) // 2
        start = max(0, center - chunk_size // 2 - window)
        start = min(start, max(0, text_len - chunk_size))
        starts.append(start)

    # compléter avec positions aléatoires biaisées vers la fin
    while len(starts) < n_samples:
        if text_len <= chunk_size:
            starts.append(0)
            continue
        r = int(random.betavariate(bias_to_end_beta[0], bias_to_end_beta[1]) * (text_len - chunk_size))
        starts.append(max(0, min(r, text_len - chunk_size)))
    # dedupe et clamp
    unique = []
    seen = set()
    for s in starts:
        s = int(max(0, s))
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:n_samples]
