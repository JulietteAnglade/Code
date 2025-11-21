# src/models/active_learning.py
"""
Sélection d'exemples incertains pour active learning.
Stratégies supportées: 'margin' (top1-top2), 'entropy', 'max_prob' (1 - max_prob).
"""

import numpy as np
from typing import List

def entropy_score(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1 - eps)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent

def margin_score(probs: np.ndarray) -> np.ndarray:
    # margin = top1 - top2 ; smaller -> more uncertain. We convert to score where larger = more uncertain
    sorted_p = -np.sort(-probs, axis=1)  # descending
    top1 = sorted_p[:, 0]
    top2 = sorted_p[:, 1] if sorted_p.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    return 1.0 - margin  # larger = more uncertain

def select_uncertain(probs: np.ndarray, k: int = 100, strategy: str = "margin") -> List[int]:
    if probs.size == 0:
        return []
    if strategy == "entropy":
        scores = entropy_score(probs)
    elif strategy == "max_prob":
        scores = 1.0 - probs.max(axis=1)
    else:  # margin
        scores = margin_score(probs)
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


