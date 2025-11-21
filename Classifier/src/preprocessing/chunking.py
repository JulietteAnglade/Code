"""
Découpage en chunks de taille fixe, avec biais vers la fin et priorisation keywords.
"""

from dataclasses import dataclass
from typing import List, Optional
import random
from .keyword_sampling import sample_positions_by_keywords

@dataclass
class TextChunk:
    chunk_id: str
    report_id: str
    text: str
    start: int
    end: int
    score: float = 0.0
    weak_labels: Optional[dict] = None

def chunk_report(
    report_id: str,
    report_text: str,
    chunk_size: int = 800,
    n_chunks: int = 10,
    keywords: Optional[List[str]] = None,
    score_fn=None
) -> List[TextChunk]:
    """
    Génère des TextChunk pour un rapport.
    - utilise sample_positions_by_keywords pour proposer positions
    - construit chunk text à partir de start:start+chunk_size
    - score_fn(chunk_text) -> float (optionnel) pour priorisation
    """
    if keywords is None:
        keywords = []
    starts = sample_positions_by_keywords(report_text, keywords, n_samples=n_chunks, chunk_size=chunk_size)
    chunks = []
    for start in starts:
        end = min(start + chunk_size, len(report_text))
        txt = report_text[start:end]
        score = 0.0
        if score_fn:
            try:
                score = float(score_fn(txt))
            except Exception:
                score = 0.0
        chunk_id = f"{report_id}_{start}"
        chunks.append(TextChunk(chunk_id=chunk_id, report_id=report_id, text=txt, start=start, end=end, score=score))
    # tri décroissant par score
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks
