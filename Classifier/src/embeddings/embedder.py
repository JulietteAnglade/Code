# src/embeddings/embedder.py
"""
ModernBert embedder simplifié (batching).
Model recommandé : 'answerdotai/ModernBERT-base'
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import math
import tqdm

@dataclass
class EmbeddedChunk:
    chunk_id: str
    embedding: np.ndarray
    weak_labels: Optional[dict] = None
    aggregated_label: Optional[str] = None

class ModernBertEmbedder:
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _batchify(self, texts: List[str]) -> List[List[str]]:
        n = len(texts)
        bs = self.batch_size
        return [texts[i:i+bs] for i in range(0, n, bs)]

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        # returns (n, dim)
        with torch.no_grad():
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            last_hidden = out.last_hidden_state  # (batch, seq_len, dim)
            # mean pooling with attention mask
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
            arr = pooled.cpu().numpy()
            return arr

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed list of texts and return array shape (n_texts, dim)."""
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        batches = self._batchify(texts)
        embeddings = []
        it = batches if not show_progress else tqdm.tqdm(batches, desc="Embedding")
        for b in it:
            emb = self._embed_batch(b)
            embeddings.append(emb)
        return np.vstack(embeddings)

    def embed_chunks(self, chunks: List[dict], show_progress: bool = False) -> List[EmbeddedChunk]:
        """chunks: list of dicts with keys chunk_id, text, weak_labels, aggregated_label"""
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        wls = [c.get("weak_labels") for c in chunks]
        aggs = [c.get("aggregated_label") for c in chunks]
        embs = self.embed_texts(texts, show_progress=show_progress)
        out = []
        for cid, vec, wl, agg in zip(ids, embs, wls, aggs):
            out.append(EmbeddedChunk(chunk_id=cid, embedding=vec.astype(np.float32), weak_labels=wl, aggregated_label=agg))
        return out
