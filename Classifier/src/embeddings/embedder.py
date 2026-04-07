# src/embeddings/embedder.py
"""
ModernBert embedder simplifié (batching).
Model : 'answerdotai/ModernBERT-base'
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

""" Embedder using ModernBert model"""


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
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text, returns numpy array (1,768)"""
        inputs = self.tokenizer(text, return_tensors="pt", padding = True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            last_hidden_state = outputs.last_hidden_state  # (1, seq_len, dim)
            # mean pooling with attention mask
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            mean_pooled = sum_embeddings / sum_mask
            return mean_pooled.cpu().numpy()[0]

    