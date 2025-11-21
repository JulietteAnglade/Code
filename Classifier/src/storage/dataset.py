# src/storage/dataset.py
"""
Dataset manager : stockage / récupération / accès aux rapports, chunks et embeddings.
Sérialisation via pickle (simple) ; peut être adapté vers parquet/DB si souhaité.
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ReportRecord:
    report_id: str
    raw_text: str
    metadata: Dict

@dataclass
class ChunkRecord:
    chunk_id: str
    report_id: str
    text: str
    start: int
    end: int
    score: float = 0.0
    weak_labels: Optional[Dict[str, float]] = None

@dataclass
class EmbeddingRecord:
    chunk_id: str
    embedding: np.ndarray
    weak_labels: Optional[Dict[str, float]] = None
    predicted_labels: Optional[Dict[str, float]] = None
    metadata: Dict = field(default_factory=dict)

class DatasetManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        # containers
        self.reports: Dict[str, ReportRecord] = {}
        self.chunks: Dict[str, ChunkRecord] = {}
        self.embeddings: Dict[str, EmbeddingRecord] = {}

    # --------------------
    # Load / Save
    # --------------------
    def save(self):
        with open(self.storage_path, "wb") as f:
            pickle.dump({
                "reports": self.reports,
                "chunks": self.chunks,
                "embeddings": self.embeddings
            }, f)

    def load(self):
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "rb") as f:
            data = pickle.load(f)
            self.reports = data.get("reports", {})
            self.chunks = data.get("chunks", {})
            self.embeddings = data.get("embeddings", {})

    # --------------------
    # Reports
    # --------------------
    def add_report(self, report_id: str, raw_text: str, metadata: Dict):
        self.reports[report_id] = ReportRecord(report_id=report_id, raw_text=raw_text, metadata=metadata)

    def get_report(self, report_id: str) -> Optional[ReportRecord]:
        return self.reports.get(report_id)

    # --------------------
    # Chunks
    # --------------------
    def add_chunk(self, chunk: ChunkRecord):
        self.chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self.chunks.get(chunk_id)

    def list_chunks(self) -> List[ChunkRecord]:
        return list(self.chunks.values())

    # --------------------
    # Embeddings
    # --------------------
    def add_embedding(self, emb: EmbeddingRecord):
        self.embeddings[emb.chunk_id] = emb

    def get_embedding(self, chunk_id: str) -> Optional[EmbeddingRecord]:
        return self.embeddings.get(chunk_id)

    def list_embeddings(self) -> List[EmbeddingRecord]:
        return list(self.embeddings.values())

    # --------------------
    # Utilities for ML
    # --------------------
    def get_embedding_matrix_and_label_matrix(self, categories: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Retourne:
          X : np.array (n_samples, dim)
          Y : np.array (n_samples, n_classes) binaire (0/1) selon weak_labels (ou 0 si manquant)
          ids: list des chunk_ids dans cet ordre
        """
        items = []
        for emb in self.embeddings.values():
            if emb.embedding is None:
                continue
            items.append(emb)

        # fix order
        items = sorted(items, key=lambda x: x.chunk_id)
        X = np.vstack([it.embedding for it in items]) if items else np.empty((0, 0))
        Y = []
        ids = []
        for it in items:
            ids.append(it.chunk_id)
            wl = it.weak_labels or {}
            row = [float(wl.get(cat, 0.0)) for cat in categories]
            Y.append(row)
        Y = np.array(Y) if Y else np.empty((0, len(categories)))
        return X, Y, ids

    def update_predicted_labels(self, chunk_id: str, predicted: Dict[str, float]):
        emb = self.embeddings.get(chunk_id)
        if emb:
            emb.predicted_labels = predicted

import pandas as pd
import json
from pathlib import Path

from src.models.utils import merge_weak_and_manual


class DatasetStore:
    def __init__(self, weak_labels_path: str, manual_review_path: str):
        self.weak_labels_path = weak_labels_path
        self.manual_review_path = manual_review_path

    def load_weak_labels(self):
        return pd.read_json(self.weak_labels_path)

    def load_manual_labels(self):
        if not Path(self.manual_review_path).exists():
            return pd.DataFrame(columns=["chunk_id", "final_label"])
        return pd.read_json(self.manual_review_path)

    def build_final_dataset(self):
        weak = self.load_weak_labels()
        manual = self.load_manual_labels()
        final = merge_weak_and_manual(weak, manual)

        # on filtre les "manual_review" restants (non traités)
        final = final[final["final_label"] != "manual_review"]

        return final
