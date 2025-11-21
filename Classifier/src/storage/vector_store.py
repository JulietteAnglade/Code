# src/storage/vector_store.py

import pickle
from typing import List
import numpy as np

def save_embeddings(file_path: str, embedded_chunks: List):
    data = [(c.chunk_id, c.embedding, c.label) for c in embedded_chunks]
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_embeddings(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
