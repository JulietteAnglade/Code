# src/analytics/pca_analysis.py

import numpy as np
from sklearn.decomposition import PCA

def compute_pca(embeddings: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca

def compute_centroid(embeddings: np.ndarray):
    return np.mean(embeddings, axis=0)
