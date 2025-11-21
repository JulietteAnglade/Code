# src/models/svm_classifier.py
"""
Classifier SVM multi-classe (sans OneVsRest). Nous utilisons LinearSVC (Crammer-Singer) + calibration
pour obtenir predict_proba fiable.
"""

import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List, Dict

class SVMClassifier:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.base = LinearSVC(C=self.C, multi_class="crammer_singer", max_iter=5000)
        self.model: Optional[CalibratedClassifierCV] = None
        self.label_encoder = LabelEncoder()

    def train(self, X: np.ndarray, y: List[str], calibrate_cv: int = 3):
        """
        X: (n_samples, dim)
        y: list of string labels (one per sample)
        """
        if len(y) == 0:
            raise ValueError("Empty training labels.")
        y_enc = self.label_encoder.fit_transform(y)
        calibrator = CalibratedClassifierCV(base_estimator=self.base, cv=calibrate_cv, method="isotonic")
        calibrator.fit(X, y_enc)
        self.model = calibrator
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        probs = self.model.predict_proba(X)  # shape (n, n_classes)
        # map to label names
        return probs

    def predict_label(self, X: np.ndarray) -> List[str]:
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def predict_with_proba(self, X: np.ndarray) -> List[Dict[str, float]]:
        probs = self.predict_proba(X)
        labels = self.label_encoder.inverse_transform(range(probs.shape[1]))
        out = []
        for row in probs:
            out.append({lab: float(p) for lab, p in zip(labels, row)})
        return out

    def save(self, path: str):
        joblib.dump({"model": self.model, "le": self.label_encoder}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["le"]
        return self
