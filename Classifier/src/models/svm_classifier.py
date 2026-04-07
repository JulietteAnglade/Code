import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SemiSupervisedSVMClassifier:
    """
    Pipeline semi-supervisé pour classifier des embeddings
    en utilisant :
    - weak labels (LLM)
    - SVM labels
    - human labels (annotations humaines)

    Stockage optimisé :
    - embeddings.npy
    - metadata.json
    - borderline.json (pour annotation)
    """

    # ---------------------------------------------------------
    # INITIALISATION
    # ---------------------------------------------------------

    def __init__(self, embeddings_path, metadata_path,
                 borderline_path="borderline.json",
                 margin_threshold=0.15):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.borderline_path = borderline_path
        self.margin_threshold = margin_threshold

        self.X = None            # embeddings
        self.metadata = None     # textes, weak_label, svm_label, human_label
        self.model = None        # modèle SVM
        self.borderline_samples = []  # à annoter

    # ---------------------------------------------------------
    # CHARGEMENT & SAUVEGARDE
    # ---------------------------------------------------------

    def load(self):
        """Charge embeddings et metadata."""
        self.X = np.load(self.embeddings_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def save_metadata(self):
        """Sauvegarde metadata.json."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def save_borderline(self):
        """Sauvegarde borderlines dans borderline.json."""
        with open(self.borderline_path, "w", encoding="utf-8") as f:
            json.dump(self.borderline_samples, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------
    # SÉLECTION DES LABELS POUR ENTRAÎNEMENT
    # ---------------------------------------------------------

    def _get_training_labels(self):
        """Construit y pour le SVM en priorisant human_label > svm_label > weak_label."""

        y = []
        for item in self.metadata:

            if item.get("human_label") is not None:
                y.append(item["human_label"])

            elif item.get("svm_label") is not None and item["svm_label"] != -1:
                y.append(item["svm_label"])

            else:
                y.append(item["weak_label"])

        return np.array(y)

    # ---------------------------------------------------------
    # ENTRAÎNEMENT SVM
    # ---------------------------------------------------------

    def train(self):
        """Entraîne un SVM multi-classes avec labels semi-supervisés."""

        y = self._get_training_labels()

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear"))
        ])

        self.model.fit(self.X, y)

    # ---------------------------------------------------------
    # PRÉDICTION + MISE À JOUR DES svm_label
    # ---------------------------------------------------------

    def assign_svm_labels(self):
        """Prédit les labels et les stocke dans svm_label."""
        preds = self.model.predict(self.X)
        for i, pred in enumerate(preds):
            self.metadata[i]["svm_label"] = int(pred)

        self.save_metadata()

    # ---------------------------------------------------------
    # IDENTIFICATION DES POINTS BORDERLINE
    # ---------------------------------------------------------

    def detect_borderline(self):
        """
        Identifie les points dont la marge top1-top2 est faible.
        On ne retient AUCUN point ayant un human_label déjà fourni.
        """

        scores = self.model.decision_function(self.X)
        self.borderline_samples = []

        for i, sc in enumerate(scores):

            # ignorer points déjà annotés
            if self.metadata[i].get("human_label") is not None:
                continue

            sorted_scores = np.sort(sc)
            margin = sorted_scores[-1] - sorted_scores[-2]

            if margin < self.margin_threshold:
                self.borderline_samples.append({
                    "index": i,
                    "chunk_ID": self.metadata[i]["chunk_ID"],
                    "text": self.metadata[i]["text"],
                    "weak_label": self.metadata[i]["weak_label"],
                    "svm_label": self.metadata[i]["svm_label"],
                    "margin": float(margin)
                })

        # ordre : du plus ambigu au moins ambigu
        self.borderline_samples.sort(key=lambda s: s["margin"])

        self.save_borderline()

    # ---------------------------------------------------------
    # FONCTION POUR L'ANNOTATION HUMAINE
    # ---------------------------------------------------------

    def apply_human_labels(self, annotations):
        """
        annotations : dict ou list de (index → label)
        Exemple :
            {
                42: 3,
                87: 1
            }
        """
        for index, new_label in annotations.items():
            self.metadata[index]["human_label"] = int(new_label)

        self.save_metadata()

    # ---------------------------------------------------------
    # PIPELINE COMPLET D'UNE ITÉRATION
    # ---------------------------------------------------------

    def run_iteration(self):
        """
        1. Charge données
        2. Entraîne SVM (weak + svm + human)
        3. Prédiction SVM
        4. Détection borderline
        Renvoie la liste des samples à annoter.
        """
        self.load()
        self.train()
        self.assign_svm_labels()
        self.detect_borderline()
        return self.borderline_samples
