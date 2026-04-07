# ...existing code...
from typing import List, Dict
from transformers import pipeline
import traceback
import sys

class LocalLLMClassifier2:
    """
    Weak labelling using a transformers zero-shot classifier (facebook/bart-large-mnli).
    Returns proposed labels based on per-label score thresholds.
    """

    def __init__(self, threshold_low: float = 0.5, threshold_high: float = 0.7):
        try:
            self.model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print("ERROR creating pipeline:", e, file=sys.stderr)
            traceback.print_exc()
            raise
        print("PIPELINE:", repr(self.model))
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def classify(self, text: str, categories: List[str]) -> Dict:
        try:
            answer = self.model(text, candidate_labels=categories, multi_label=True)
        except Exception as e:
            print("ERROR calling pipeline:", e, file=sys.stderr)
            traceback.print_exc()
            raise

        print("RAW ANSWER:", repr(answer))

        # defensive checks
        if not isinstance(answer, dict) or "labels" not in answer or "scores" not in answer:
            print("Unexpected pipeline output structure.", file=sys.stderr)
            return {
                "text": text,
                "labels_proposed": [],
                "final_label": "other",
                "raw": answer,
            }

        labels = answer["labels"]
        scores = answer["scores"]

        # propose labels whose score >= threshold_low
        proposed = [lbl for lbl, sc in zip(labels, scores) if sc >= self.threshold_low]

        if len(proposed) == 0:
            final_label = "other"
        elif len(proposed) == 1:
            final_label = proposed[0]
        else:
            final_label = "manual_review"

        if len(proposed) > 1 and max(scores) < self.threshold_high:
            final_label = "manual_review"

        print("PROPOSED:", proposed, "SCORES:", scores)

        return {
            "text": text,
            "labels_proposed": proposed,
            "final_label": final_label,
            "all_labels": labels,
            "all_scores": scores,
        }

# -------------------------
if __name__ == "__main__":
    classifier = LocalLLMClassifier2()
    text_sample = "Einstein developed the theory of relativity."
    categories = ["science", "politics", "sports"]
    result = classifier.classify(text_sample, categories)
    print(result)
