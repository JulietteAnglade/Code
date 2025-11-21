from ollama import Client
import json
from typing import List, Dict

class LocalLLMClassifier:
    """
    Weak labelling using a local LLM (via Ollama).
    The model answers "OUI" or "NON" for each category.
    """

    def __init__(self, model_name: str = "mixtral"):
        self.client = Client()
        self.model_name = model_name

    def ask_yes_no(self, text: str, category: str) -> bool:
        prompt = (
            f"Voici un texte:\n\n"
            f"{text}\n\n"
            f"Question: Does this text belong to the following category: {category} ? "
            "Answer by Yes or No."
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip().upper()

        if answer.startswith("Yes"):
            return True
        elif answer.startswith("No"):
            return False
        else:
            # fallback
            return False

    def classify(self, text: str, categories: List[str]) -> Dict:
        labels = []
        for cat in categories:
            if self.ask_yes_no(text, cat):
                labels.append(cat)

        if len(labels) == 0:
            final_label = "other"
        elif len(labels) == 1:
            final_label = labels[0]
        else:
            final_label = "manual_review"  # for Streamlit

        return {
            "text": text,
            "labels_proposed": labels,
            "final_label": final_label,
        }
