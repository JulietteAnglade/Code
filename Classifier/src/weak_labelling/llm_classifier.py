import os
from typing import List, Dict


# Install dependencies if not already installed:
# pip install llama-cpp-python huggingface_hub

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class LocalLLMClassifier:
    """
    Weak labelling using a small local LLM (Phi-2 Q4_K_M).
    The model answers "Yes" or "No" for each category.
    """

    def __init__(self):
        self.client = Llama.from_pretrained(
	    repo_id="tensorblock/phi-2-GGUF",
	    filename="phi-2-Q2_K.gguf",
        )
        

    def ask_yes_no(self, text: str, category: str) -> bool:
        """
        Returns True if model thinks text belongs to category, else False.
        """
        prompt = (
            f"Text:\n{text}\n\n"
            f"Question: Does this text belong to the category '{category}'? "
            "Answer strictly with Yes or No."
        )

        resp = self.client(prompt, max_tokens=4)
        answer = resp["choices"][0]["text"].strip().lower()
        print(answer)

        return answer.startswith("yes") or answer.startswith("oui")

    def classify(self, text: str, categories: List[str]) -> Dict:
        """
        Returns labels proposed by the model, and final label decision.
        """
        labels = [cat for cat in categories if self.ask_yes_no(text, cat)]

        if len(labels) == 0:
            final_label = "other"
        elif len(labels) == 1:
            final_label = labels[0]
        else:
            final_label = "manual_review"

        return {
            "text": text,
            "labels_proposed": labels,
            "final_label": final_label,
        }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    classifier = LocalLLMClassifier()  # will download model if missing

    text_sample = "Football is a popular sport worldwide."
    categories = ["science", "politics", "sports"]

    result = classifier.classify(text_sample, categories)
    print(result)
