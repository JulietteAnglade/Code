import streamlit as st
import pandas as pd
import json
import os


class ManualLabelApp:
    """
    Streamlit app for manual review of texts with ambiguous categories.
    Need a dataframe with 'chunk_id' and 'text' columns, and a list of possible categories.
    Save results chunk_id, text, final_label to a JSON file.
    """

    def __init__(self, data: pd.DataFrame, categories: list, output_path: str = "data/manual_labelling/manual_labels.json"):
        self.data = data
        self.categories = categories
        self.output_path = output_path

    def manual_review(self):
        st.title("🔍 Manual Review — Texts with Ambiguous Categories"
                 )
        st.write("## Textes à ré-étiqueter")
        st.write("Sélectionne la bonne catégorie pour chaque texte.")

        # initialize accumulator in session state
        if "manual_results" not in st.session_state:
            st.session_state["manual_results"] = []

        for i, row in self.data.iterrows():
            st.write("---")
            st.write(f"### Chunk ID: {row['chunk_id']}")
            st.write(f"**Texte :** {row['text']}")

            key = str(row["chunk_id"])
            label = st.radio(
                "Choose the right category:",
                options=self.categories + ["other"],
                key=key
            )
            submit = st.button("Submit", key=f"submit_{key}")

            if submit:
                entry = {
                    "chunk_id": row["chunk_id"],
                    "text": row["text"],
                    "final_label": label
                }
                # avoid duplicates
                if entry not in st.session_state["manual_results"]:
                    st.session_state["manual_results"].append(entry)

                # ensure directory exists and save file
                os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8") as f:
                    json.dump(st.session_state["manual_results"], f, ensure_ascii=False, indent=2)

                st.success(f"Saved {len(st.session_state['manual_results'])} labels to {self.output_path}")

        return None
        
if __name__ == "__main__":
    data = pd.DataFrame([
        {"chunk_id": 1, "text": "The stock market crashed today."},
        {"chunk_id": 2, "text": "The new species of bird was discovered."}
    ])
    categories = ["finance", "science", "sports"]
    output_path = "data/manual_labelling/manual_label_test.json"
    app = ManualLabelApp(data, categories)
    result = app.manual_review()
    print(result)

