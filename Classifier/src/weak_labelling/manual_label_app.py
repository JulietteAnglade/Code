import streamlit as st
import pandas as pd
import json
import os

DATA_PATH = "data/weak_labels/multi_label_cases.json"
OUTPUT_PATH = "data/weak_labels/manual_review_results.json"

st.title("🔍 Manual Review — Texts with Ambiguous Categories")

if not os.path.exists(DATA_PATH):
    st.error(f"Aucun fichier trouvé à {DATA_PATH}")
    st.stop()

df = pd.read_json(DATA_PATH)

st.write("## Textes à ré-étiqueter")
st.write("Sélectionne la bonne catégorie pour chaque texte.")

results = []

for i, row in df.iterrows():
    st.write("---")
    st.write(f"### Chunk ID: {row['chunk_id']}")
    st.write(f"**Texte :** {row['text']}")
    st.write(f"*LLM proposait :* {row['labels']}")

    label = st.radio(
        "Choisis la bonne catégorie :",
        options=["other", "manual_review"] + row["labels"],
        key=row["chunk_id"]
    )

    results.append({
        "chunk_id": row["chunk_id"],
        "final_label": label
    })

if st.button("💾 Enregistrer les corrections"):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    st.success("Corrections sauvegardées !")
