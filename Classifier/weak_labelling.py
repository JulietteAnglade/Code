import os
import json
import yaml
import logging
import streamlit as st
from src.weak_labelling.llm_classifier import LocalLLMClassifier

""" utilier un llm via hugging face """

# -------------------------
# 1️⃣ Config YAML et logging
# -------------------------
with open("Classifier/config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

categories = config["weak_labelling"]["categories"]
llm_model = config["weak_labelling"]["llm_model"]
weak_file = config["weak_labelling"]["output_file"]
multi_file = config["weak_labelling"]["multi_label_file"]

chunk_dir = config["chunking"]["output_dir"]
chunk_size = config["chunking"]["chunk_size"]
overlap = config["chunking"].get("overlap", 0)

logging.basicConfig(level=getattr(logging, config["logging"]["level"].upper(), logging.INFO))
logger = logging.getLogger(__name__)

# -------------------------
# 2️⃣ Charger les chunks depuis le dossier
# -------------------------
def load_chunks(chunk_dir):
    chunks = []
    for fname in os.listdir(chunk_dir):
        if fname.endswith(".json"):
            path = os.path.join(chunk_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                chunks.extend(json.load(f))
    logger.info(f"{len(chunks)} chunks chargés depuis {chunk_dir}")
    return chunks

all_chunks = load_chunks(chunk_dir)

# -------------------------
# 3️⃣ Weak labelling
# -------------------------
model_path = os.environ.get("LLAMA_GGML_PATH") or config["weak_labelling"].get("model_path")
try:
    llm = LocalLLMClassifier(model_path=model_path)
except ValueError as e:
    logger.error("LLM model path not set. Set LLAMA_GGML_PATH or weak_labelling.model_path in config.")
    raise
weak_results = []

for chunk in all_chunks:
    # Chaque chunk peut appartenir à plusieurs catégories -> révision manuelle
    result = llm.classify(chunk["text"], categories)
    result["chunk_id"] = chunk["chunk_id"]
    weak_results.append(result)

# -------------------------
# 4️⃣ Séparer les cas nécessitant une relecture manuelle
# -------------------------
for r in weak_results:
    if not r["labels_proposed"]:
        r["final_label"] = "other"
    elif len(r["labels_proposed"]) > 1:
        r["final_label"] = "manual_review"
    else:
        r["final_label"] = r["labels_proposed"][0]

manual_cases = [r for r in weak_results if r["final_label"] == "manual_review"]

# -------------------------
# 5️⃣ Sauvegarder les résultats weak labelling
# -------------------------
os.makedirs(os.path.dirname(weak_file), exist_ok=True)
with open(weak_file, "w", encoding="utf-8") as f:
    json.dump(weak_results, f, indent=2, ensure_ascii=False)

with open(multi_file, "w", encoding="utf-8") as f:
    json.dump(manual_cases, f, indent=2, ensure_ascii=False)

logger.info(f"Weak labelling terminé: {len(weak_results)} chunks, {len(manual_cases)} à relecture")

# -------------------------
# 6️⃣ Interface Streamlit pour relecture manuelle
# -------------------------
st.title("🔍 Manual Review — Textes avec catégories ambiguës")

if not manual_cases:
    st.info("Aucun chunk nécessitant une relecture manuelle.")
else:
    results = []
    for r in manual_cases:
        st.write("---")
        st.write(f"### Chunk ID: {r['chunk_id']}")
        st.write(f"**Texte :** {r['text']}")
        st.write(f"*LLM proposait :* {r['labels_proposed']}")

        label = st.radio(
            "Choisis la bonne catégorie :",
            options=["other", "manual_review"] + r["labels_proposed"],
            key=r["chunk_id"]
        )

        results.append({
            "chunk_id": r["chunk_id"],
            "final_label": label
        })

    if st.button("💾 Enregistrer les corrections"):
        output_manual_file = multi_file.replace(".json", "_manual_reviewed.json")
        with open(output_manual_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        st.success(f"Corrections sauvegardées dans {output_manual_file}")
