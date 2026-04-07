import json
import streamlit as st
from pathlib import Path

BORDERLINE_PATH = "borderline.json"
OUTPUT_ANNOTATIONS_PATH = "annotations.json"


st.set_page_config(page_title="Annotation humaine", layout="wide")

st.title("📝 Interface d’Annotation Humaine")
st.write("Annoter manuellement les samples borderline détectés par le classifieur.")


# -------------------------------------------------------------------
# 1. Charger borderline.json
# -------------------------------------------------------------------

def load_borderline():
    if not Path(BORDERLINE_PATH).exists():
        st.error(f"Fichier {BORDERLINE_PATH} introuvable.")
        return []

    with open(BORDERLINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


borderline = load_borderline()

if not borderline:
    st.stop()

st.success(f"{len(borderline)} samples à annoter")


# -------------------------------------------------------------------
# 2. Initialiser session_state pour stocker les annotations
# -------------------------------------------------------------------

if "annotations" not in st.session_state:
    st.session_state.annotations = {}   # {index: label}
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0     # position dans la liste


# -------------------------------------------------------------------
# 3. Accès au sample courant
# -------------------------------------------------------------------

def get_current_sample():
    idx = st.session_state.current_idx
    if 0 <= idx < len(borderline):
        return borderline[idx]
    return None


sample = get_current_sample()

if not sample:
    st.success("🎉 Tous les samples ont été annotés !")
    st.write("Vous pouvez maintenant exporter les annotations.")
else:

    st.markdown("---")
    st.subheader(f"Sample {st.session_state.current_idx + 1} / {len(borderline)}")
    st.markdown(f"**Chunk ID :** `{sample['chunk_ID']}`")
    st.markdown(f"**Marge :** `{sample['margin']:.4f}`")
    st.markdown(f"**Weak Label (LLM) :** `{sample['weak_label']}`")
    st.markdown(f"**SVM Label :** `{sample['svm_label']}`")
    st.write("### Texte :")
    st.text_area("", value=sample["text"], height=200, disabled=True)

    st.markdown("### Choisissez un label :")
    chosen_label = st.radio(
        "Label",
        options=[1, 2, 3, 4],
        horizontal=True,
        index=None,
        key=f"radio_{sample['index']}"
    )

    # Bouton pour valider
    if st.button("Valider l’annotation"):
        if chosen_label is None:
            st.warning("Veuillez choisir un label avant de valider.")
        else:
            st.session_state.annotations[sample["index"]] = chosen_label
            st.success(f"Annotation enregistrée : index={sample['index']} → label={chosen_label}")

            # avancer au prochain sample
            st.session_state.current_idx += 1
            st.experimental_rerun()


# -------------------------------------------------------------------
# 4. Export des annotations
# -------------------------------------------------------------------

st.markdown("---")
st.subheader("📤 Exporter les annotations")

st.write("Lorsque vous avez terminé, cliquez sur le bouton ci-dessous pour générer le fichier :")
st.code("annotations.json", language="text")

if st.button("Exporter les annotations"):
    with open(OUTPUT_ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state.annotations, f, indent=2)

    st.success(f"Annotations exportées dans {OUTPUT_ANNOTATIONS_PATH}")
    st.write("Vous pouvez maintenant utiliser :")
    st.code("clf.apply_human_labels(annotations)", language="python")
