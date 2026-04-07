import os
import json
import yaml
from glob import glob
from src.preprocessing.chunking import chunk_report

# -------------------------
# 1️⃣ Charger la config YAML
# -------------------------
with open("Classifier/config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

storage_raw = config["fetch"]["storage_dir"]
chunk_size = config["chunking"]["chunk_size"]
overlap = config["chunking"].get("overlap", 0)
chunk_dir = config["chunking"]["output_dir"]

os.makedirs(chunk_dir, exist_ok=True)

# -------------------------
# 2️⃣ Parcourir tous les markdowns et créer les chunks
# -------------------------
all_chunks = []

markdown_files = glob(os.path.join(storage_raw, "*.md"))

for md_file in markdown_files:
    report_id = os.path.splitext(os.path.basename(md_file))[0]
    with open(md_file, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Découpage en chunks (tu peux passer tes keywords si tu en as)
    chunks = chunk_report(
        report_id=report_id,
        report_text=report_text,
        chunk_size=chunk_size,
        n_chunks=10,        # nombre de chunks par rapport, modifiable
        keywords=[],         # éventuellement tes keywords
        score_fn=None        # éventuellement une fonction de scoring
    )

    # Transformer les TextChunk en dicts pour sauvegarde JSON
    for c in chunks:
        all_chunks.append({
            "chunk_id": c.chunk_id,
            "report_id": c.report_id,
            "text": c.text,
            "start": c.start,
            "end": c.end,
            "score": c.score
        })

# -------------------------
# 3️⃣ Sauvegarder les chunks
# -------------------------
chunks_file = os.path.join(chunk_dir, "all_chunks.json")
with open(chunks_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"✅ {len(all_chunks)} chunks créés et sauvegardés dans {chunks_file}")


