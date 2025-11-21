"""
main.py - pipeline complète YAML-friendly

Étapes :
1. Télécharger les filings aléatoires (filtrés par type si nécessaire)
2. Créer des chunks à partir des markdowns
3. Weak labelling avec LLM local
4. Sauvegarde des résultats et des cas multi-label à revoir
"""

import os
import json
import yaml
from src.data_fetching.fetch_reports import ReportFetcher
from src.preprocessing.chunking import create_chunks
from src.weak_labelling.llm_classifier import LocalLLMClassifier

# -------------------------
# 1️⃣ Charger la config YAML
# -------------------------
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

api_key = config["api"]["key"]
storage_raw = config["download"]["output_dir"]
n_companies = config["download"].get("n_random_companies", 100)
n_filings_per_company = config["download"].get("n_filings_per_company", 2)
filing_type_id = config["download"].get("filing_type_id")  # optional

chunk_size = config["chunking"]["chunk_size"]
overlap = config["chunking"].get("overlap", 0)
chunk_dir = config["chunking"]["output_dir"]

categories = config["weak_labelling"]["categories"]
llm_model = config["weak_labelling"]["llm_model"]
weak_file = config["weak_labelling"]["output_file"]
multi_file = config["weak_labelling"]["multi_label_file"]

# -------------------------
# 2️⃣ Télécharger les filings aléatoires
# -------------------------
os.makedirs(storage_raw, exist_ok=True)
fetcher = ReportFetcher(api_key=api_key, storage_dir=storage_raw)

print(f"📥 Télécharger {n_companies} entreprises aléatoires...")
reports = fetcher.fetch_random_filings(
    n_companies=n_companies,
    n_filings_per_company=n_filings_per_company,
    filing_type_id=filing_type_id
)
print(f"✅ {len(reports)} rapports téléchargés")

# -------------------------
# 3️⃣ Créer les chunks
# -------------------------
os.makedirs(chunk_dir, exist_ok=True)
all_chunks = []

for report in reports:
    chunks = create_chunks(
        markdown_path=os.path.join(storage_raw, f"{report.report_id}.md"),
        chunk_size=chunk_size,
        overlap=overlap,
        output_dir=chunk_dir
    )
    all_chunks.extend(chunks)

print(f"✅ {len(all_chunks)} chunks créés")

# -------------------------
# 4️⃣ Weak labelling
# -------------------------
llm = LocalLLMClassifier(model_name=llm_model)
weak_results = []

for chunk in all_chunks:
    # Chaque chunk peut appartenir à plusieurs catégories -> révision manuelle
    result = llm.classify_multi(chunk["text"], categories)
    result["chunk_id"] = chunk["chunk_id"]
    weak_results.append(result)

# -------------------------
# 5️⃣ Séparer les cas nécessitant une relecture manuelle
# -------------------------
for r in weak_results:
    # Si aucune catégorie détectée -> "Other"
    if not r["predicted_categories"]:
        r["final_label"] = "Other"
    # Si plusieurs catégories -> mark manual_review
    elif len(r["predicted_categories"]) > 1:
        r["final_label"] = "manual_review"
    else:
        r["final_label"] = r["predicted_categories"][0]

manual_cases = [r for r in weak_results if r["final_label"] == "manual_review"]

# -------------------------
# 6️⃣ Sauvegarde des résultats
# -------------------------
os.makedirs(os.path.dirname(weak_file), exist_ok=True)
with open(weak_file, "w", encoding="utf-8") as f:
    json.dump(weak_results, f, indent=2, ensure_ascii=False)

with open(multi_file, "w", encoding="utf-8") as f:
    json.dump(manual_cases, f, indent=2, ensure_ascii=False)

print("✅ Weak labelling terminé")
print(f"- Total chunks : {len(weak_results)}")
print(f"- Chunks nécessitant relecture manuelle : {len(manual_cases)}")
