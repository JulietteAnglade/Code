import os
import random
import requests
import yaml
from pathlib import Path
from typing import List

class FinancialReportsDownloader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.api_key = self.config["api"]["key"]
        self.base_url = self.config["api"]["base_url"]
        self.headers = {"X-API-Key": self.api_key}
        self.storage_dir = Path(self.config["fetch"]["storage_dir"])
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.n_files = self.config["fetch"].get("n_files", 100)
        self.filing_type_names = self.config["fetch"]["filing_type_names"]
        self.language = self.config["fetch"].get("language", "en")

    @staticmethod
    def load_config(path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def fetch_filings_by_type(self, filing_type_name: str) -> List[dict]:
        """
        Récupère tous les filings correspondant à un type donné et à la langue définie.
        """
        filings = []
        url = f"{self.base_url}filings/"
        params = {
            "page": 1,
            "page_size": 100,
            "filing_type_name": filing_type_name,
            "language": self.language
        }

        while url:
            resp = requests.get(url, headers=self.headers, params=params)
            if resp.status_code != 200:
                print(f"Erreur récupération filings {filing_type_name}: {resp.status_code}")
                break
            data = resp.json()
            filings.extend(data.get("results", []))
            url = data.get("next")
            params = None  # params seulement pour la première page

        return filings

    def download_markdown(self, filing_id: str, company_name: str):
        """
        Télécharge le markdown d'un filing.
        """
        url = f"{self.base_url}filings/{filing_id}/markdown/"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code == 200:
            filename = self.storage_dir / f"{company_name}_{filing_id}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"Téléchargé: {filename}")
        else:
            print(f"Erreur téléchargement filing {filing_id}: {resp.status_code}")

    def run(self):
        all_filings = []
        for filing_type in self.filing_type_names:
            filings = self.fetch_filings_by_type(filing_type)
            print(f"{len(filings)} filings récupérés pour type '{filing_type}'")
            all_filings.extend(filings)

        if not all_filings:
            print("Aucun filing disponible pour les types sélectionnés.")
            return

        # Échantillonnage aléatoire N filings
        sampled_filings = random.sample(all_filings, min(self.n_files, len(all_filings)))

        for filing in sampled_filings:
            company_name = filing.get("company_name", f"company_{filing['company_id']}")
            self.download_markdown(filing["id"], company_name)
