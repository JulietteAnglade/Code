import os
import time
import random
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BASE_URL = "https://api.financialreports.eu"

@dataclass
class Report:
    report_id: str
    markdown: str
    metadata: Dict

class ReportFetcher:
    """
    Client FinancialReports avec téléchargement aléatoire de filings d'entreprises.
    Compatible Level 1 (Markdown inaccessible).
    """

    def __init__(self, api_key: str, base_url: str = BASE_URL, storage_dir: str = "data/raw"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Accept": "application/json"})
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _handle_response(self, r: requests.Response):
        if r.status_code == 429:
            print("Rate limit exceeded!")
            raise requests.exceptions.RetryError("Rate limit exceeded (429).")
        if r.status_code >= 400 and r.status_code != 403:
            print(f"HTTP Error {r.status_code}: {r.text}")
            r.raise_for_status()
        return r

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10),
           retry=retry_if_exception_type(requests.exceptions.RequestException))
    def fetch_markdown(self, filing_id: str) -> Report:
        """Télécharge le markdown et metadata d'un filing, fallback Level 1"""
        url_md = f"{self.base_url}/filings/{filing_id}/markdown/"
        url_meta = f"{self.base_url}/filings/{filing_id}/"

        try:
            r = self.session.get(url_md, timeout=30)
            if r.status_code == 403:
                # Level 1 plan -> Markdown inaccessible
                print(f"⚠️ Level 1 plan: accès Markdown refusé pour {filing_id}")
                markdown = ""
            else:
                self._handle_response(r)
                markdown = r.text
        except requests.exceptions.RequestException as e:
            print(f"[fetch_markdown] Erreur Markdown {filing_id}: {e}")
            markdown = ""

        # récupérer metadata (toujours possible en Level 1)
        try:
            mr = self.session.get(url_meta, timeout=10)
            self._handle_response(mr)
            metadata = mr.json()
        except Exception:
            metadata = {"id": filing_id}

        # sauvegarde locale
        path = os.path.join(self.storage_dir, f"{filing_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return Report(report_id=filing_id, markdown=markdown, metadata=metadata)

    def list_filings(self, page: int = 1, page_size: int = 100, filters: Optional[Dict] = None) -> Dict:
        """Wrapper pour GET /filings/"""
        url = f"{self.base_url}/filings/"
        params = {"page": page, "page_size": page_size}
        if filters:
            params.update(filters)
        r = self.session.get(url, params=params, timeout=30)
        self._handle_response(r)
        return r.json()

    def list_all_company_ids(self, page_size: int = 100, max_pages: Optional[int] = None) -> List[str]:
        """Retourne tous les IDs d'entreprises existantes."""
        page = 1
        pages_fetched = 0
        company_ids = set()

        while True:
            url = f"{self.base_url}/companies/"
            params = {"page": page, "page_size": page_size}
            r = self.session.get(url, params=params, timeout=30)
            self._handle_response(r)
            data = r.json()

            for comp in data.get("results", []):
                company_ids.add(str(comp.get("id")))

            pages_fetched += 1
            if not data.get("next") or (max_pages and pages_fetched >= max_pages):
                break
            page += 1
            time.sleep(0.1)

        return list(company_ids)

    def fetch_random_filings(
        self, 
        n_companies: int = 100, 
        n_filings_per_company: int = 2, 
        filing_type_id: Optional[str] = None,
        sleep_between: float = 0.1
    ) -> List[Report]:
        """
        Sélectionne des entreprises au hasard et télécharge un nombre limité de filings pour chacune,
        avec filtrage optionnel par type de filing.
        Compatible Level 1: fallback Markdown vide si accès refusé.
        """
        reports = []

        # 1️⃣ Récupérer toutes les entreprises
        all_companies = self.list_all_company_ids()
        selected_companies = random.sample(all_companies, min(n_companies, len(all_companies)))

        # 2️⃣ Pour chaque entreprise, récupérer ses filings et en télécharger quelques-uns
        for company_id in selected_companies:
            filters = {"search": company_id}
            if filing_type_id:
                filters["filing_type_id"] = filing_type_id

            data = self.list_filings(filters=filters, page_size=50)
            filings = [f["id"] for f in data.get("results", [])]
            if not filings:
                continue

            sampled_filings = random.sample(filings, min(n_filings_per_company, len(filings)))
            for fid in sampled_filings:
                try:
                    report = self.fetch_markdown(fid)
                    reports.append(report)
                except Exception as e:
                    print(f"[fetch_markdown] Error {fid}: {e}")
                time.sleep(sleep_between)

        return reports
