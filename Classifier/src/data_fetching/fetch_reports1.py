import os
import time
import random
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass

BASE_URL = "https://api.financialreports.eu"


@dataclass
class Report:
    report_id: str
    markdown: str
    metadata: Dict


class ReportFetcher:
    """
    Version solide :
    - Corrige automatiquement les URLs 'next' brisées renvoyées par l'API
    - Fonctionne avec /companies/ et /filings/
    - Compatible avec plusieurs filing types (quarterly, annual…)
    """

    def __init__(self, api_key: str, base_url: str = BASE_URL, storage_dir: str = "data/raw"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-Key": api_key, "Accept": "application/json"}
        )
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Clean API URLs
    # ----------------------------------------------------------
    def _normalize_url(self, url: str) -> str:
        """Corrige les URLs anormales renvoyées dans 'next'."""
        # Cas 1 : URL relative "/companies/?page=2"
        if url.startswith("/"):
            url = self.base_url + url

        # Cas 2 : /api/ ajouté à tort
        url = url.replace("/api/companies", "/companies")
        url = url.replace("/api/filings", "/filings")

        # Suppression des double-slashes
        url = url.replace("//filings", "/filings")
        url = url.replace("//companies", "/companies")

        return url

    # ----------------------------------------------------------
    # Requests wrapper
    # ----------------------------------------------------------
    def request(self, method: str, path_or_url: str, params=None, timeout=10):
        # Construire l’URL
        if path_or_url.startswith("http"):
            url = path_or_url
        else:
            url = f"{self.base_url}/{path_or_url.lstrip('/')}"
        
        # Normalisation globale
        url = self._normalize_url(url)

        r = self.session.request(method, url, params=params, timeout=timeout)

        if r.status_code >= 400:
            print(f"[HTTP {r.status_code}] {r.text}")

        r.raise_for_status()
        return r

    # ----------------------------------------------------------
    # Companies
    # ----------------------------------------------------------
    def list_all_company_ids(self, page_size=100, max_pages=None) -> List[str]:
        company_ids = []
        next_url = f"{self.base_url}/companies/?page_size={page_size}"

        pages = 0

        while next_url:
            next_url = self._normalize_url(next_url)

            r = self.request("GET", next_url)
            data = r.json()

            for c in data.get("results", []):
                company_ids.append(str(c["id"]))

            next_url = data.get("next")

            pages += 1
            if max_pages and pages >= max_pages:
                break

            time.sleep(0.1)

        return company_ids

    # ----------------------------------------------------------
    # Filings
    # ----------------------------------------------------------
    def iter_filings(self, company_id: str, filing_type_id: Optional[str] = None, page_size=50):
        params = {"page_size": page_size, "company_id": company_id}

        if filing_type_id:
            params["filing_type"] = filing_type_id

        next_url = None
        first = True

        while True:
            if first:
                r = self.request("GET", "/filings/", params=params)
                first = False
            else:
                if not next_url:
                    break
                next_url = self._normalize_url(next_url)
                r = self.request("GET", next_url)

            data = r.json()

            for item in data.get("results", []):
                yield item

            next_url = data.get("next")
            if not next_url:
                break

    # ----------------------------------------------------------
    # Markdown download
    # ----------------------------------------------------------
    def fetch_markdown(self, filing_id: str) -> Report:
        # markdown
        r = self.request("GET", f"/filings/{filing_id}/markdown/")
        markdown = r.text

        # metadata
        mr = self.request("GET", f"/filings/{filing_id}/")
        metadata = mr.json()

        # save
        path = os.path.join(self.storage_dir, f"{filing_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return Report(report_id=filing_id, markdown=markdown, metadata=metadata)

    # ----------------------------------------------------------
    # Random sampling
    # ----------------------------------------------------------
    def fetch_random_filings(self, n_companies=50, n_filings_per_company=2, filing_type_ids=None):
        if filing_type_ids is None:
            filing_type_ids = []

        # Get all company IDs
        all_ids = self.list_all_company_ids()
        selected = random.sample(all_ids, min(n_companies, len(all_ids)))

        reports = []

        for cid in selected:
            for ft in filing_type_ids:
                filings = list(self.iter_filings(cid, filing_type_id=ft))

                if not filings:
                    continue

                sample = random.sample(filings, min(n_filings_per_company, len(filings)))

                for f in sample:
                    fid = f["id"]
                    try:
                        rep = self.fetch_markdown(fid)
                        reports.append(rep)
                    except Exception as e:
                        print(f"[Error downloading {fid}] {e}")

        return reports
