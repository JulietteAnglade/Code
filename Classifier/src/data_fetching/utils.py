def resolve_filing_type_ids(fetcher, filing_type_names):
    """
    Convertit une liste de noms de types de filings en liste d'IDs.

    filing_type_names = ["Quarterly Report", "Annual Report"]
    """
    ids = []

    for name in filing_type_names:
        data = fetcher.session.get(
            f"{fetcher.base_url}/filing-types/",
            params={"search": name, "page_size": 5},
            timeout=10
        )

        data = data.json().get("results", [])
        if not data:
            print(f"⚠️ Aucun filing type trouvé pour '{name}'")
            continue

        ids.append(data[0]["id"])

    return ids
