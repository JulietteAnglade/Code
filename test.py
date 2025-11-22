BASE = "https://api.financialreports.eu"
API_KEY = "my API Key"

HEADERS = {"X-API-Key": API_KEY, "Accept": "application/json"}
DEFAULT_TIMEOUT = 10

def request_simple(method, path_or_url, params=None, json_body=None, timeout=DEFAULT_TIMEOUT):
    """
    Envoie une requête et lève une erreur si le statut n'est pas 2xx.
    Accepte un chemin (ex: '/companies/') ou une URL complète.
    Retourne l'objet requests.Response.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        url = BASE.rstrip("/") + "/" + path_or_url.lstrip("/")
    resp = requests.request(method, url, headers=HEADERS, params=params, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp

def search_companies(query, page_size=5):
    """Retourne la liste (première page) des sociétés correspondant à `query`."""
    resp = request_simple("GET", "/companies/", params={"search": query, "page_size": page_size})
    return resp.json().get("results", [])

def search_filing_types(query, page_size=10):
    """Retourne la liste (première page) des types de rapport correspondant à `query`."""
    resp = request_simple("GET", "/filing-types/", params={"search": query, "page_size": page_size})
    return resp.json().get("results", [])

def iter_filings(company_id=None, filing_type_id=None, page_size=50):
    """
    Générateur qui parcourt tous les filings correspondant aux filtres.
    Suit le champ 'next' renvoyé par l'API.
    """
    params = {"page_size": page_size}
    if company_id is not None:
        params["company_id"] = company_id
    if filing_type_id is not None:
        params["filing_type"] = filing_type_id

    next_url = None
    first = True
    while True:
        if first:
            resp = request_simple("GET", "/filings/", params=params)
            first = False
        else:
            if not next_url:
                break
            resp = request_simple("GET", next_url)  # next_url est une URL complète fournie par l'API
        data = resp.json()
        for item in data.get("results", []):
            yield item
        next_url = data.get("next")
        if not next_url:
            break

def get_filing_markdown(filing_id):
    """Retourne le texte brut (markdown) d'un filing."""
    resp = request_simple("GET", f"/filings/{filing_id}/markdown/")
    return resp.text

def save_text(filename, text):
    """Écrit `text` dans `filename` en UTF-8 (écrase si existe)."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def choose_first(results):
    """Retourne le premier élément de la liste ou None si vide."""
    return results[0] if results else None

def main():
    company_q = "Thales"            # nom cherché (modifiez si besoin)
    type_q = "quarterly"             # type de rapport cherché
    print("1) Recherche de la société :", company_q)
    companies = search_companies(company_q, page_size=5)
    if not companies:
        print(" -> aucune société trouvée.")
        return
    print(" -> sociétés trouvées (extraits):")
    for i, c in enumerate(companies, 1):
        print(f"    {i}. id={c.get('id')} name={c.get('name')}")
    chosen = companies[0]
    print(" -> société choisie :", chosen.get("name"), "id=", chosen.get("id"))

    print("\n2) Recherche du type de rapport :", type_q)
    types = search_filing_types(type_q, page_size=5)
    if not types:
        print(" -> aucun type trouvé.")
        return
    for i, t in enumerate(types, 1):
        print(f"    {i}. id={t.get('id')} name={t.get('name')}")
    chosen_type = types[0]
    print(" -> type choisi :", chosen_type.get("name"), "id=", chosen_type.get("id"))

    print("\n3) Lister quelques filings (maximum 3) pour la société et le type choisis")
    cnt = 0
    first_filing_id = None
    for f in iter_filings(company_id=chosen.get("id"), filing_type_id=chosen_type.get("id"), page_size=20):
        cnt += 1
        print(f"    - filing id={f.get('id')} title={f.get('title')[:60]!s} date={f.get('release_datetime')}")
        if first_filing_id is None:
            first_filing_id = f.get('id')
        if cnt >= 3:
            break
    if cnt == 0:
        print(" -> aucun filing trouvé.")
        return

    print("\n4) Télécharger le markdown du premier filing trouvé (id=", first_filing_id, ")")
    md = get_filing_markdown(first_filing_id)
    filename = f"filing_{first_filing_id}.md"
    save_text(filename, md)
    print(" -> sauvegardé dans :", filename)
    print("\nFin de l'exemple.")

if __name__ == "__main__":
    main()

