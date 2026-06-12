import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd

QUERY = "GWAS Parkinson"
MAX_RESULTS = 1000
BATCH_SIZE = 100

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 1. Search PubMed IDs
search_params = {
    "db": "pubmed",
    "term": QUERY,
    "retmax": MAX_RESULTS,
    "retmode": "json"
}

search_response = requests.get(ESEARCH_URL, params=search_params, timeout=30)
search_response.raise_for_status()

ids = search_response.json()["esearchresult"]["idlist"]
print(f"Found {len(ids)} PubMed IDs")

papers = []

# 2. Fetch abstracts in batches
for start in range(0, len(ids), BATCH_SIZE):
    batch_ids = ids[start:start + BATCH_SIZE]
    print(f"Fetching {start + 1} to {start + len(batch_ids)}")

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(batch_ids),
        "retmode": "xml"
    }

    fetch_response = requests.get(EFETCH_URL, params=fetch_params, timeout=60)

    if not fetch_response.text.strip():
        print("Empty response. Skipping batch.")
        continue

    try:
        root = ET.fromstring(fetch_response.text)
    except ET.ParseError:
        print("XML parse error. Skipping batch.")
        continue

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            ["".join(part.itertext()) for part in abstract_parts]
        )

        if abstract.strip():
            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract
            })

    time.sleep(0.4)

df = pd.DataFrame(papers)
df.to_csv("data/pubmed_abstracts.csv", index=False)

print(f"Saved {len(df)} papers to data/pubmed_abstracts.csv")
