import chromadb
from sentence_transformers import SentenceTransformer

QUESTION = "How is GWAS used in neurogenomics?"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="pubmed_papers")

query_embedding = model.encode(QUESTION).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("QUESTION:", QUESTION)
print()

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]

    print("=" * 80)
    print("PMID:", meta["pmid"])
    print("TITLE:", meta["title"])
    print()
    print(doc[:1000])
