import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/pubmed_abstracts.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="pubmed_papers")

for i, row in df.iterrows():
    text = f"Title: {row['title']}\nAbstract: {row['abstract']}"

    embedding = model.encode(text).tolist()

    collection.add(
        ids=[str(row["pmid"])],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "pmid": str(row["pmid"]),
            "title": str(row["title"])
        }]
    )

print("Stored papers in ChromaDB")
print("Number of papers:", collection.count())

