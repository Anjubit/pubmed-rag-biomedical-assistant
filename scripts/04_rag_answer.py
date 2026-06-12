import chromadb
import ollama
from sentence_transformers import SentenceTransformer

question = "What genes are associated with parkinson risk ?"

OLLAMA_MODEL = "gemma3:4b"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="pubmed_papers")

question_embedding = embedding_model.encode([question])[0]

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=10
)

retrieved_context = ""

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    retrieved_context += (
        f"PMID: {meta['pmid']}\n"
        f"{doc[:2000]}\n\n"
    )

prompt = f"""

You are a biomedical research assistant.

Answer ONLY the question below using ONLY the PubMed abstracts.
Do not write "The following are the answers".
Do not add extra introduction.

Question:
{question}

PubMed abstracts:
{retrieved_context}

Return format:

Summary:
...

Evidence:
- PMID:
- PMID:

"""

response = ollama.chat(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "user", "content": prompt}
    ],
    options={
        "temperature": 0,
        "num_predict": 250,
        "num_ctx": 4096
    }
)

answer = response["message"]["content"]

print(answer)

with open("results/rag_answer.txt", "w") as f:
    f.write(answer + "\n")

with open("results/full_prompt.txt", "w") as f:
    f.write(prompt)

print("\nSaved answer to results/rag_answer.txt")
print("Saved prompt to results/full_prompt.txt")
