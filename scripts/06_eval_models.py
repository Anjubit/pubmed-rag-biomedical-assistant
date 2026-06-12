import os
import re
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# Settings
# -----------------------------
question = "What genes are associated with Parkinson risk?"

OLLAMA_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

N_RETRIEVE = 10
N_PASS_TO_LLM = 5

os.makedirs("results", exist_ok=True)

# -----------------------------
# Helper: remove thinking output
# -----------------------------
def strip_think(text):
    """
    Removes <think>...</think> blocks if a reasoning model leaks hidden reasoning.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

# -----------------------------
# Load embedding model + Chroma
# -----------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="pubmed_papers")

# -----------------------------
# Retrieve PubMed abstracts
# -----------------------------
question_embedding = embedding_model.encode([question])[0]

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=N_RETRIEVE
)

retrieved_context = ""

for doc, meta in zip(
    results["documents"][0][:N_PASS_TO_LLM],
    results["metadatas"][0][:N_PASS_TO_LLM]
):
    retrieved_context += (
        f"PMID: {meta['pmid']}\n"
        f"Title: {meta['title']}\n"
        f"{doc[:2000]}\n\n"
    )

# -----------------------------
# Build prompt
# -----------------------------
prompt = f"""
You are a biomedical research assistant.

Answer ONLY the question below using ONLY the PubMed abstracts provided.

Rules:
1. Use only the retrieved PubMed abstracts.
2. Cite PMIDs for every important claim.
3. Do not invent genes, mechanisms, diseases, or citations.
4. If the retrieved abstracts do not provide enough evidence, say:
   "The retrieved PubMed abstracts do not provide enough evidence to answer this question."
5. Do not add extra introduction.

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

# -----------------------------
# Ask Ollama
# -----------------------------
try:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0,
            "num_predict": 400,
            "num_ctx": 4096
        },
        think=False
    )
except TypeError:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0,
            "num_predict": 400,
            "num_ctx": 4096
        }
    )

answer = strip_think(response["message"]["content"])

# -----------------------------
# Save outputs
# -----------------------------
print(answer)

with open("results/rag_answer.txt", "w") as f:
    f.write(answer + "\n")

with open("results/full_prompt.txt", "w") as f:
    f.write(prompt)

print("\nSaved answer to results/rag_answer.txt")
print("Saved prompt to results/full_prompt.txt")
