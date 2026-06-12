import time
import csv
import re
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

QUESTION = "How is GWAS used in Parkinson's disease?"

OLLAMA_MODELS = [
    "jsk/bio-mistral:latest",
    "qwen3:4b",
    "llama3.2:latest",
    "gemma3:4b",
    "deepseek-r1:8b",
    "qwen2.5:7b",
    "shuai/Bio-Medical-Llama:latest",
]

# Models known to emit <think>...</think> reasoning blocks before the answer.
# These need a larger token budget and/or think disabled, otherwise the
# answer can come back empty.
THINKING_MODELS = {
    "qwen3:4b",
    "qwen3.5:latest",
    "deepseek-r1:8b",
}

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="pubmed_papers")

query_embedding = model.encode(QUESTION).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context_blocks = []

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]

    context_blocks.append(
        f"""
PMID: {meta['pmid']}
Title: {meta['title']}

{doc}
"""
    )

context = "\n\n".join(context_blocks)

prompt = f"""
You are a biomedical research assistant.

Answer the question using ONLY the PubMed context below.

Rules:
1. Use only the provided context.
2. Cite PMIDs when making claims.
3. Do not invent information.
4. If the answer is not in the context, say so.

QUESTION:
{QUESTION}

PUBMED CONTEXT:
{context}

ANSWER FORMAT:

Summary:
(1 short paragraph)

Evidence:
- PMID: xxxx
- PMID: xxxx
"""


def strip_think(text):
    """Remove <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


summary_rows = []

for model_name in OLLAMA_MODELS:
    print("\n" + "=" * 80)
    print(f"Running model: {model_name}")
    print("=" * 80)

    start_time = time.time()

    is_thinking_model = model_name in THINKING_MODELS

    chat_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": 0.2,
            # Thinking models need more headroom: the <think> block eats
            # into num_predict before the real answer is written.
            "num_predict": 900 if is_thinking_model else 400,
            "num_ctx": 4096
        }
    }

    if is_thinking_model:
        chat_kwargs["think"] = False

    try:
        response = ollama.chat(**chat_kwargs)
        raw_answer = response["message"]["content"]
        answer = strip_think(raw_answer)

        if not answer:
            answer = "ERROR: empty response after stripping <think> block."
            status = "empty"
        else:
            status = "success"

    except Exception as e:
        answer = f"ERROR: {e}"
        status = "failed"

    runtime = round(time.time() - start_time, 2)

    safe_name = (
        model_name
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
    )

    output_file = f"results/rag_answer_{safe_name}.txt"

    with open(output_file, "w") as f:
        f.write(answer + "\n")

    summary_rows.append({
        "model": model_name,
        "status": status,
        "runtime_seconds": runtime,
        "output_file": output_file
    })

    print(answer)
    print(f"\nSaved to {output_file}")
    print(f"Runtime: {runtime} seconds")

with open("results/model_test_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "status",
            "runtime_seconds",
            "output_file"
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nDone.")
print("Saved summary to results/model_test_summary.csv")
