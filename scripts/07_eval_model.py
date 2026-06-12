import os
import re
import csv
import time
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

os.makedirs("results/eval_answers", exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MODELS = [
    "qwen2.5:7b",
    "qwen3:4b",
    "jsk/bio-mistral:latest",
    "llama3.2:latest",
    "gemma3:4b",
    "deepseek-r1:8b"
]

EVAL_QUESTIONS = [
    {
        "id": "pd_genes",
        "question": "What genes are associated with Parkinson risk?",
        "expected_terms": ["SNCA", "LRRK2", "VPS35", "PRKN", "PINK1", "DJ1", "GBA"],
        "expected_pmids": ["35248195", "33020390"]
    },
    {
        "id": "pd_gwas_use",
        "question": "How is GWAS used in Parkinson's disease?",
        "expected_terms": ["GWAS", "risk", "loci", "polygenic", "genetic"],
        "expected_pmids": []
    },
    {
        "id": "gba1_pd",
        "question": "What is the role of GBA1 in Parkinson's disease risk?",
        "expected_terms": ["GBA1", "risk", "Parkinson"],
        "expected_pmids": []
    },
    {
        "id": "prs_pd",
        "question": "How are polygenic risk scores used in Parkinson's disease?",
        "expected_terms": ["polygenic risk score", "PRS", "risk"],
        "expected_pmids": []
    },
    {
        "id": "snca_pd",
        "question": "What is the role of SNCA in Parkinson's disease genetics?",
        "expected_terms": ["SNCA", "Parkinson", "risk"],
        "expected_pmids": []
    }
]


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def contains_term(answer, term):
    return term.lower() in answer.lower()


def ask_rag(question, model_name, collection, embedding_model):
    question_embedding = embedding_model.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=8
    )

    context_blocks = []

    for doc, meta in zip(results["documents"][0][:5], results["metadatas"][0][:5]):
        context_blocks.append(
            f"""
PMID: {meta['pmid']}
Title: {meta['title']}

{doc[:2000]}
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
4. If the answer is not in the context, say:
   "The retrieved PubMed abstracts do not provide enough evidence to answer this question."

QUESTION:
{question}

PUBMED CONTEXT:
{context}

ANSWER FORMAT:

Summary:
(1 short paragraph)

Evidence:
- PMID:
- PMID:
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 400,
                "num_ctx": 4096
            },
            think=False
        )
    except TypeError:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 400,
                "num_ctx": 4096
            }
        )

    return strip_think(response["message"]["content"])


def main():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="pubmed_papers")

    rows = []

    for model_name in MODELS:
        print("\n" + "=" * 80)
        print("MODEL:", model_name)
        print("=" * 80)

        for item in EVAL_QUESTIONS:
            qid = item["id"]
            question = item["question"]

            print(f"\nQuestion: {question}")

            start = time.time()

            try:
                answer = ask_rag(question, model_name, collection, embedding_model)
                status = "success"
            except Exception as e:
                answer = f"ERROR: {e}"
                status = "failed"

            runtime = round(time.time() - start, 2)

            expected_terms = item["expected_terms"]
            expected_pmids = item["expected_pmids"]

            term_hits = [term for term in expected_terms if contains_term(answer, term)]
            pmid_hits = [pmid for pmid in expected_pmids if pmid in answer]

            term_score = len(term_hits) / len(expected_terms) if expected_terms else 0
            pmid_score = len(pmid_hits) / len(expected_pmids) if expected_pmids else "NA"

            safe_model = model_name.replace("/", "_").replace(":", "_").replace(".", "_")
            answer_file = f"results/eval_answers/{qid}_{safe_model}.txt"

            with open(answer_file, "w") as f:
                f.write(answer + "\n")

            rows.append({
                "question_id": qid,
                "model": model_name,
                "status": status,
                "runtime_seconds": runtime,
                "term_score": term_score,
                "pmid_score": pmid_score,
                "term_hits": "; ".join(term_hits),
                "pmid_hits": "; ".join(pmid_hits),
                "answer_file": answer_file
            })

            print("Status:", status)
            print("Runtime:", runtime)
            print("Term score:", term_score)
            print("PMID score:", pmid_score)

    with open("results/eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "model",
                "status",
                "runtime_seconds",
                "term_score",
                "pmid_score",
                "term_hits",
                "pmid_hits",
                "answer_file"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print("Saved evaluation summary to results/eval_summary.csv")


if __name__ == "__main__":
    main()
