# Local PubMed RAG Biomedical Assistant

This project is a local biomedical Retrieval-Augmented Generation (RAG) prototype for Parkinson’s disease genetics and GWAS literature.

## What it does

The system:

1. Fetches PubMed abstracts for a biomedical query.
2. Embeds the abstracts using a sentence-transformer model.
3. Stores embeddings in ChromaDB.
4. Retrieves relevant abstracts for a user question.
5. Sends retrieved evidence to a local Ollama model.
6. Generates an answer with PMID citations.
7. Benchmarks multiple local LLMs.

## Pipeline

PubMed → embeddings → ChromaDB → local Ollama LLM → PMID-grounded answer

## Main scripts

- `scripts/01_fetch_pubmed.py` — fetches PubMed abstracts
- `scripts/02_embed_chroma.py` — creates embeddings and ChromaDB collection
- `scripts/03_retrieve.py` — tests retrieval
- `scripts/04_rag_answer.py` — generates one final RAG answer
- `scripts/05_mutiple_modeltest.py` — compares multiple local models
- `scripts/06_eval_models.py` — evaluates models across multiple questions

## Best current model

`gemma3:4b`

## Example question

What genes are associated with Parkinson risk?

## Example result

The system identified SNCA, LRRK2, VPS35, PRKN, PINK1, DJ1/PARK7, and GBA/GBA1 as Parkinson’s disease risk-associated genes using retrieved PubMed evidence.

## Current status

Working local RAG prototype completed. Next steps are hybrid BM25 + dense retrieval, GWAS interpretation mode, and a simple Streamlit interface
