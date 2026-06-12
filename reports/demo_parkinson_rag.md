# Local PubMed RAG Assistant for Parkinson’s Disease Genetics

## Project Aim
This project builds a local biomedical RAG assistant for Parkinson’s disease genetics and GWAS literature.

## Pipeline
PubMed abstracts → sentence embeddings → ChromaDB → local Ollama LLM → answer with PMID citations.

## Question
What genes are associated with Parkinson risk?

## Best Model
gemma3:4b

## Result
The system identified several Parkinson’s disease risk genes, including SNCA, LRRK2, VPS35, PRKN, PINK1, DJ1/PARK7, and GBA/GBA1.

## Evidence
- PMID: 35248195
- PMID: 33020390
- PMID: 38173558
- PMID: 32967142

## Model Evaluation
Six local Ollama models were tested across five Parkinson/GWAS questions. Gemma3:4b gave the best practical balance of biomedical term recovery, PMID citation behaviour, and runtime.

## Current Status
The local PubMed RAG pipeline works end-to-end and is ready for the next stage: hybrid retrieval and GWAS interpretation.
