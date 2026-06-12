# Related Work

This project implements a Retrieval-Augmented Generation (RAG) pipeline for
biomedical literature QA, following an established and active pattern in
biomedical NLP research. Related systems and papers:

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks.* Original RAG paper.

- [Efficient and Reproducible Biomedical Question Answering using
  Retrieval Augmented Generation](https://arxiv.org/abs/2505.07917)
  (IEEE SDS 2025). Closest methodological comparison — PubMed-scale RAG
  with PMID-grounded answers, BM25+MedCPT retrieval comparison.

- [Biomedical Literature Q&A System Using RAG](https://arxiv.org/abs/2509.05505)
  (USC). Integrates PubMed, curated QA datasets, and medical encyclopedias.

- [PubMed Reasoner](https://arxiv.org/abs/2603.27335). Iterative,
  evidence-grounded retrieval with self-critic query refinement — a more
  advanced direction beyond simple top-k retrieval.

- [PubTator 3.0](https://arxiv.org/abs/2401.11048). Demonstrates that
  PubMed-grounded RAG significantly reduces LLM citation hallucination.

- [Queryome](https://www.biorxiv.org/content/10.64898/2025.12.22.696019v1)
  (bioRxiv, 2025). Multi-agent hybrid retrieval over 28.3M PubMed abstracts —
  illustrates a future direction (Stage 5: agentic co-scientist).

- [Hybrid RAG for Medical QA](https://www.mdpi.com/2078-2489/17/2/133).
  BM25 + MedCPT dense retrieval comparison for GPT-4o vs. BioGPT.

- [Clinfo.ai](https://github.com/som-shahlab/Clinfo_ai) / i-MedRAG
  (PSB 2025). Open-source PubMed-grounded clinical QA, iterative retrieval
  for complex multi-hop questions.

## How this project relates

This pipeline is a local, offline implementation of the same general
pattern — PubMed abstracts → vector database → LLM with citation grounding —
adapted for GWAS/neurogenomics use cases and run entirely with open-source
models via Ollama, as Stage 1–2 of a larger agentic bioinformatics workflow.
