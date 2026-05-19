# MERA: Medical Electronic Records Assistant

<p align="center">
  <img src="assets/banner.png" alt="MERA: Medical Electronic Records Assistant" width="100%">
</p>

<h1 align="center">MERA: Medical Electronic Records Assistant</h1>

<p align="center">
  <a href="https://www.mdpi.com/2504-4990/7/3/73"><img src="https://img.shields.io/badge/MDPI%20MAKE-View%20Paper-orange" alt="Paper"></a>
  <a href="https://huggingface.co/serag-ai"><img src="https://img.shields.io/badge/Hugging%20Face-Datasets-blue" alt="Datasets"></a>
  <a href="http://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/license-CC--BY--4.0-brightgreen" alt="License: CC BY 4.0"></a>
</p>

## Overview

**MERA** is a Retrieval-Augmented Generation (RAG) system for analyzing electronic health records (EHRs). It addresses well-known limitations of standard large language models in healthcare — **hallucinations**, **outdated knowledge**, and **limited explainability** — by combining a domain-specific retrieval pipeline with LLMs to deliver accurate, grounded, and privacy-conscious clinical insights.

To our knowledge, MERA is the **first system to unify clinical question answering, report summarization, and patient similarity search** within a single RAG-based framework. It reaches **0.91 correctness** on question answering, **0.70 ROUGE-1 F1** on summarization, and **0.70–1.00 METEOR** on case similarity, validated on both synthetic EHRs and real MIMIC-IV-Note records.

This repository contains the code and models introduced in our paper:

> **MERA: Medical Electronic Records Assistant**
> Ahmed Ibrahim, Abdullah Khalili, Maryam Arabi, Aamenah Sattar, Abdullah Hosseini, and Ahmed Serag.
> *Machine Learning and Knowledge Extraction*, 7(3):73 (2025). Published 30 July 2025.
> [Read the paper on MDPI](https://www.mdpi.com/2504-4990/7/3/73)

<p align="center">
  <img src="assets/intro.png" alt="Graphical illustration of the MERA architecture: indexing medical records, retrieving documents based on the user query, and generating a response with an LLM grounded in the retrieved documents and prompt." width="800px">
</p>

## Key Features

- **Clinical Question Answering:** Answers medical questions about one or more patients grounded in their health records (single- and multi-patient queries).
- **Report Summarization:** Produces clear, structured summaries of medical reports.
- **Similarity Search:** Finds patients with similar cases to support diagnosis and treatment decisions.
- **Intent-Driven Routing:** Classifies each query and routes it to the appropriate handler (QA, summarization, or similarity).
- **Multi-Source Records:** Resolves patients across Llama (`L`), Mistral (`M`), and Qwen (`Q`) sources via prefix-based identifiers (for example, `L42`, `M5`).
- **Stateful Conversations:** Preserves context across interactions using LangGraph.
- **Validated on Real and Synthetic EHRs:** Evaluated on synthetic data generated with Mistral, Qwen, and LLaMA, and on de-identified MIMIC-IV-Note records.

## Repository Structure

- **`mera.ipynb`** — Main pipeline: record indexing, retrieval and re-ranking, and grounded response generation.
- **`architecture.md`** — Detailed system architecture documentation (intent classification, retrieval, multi-source resolution, execution flows).
- **`requirements.txt`** — Python dependencies.
- **`data/`** — Synthetic and real EHR data (not redistributed).
- **`faiss_index/`** — Persisted FAISS vector index.
- **`assets/`** — Repository media (banner, figures).

## Installation

```bash
git clone https://github.com/serag-ai/MERA.git
cd MERA

python -m venv mera-env
# Windows
mera-env\Scripts\activate
# Linux / macOS
source mera-env/bin/activate

pip install -r requirements.txt
```

Core dependencies: `torch`, `langchain`, `langgraph`, `faiss-cpu`, `sentence-transformers`, and `langchain-openai`. MERA loads credentials from a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_huggingface_token
LANGCHAIN_API_KEY=your_langsmith_key   # optional, for LangSmith tracing
```

Do not commit your `.env` file; ensure it is listed in `.gitignore`.

## Usage

### Querying Records

Open `mera.ipynb` and run the cells in order to initialize the pipeline, then query records:

```python
# Question answering about a single patient
LLMHandler.llm_response("What is the diagnosis for patient L42?")

# Compare multiple patients
LLMHandler.llm_response("Compare patients L1 and M1")

# Summarize a report
LLMHandler.llm_response("Summarize the report for patient Q8")

# Find similar cases
LLMHandler.llm_response("Find patients similar to patient L5")

# Reset conversation state
reset_context()
```

When only a numeric ID is given (for example, `patient 10`), MERA searches all sources, auto-resolves if the ID is unique, and prompts for selection if it is ambiguous.

### Models Used

| Model | Purpose |
|-------|---------|
| `all-mpnet-base-v2` | Query and document embeddings (768-dim) |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranking |
| OpenAI LLM | Answer generation and synthesis |

## Datasets

Synthetic EHR datasets are publicly available on Hugging Face. All were generated with advanced open LLMs and validated for the tasks reported in the paper.

| Dataset | Generator | Hugging Face |
|---------|-----------|--------------|
| Synthetic-EHR-Mistral | Mistral | [serag-ai/Synthetic-EHR-Mistral](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Mistral) |
| Synthetic-EHR-Qwen | Qwen | [serag-ai/Synthetic-EHR-Qwen](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Qwen) |
| Synthetic-EHR-Llama | LLaMA | [serag-ai/Synthetic-EHR-Llama](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Llama) |
| MIMIC-IV-Note v2.2 | Real (de-identified) | [PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/) (credentialed access) |

The MIMIC-IV-Note dataset is governed by the PhysioNet Credentialed Health Data License and is not redistributed in this repository.

## Citation

If you use MERA or this code in your research, please cite:

```bibtex
@article{ibrahim2025mera,
  title   = {MERA: Medical Electronic Records Assistant},
  author  = {Ibrahim, Ahmed and Khalili, Abdullah and Arabi, Maryam and
             Sattar, Aamenah and Hosseini, Abdullah and Serag, Ahmed},
  journal = {Machine Learning and Knowledge Extraction},
  volume  = {7},
  number  = {3},
  pages   = {73},
  year    = {2025},
  publisher = {MDPI},
  doi     = {10.3390/make7030073},
  url     = {https://www.mdpi.com/2504-4990/7/3/73}
}
```

## Acknowledgements

This work builds on open-source efforts from the community, including
[LangChain](https://github.com/langchain-ai/langchain) and
[LangGraph](https://github.com/langchain-ai/langgraph) for retrieval and orchestration, and
[MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) (PhysioNet) for the real-EHR evaluation corpus.

## License

Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/) license.
