[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-orange.svg)](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Mistral)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-purple.svg)](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Qwen)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-blue.svg)](https://huggingface.co/serag-ai/datasets/Synthetic-EHR-Llama) 


🚧 **Under Development** 🚧  

# MERA: Medical Electronic Records Assistant

**MERA** (Medical Electronic Records Assistant) is a Retrieval-Augmented Generation (RAG)-based AI system designed to improve electronic health record (EHR) analysis through clinical question answering, medical report summarization, and patient similarity search. It integrates modern LLMs with a domain-specific retrieval pipeline, aiming to deliver accurate, grounded, and explainable insights to healthcare professionals.

This repository contains the code and models introduced in our paper:  
> **"MERA: Medical Electronic Records Assistant"**  
> *Ahmed Ibrahim, Abdullah Khalili, Maryam Arabi, Aamenah Sattar, Abdullah Hosseini, and Ahmed Serag. Machine Learning and Knowledge Extraction (2025)*
> *[Downlad Paper](https://www.mdpi.com/2504-4990/7/3/73)*

---

 <p align="center">
  <img align="center" src="./assets/intro.png" width="800px" alt=" Graphical illustration of the MERA architecture. It comprises three modules: indexing the medical records, retrieve based on the query from the user, and generate response using LLM based on the retrieval document and prompt."/>
 </p>

## 🧠 Features

- **💬 Clinical Question Answering**  
  Answers medical questions about one or more patients based on their health records.

- **📄 Report Summarization**  
  Summarizes medical reports into clear, and structured overviews.

- **🔍 Similarity Search**  
  Finds patients with similar cases to support diagnosis and treatment decisions.

- **📈 Evaluated on Real & Synthetic EHRs**  
  Tested on synthetic data generated via Mistral, Qwen, and LLaMA, as well as real MIMIC-IV de-identified records.

---

## 🛠️ Installation

```bash
git clone https://github.com/serag-ai/MERA.git
cd MERA
pip install -r requirements.txt
```

Ensure you have access to the required models and vector databases (e.g., FAISS, HuggingFace Transformers, SentenceTransformers, OpenAI API, or local LLMs via vLLM).

---

## 🗃️ Datasets

Synthetic datasets are publicly available:

- [Synthetic-EHR-Mistral](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Mistral)
- [Synthetic-EHR-Qwen](https://huggingface.co/datasets/serag-ai/Synthetic-EHR-Qwen)
- [Synthetic-EHR-Llama](https://huggingface.co/serag-ai/datasets/Synthetic-EHR-Llama)

Real EHR evaluation was conducted on de-identified MIMIC-IV-Note dataset: [MIMIC-IV Notes v2.2](https://physionet.org/content/mimic-iv-note/2.2/)


## ⭐ Acknowledgments

- [Langchain](https://github.com/langchain-ai/langchain)
