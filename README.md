
# MERA: Medical Electronic Records Assistant

**MERA** (Medical Electronic Records Assistant) is a Retrieval-Augmented Generation (RAG)-based AI system designed to improve electronic health record (EHR) analysis through clinical question answering, medical report summarization, and patient similarity search. It integrates modern LLMs with a domain-specific retrieval pipeline, aiming to deliver accurate, grounded, and explainable insights to healthcare professionals.

🚧 **Under Development** 🚧  
---

## 🔬 Paper



---

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

- [Synthetic-EHR-Mistral](https://huggingface.co/serag-ai/serag-ai/Synthetic-EHR-Mistral)
- [Synthetic-EHR-Qwen](https://huggingface.co/serag-ai/serag-ai/Synthetic-EHR-Qwen)
- [Synthetic-EHR-Llama](https://huggingface.co/serag-ai/serag-ai/Synthetic-EHR-Llama)

Real EHR evaluation was conducted on de-identified MIMIC-IV-Note dataset: [MIMIC-IV Notes v2.2](https://physionet.org/content/mimic-iv-note/2.2/)

---

## 🚀 Usage

The core logic is implemented in [`mera.ipynb`](./mera.ipynb). It includes:

- **Data ingestion** and vector store creation
- **Query pipelines** for:
  - `summarize(patient_id)`
  - `ask_question("What is the treatment plan for patient 23?")`
  - `find_similar_cases(patient_id)`

Modify the notebook or refactor into scripts for production.

---

## 📊 Evaluation Metrics

| Task                | Metrics                          |
|---------------------|----------------------------------|
| Question Answering  | Correctness, Relevance, Groundedness, Retrieval Relevance |
| Summarization       | ROUGE-1, ROUGE-2, Jaccard Similarity, Human Evaluation |
| Similarity Search   | METEOR score on Top-K retrievals |

Performance on MIMIC-IV data:
- QA correctness: 0.92
- Summarization ROUGE-1: 0.45
- Similarity Search METEOR: 1.0

---

## 📚 Architecture Overview

- **RAG Pipeline**: Retrieve → Rank → Generate
- **Embedding Models**: SentenceTransformers (e.g., `MiniLM-L6-v2`)
- **LLMs Used**: Mistral v0.3, Qwen 2.5, LLaMA 3
- **Cross-Encoder Reranking**: Improves chunk retrieval before generation

---

## 👥 Authors

- Ahmed Ibrahim
- Abdullah Khalili
- Maryam Arabi
- Aamenah Sattar
- Abdullah Hosseini
- Ahmed Serag (PI)

Affiliated with **AI Innovation Lab, Weill Cornell Medicine—Qatar**  
📧 afs4002@qatar-med.cornell.edu

---

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
See [LICENSE](https://creativecommons.org/licenses/by/4.0/) for details.

---

## ⭐ Acknowledgments

- MIMIC-IV team for EHR data
- Hugging Face for transformer model hosting
- MDPI and *Machine Learning and Knowledge Extraction* journal for publication support
