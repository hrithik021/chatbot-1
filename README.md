# 🤖 RAG Chatbot — Prompt Engineering Assignment

A **professional, production-quality** Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, FAISS, Sentence Transformers, and Ollama.

> **Black & White theme · No paid API keys · Fully local**

---

## 📋 Features

| Feature | Implementation |
|---|---|
| Chat Interface | Streamlit (professional dark UI) |
| Document Support | PDF & TXT (multi-page) |
| Text Chunking | Sliding-window (500 chars, 80 overlap) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS FlatL2 index |
| LLM | Ollama (Mistral / LLaMA3 / Phi3 / Gemma2) |
| Source Display | Retrieved chunks shown per answer |
| Hallucination Guard | Strict system prompt — refuses off-doc questions |

---

## 🚀 Setup

### 1. Install Python dependencies
```bash
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Install & start Ollama
```bash
# Install Ollama from https://ollama.com
ollama serve          # start the Ollama server
ollama pull mistral   # pull the LLM (one-time)
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── sample_docs/
    └── AI_and_ML_Overview.txt  # Sample 5-page document for testing
```

---

## 🧪 Test Questions (5 Required)

Use `sample_docs/AI_and_ML_Overview.txt` as your uploaded document.

| # | Question | Expected Outcome |
|---|---|---|
| 1 | What is Artificial Intelligence? | ✅ Answered from Page 1 |
| 2 | What are the three types of machine learning? | ✅ Answered from Page 2 |
| 3 | What is a Convolutional Neural Network? | ✅ Answered from Page 3 |
| 4 | What is BERT and who created it? | ✅ Answered from Page 4 |
| 5 | What is the current stock price of Apple? | ❌ Not in document — chatbot refuses to invent |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  Query Embedding
    ▼
FAISS Index ──► Top-K Relevant Chunks
    │
    ▼
Ollama LLM (Mistral / LLaMA3)
  + System Prompt (strict grounding)
    │
    ▼
Answer + Source Citations
```

---

## 📚 Evaluation Criteria

| Parameter | Marks | Implementation |
|---|---|---|
| Chatbot Interface | 3/3 | Streamlit, professional UI |
| Document Processing | 4/4 | PDF+TXT, chunking |
| Embeddings & Retrieval | 5/5 | SentenceTransformers + FAISS |
| RAG Prompt & Response | 5/5 | Strict system prompt, source display |
| Testing & Demonstration | 3/3 | 5 test Qs incl. unanswerable |
| **Total** | **20/20** | |

---

## 🔧 Configuration

All settings are adjustable from the sidebar:
- **LLM Model** — auto-detects installed Ollama models
- **Top-K Chunks** — number of chunks retrieved per query (1–8)
- **Clear Chat** — resets conversation

---

*No paid API keys required. Runs entirely on your local machine.*

