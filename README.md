# 🤖 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) chatbot for answering questions from PDF documents using semantic search, FAISS vector database, BGE embeddings, and Groq Llama models.

---

# ✨ Features

- 📄 PDF upload and ingestion
- 🔍 OCR support for scanned PDFs
- 🧹 Text cleaning and normalization
- 🌍 Language detection
- ✂️ Token-aware chunking
- 🧠 BAAI/bge-small-en-v1.5 embeddings
- ⚡ FAISS vector database
- 🎯 Top-K semantic retrieval
- 📚 Citation-aware responses
- 💬 Interactive web interface
- 🚀 FastAPI backend
- 💾 Persistent vector index
- 🔄 Duplicate protection
- 🛡 Retrieval-only fallback mode
- ✅ Modular RAG pipeline

---

# 🏗 Architecture

```text
PDF Upload
     │
     ▼
Text Extraction / OCR
     │
     ▼
Cleaning & Language Detection
     │
     ▼
Chunking (200 tokens, overlap 20)
     │
     ▼
BGE Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
Groq Llama 3.1 8B Instant
     │
     ▼
Answer + Citations
