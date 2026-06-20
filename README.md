# 🤖 RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot for PDF Question Answering with OCR support, semantic search, citation-based responses, and an interactive web interface.

---

## ✨ Features

- 📄 Native PDF text extraction
- 🔍 OCR support for scanned PDFs (Tesseract + RapidOCR)
- 🧹 Text cleaning and normalization
- 🌍 Language detection
- ✂️ Token-aware chunking with metadata
- 🧠 BAAI/bge-small-en-v1.5 embeddings
- ⚡ FAISS vector database
- 🎯 Top-K semantic retrieval with similarity scores
- 📚 Citation-aware responses (filename + page number)
- 💬 Interactive chat interface
- 📤 PDF upload support
- 💾 Persistent storage and duplicate protection
- 🚀 FastAPI backend
- ✅ Extensive automated tests

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
Chunking (512 tokens, 64 overlap)
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
LLM (MiniMax-M3)
     │
     ▼
Answer + Citations
```

---

# 🛠 Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Language | Python |
| PDF Parsing | PyMuPDF |
| OCR | Tesseract OCR + RapidOCR |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | FAISS |
| LLM | MiniMax-M3 |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |

---

# 📂 Project Structure

```text
rag-chatbot/
│
├── app/
├── rag/
├── scripts/
├── tests/
│
├── README.md
├── requirements.txt
├── DATASET.md
├── pyproject.toml
└── .env.example
```

---

# 🚀 Installation

```bash
git clone https://github.com/NafeesAhmedBhatti/rag-chatbot.git

cd rag-chatbot

pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000
```

---

# 🔌 API Endpoints

| Endpoint | Method | Purpose |
|-----------|--------|----------|
| /api/health | GET | Health status |
| /api/stats | GET | Corpus statistics |
| /api/chat | POST | Ask questions |
| /api/ingest | POST | Upload PDF |

---

# 📚 Dataset

The system supports large PDF collections including:

- Machine Learning textbooks
- Research papers
- OCR demonstration PDFs
- User uploaded PDFs

All documents are automatically chunked and indexed.

---

# 🔎 OCR Support

Scanned PDFs are supported using:

- Tesseract OCR
- RapidOCR fallback

Both image-only and native PDFs are handled automatically.

---

# 🧠 RAG Pipeline

1. Extract text from PDF.
2. Clean and normalize content.
3. Split into chunks.
4. Generate embeddings.
5. Store vectors in FAISS.
6. Retrieve top-k relevant chunks.
7. Generate grounded answers.
8. Return citations with page numbers.

---

# 💡 Example Questions

```text
What is attention mechanism?

How does BERT differ from GPT?

Explain Adam optimizer.

What is gradient descent?

Explain convolutional neural networks?
```

---

# ⚡ Performance

- Fast semantic retrieval
- Similarity score ranking
- Persistent vector index
- Scalable architecture
- Supports large PDF collections

---

# 🧪 Testing

Comprehensive test suite includes:

- API tests
- OCR tests
- Chunking tests
- Embedding tests
- FAISS tests
- Generator tests
- Retriever tests

✅ All tests passing

---

# 🔮 Future Improvements

- Hybrid search (BM25 + Dense Retrieval)
- Streaming responses
- Multilingual support
- Conversation memory
- Graph RAG
- Re-ranking models

---

# 🤝 Contributing

Pull requests and improvements are welcome.

---

# 📜 License

This project was developed for educational, research, and hackathon purposes.

---

⭐ If you found this project useful, please give it a star.
