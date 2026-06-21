# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) application for question answering over PDF documents using semantic search, FAISS vector storage, BGE embeddings, and Groq-hosted Llama models.

## Features

- PDF ingestion and indexing
- OCR support for scanned documents
- Text cleaning and preprocessing
- Token-based chunking
- Semantic retrieval with FAISS
- BAAI/bge-small-en-v1.5 embeddings
- Citation-aware responses
- Interactive web interface
- Persistent vector storage
- Retrieval fallback mechanism
- FastAPI backend

## Architecture

```text
PDF Upload
    ↓
Text Extraction / OCR
    ↓
Preprocessing
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
LLM (Groq Llama 3.1 8B)
    ↓
Answer + Citations
```

## Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Language | Python |
| PDF Processing | PyMuPDF |
| OCR | Tesseract OCR, RapidOCR |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| LLM | Groq Llama 3.1 8B Instant |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |

## Project Structure

```text
rag-chatbot/
├── app/
├── rag/
├── scripts/
├── tests/
├── requirements.txt
├── README.md
├── pyproject.toml
└── .env.example
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NafeesAhmedBhatti/rag-chatbot.git
cd rag-chatbot
```

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
LLM_MODEL=llama-3.1-8b-instant
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=YOUR_GROQ_API_KEY

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

CHUNK_SIZE=200
CHUNK_OVERLAP=20

TOP_K=5
SCORE_THRESHOLD=0.3
```

## Running the Application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000
```

## API Endpoints

| Endpoint | Method |
|-----------|--------|
| `/api/health` | GET |
| `/api/stats` | GET |
| `/api/documents` | GET |
| `/api/chat` | POST |
| `/api/ingest` | POST |

## Example Queries

```text
What is machine learning?

What is linear regression?

What is K-Nearest Neighbor?

What are the different types of machine learning?

Explain reinforcement learning.

What is Random Forest?
```

## Testing

Run tests:

```bash
pytest
```

## License

This project is intended for educational and research purposes.
