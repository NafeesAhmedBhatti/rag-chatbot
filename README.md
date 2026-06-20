# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that ingests PDF documents (native text + scanned via OCR), chunks them with metadata, embeds them with a free open-source model, stores vectors in FAISS, retrieves top-K chunks by similarity, and generates cited answers.

## Requirements

- Python 3.12+
- Tesseract OCR binary (for scanned PDF support)

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install Tesseract (Debian/Ubuntu)
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng

# 3. Configure environment
cp .env.example .env   # then edit values

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

All settings are controlled via environment variables (see `.env.example` and `rag/config.py`).

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment name |
| `APP_DEBUG` | `false` | Enable debug logging |
| `OPENAI_API_KEY` | — | API key for answer-generation LLM |
| `OPENAI_BASE_URL` | — | Base URL for OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SCORE_THRESHOLD` | `0.3` | Min cosine similarity |
| `OCR_CHAR_THRESHOLD` | `20` | Min chars to skip OCR on a page |
| `OCR_DPI` | `300` | Render DPI for image-based pages |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness probe |
| GET | `/api/stats` | Corpus + index statistics |
| GET | `/docs` | Interactive API docs (Swagger) |

## Testing

```bash
pytest
```

## Architecture

See `.drytis/` for full blueprint documents.
