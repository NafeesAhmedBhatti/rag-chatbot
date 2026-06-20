# Dataset Provenance & Technical Interview Guide

## Corpus Overview

| Metric | Value |
|--------|-------|
| Total PDFs | 12 |
| Total pages | 10,266 |
| Total chunks | 16,137 |
| Index size | 23.76 MB |
| Embedding dimensions | 384 |
| OCR pages processed | 27 (across 3 PDFs) |
| Ingestion time | 34.5 minutes |

**Every PDF is 200+ pages.** The smallest is 212 pages; the largest is 2,118 pages. This exceeds the challenge requirement of "10+ PDFs each 200+ pages."

## Dataset Sources

All PDFs are real publications from established academic and literary sources.

### Computer Science Textbooks (7 PDFs)

| # | Filename | Source | Topic | Pages | Chunks |
|---|----------|--------|-------|-------|--------|
| 1 | `algorithm_design_manual.pdf` | Steven Skiena, "The Algorithm Design Manual" (Springer) | Algorithm design, complexity, data structures | 742 | 1,166 |
| 2 | `cryptography_network_security.pdf` | William Stallings, "Cryptography and Network Security" (Pearson) | Cryptography, network security protocols | 261 | 970 |
| 3 | `database_system_concepts.pdf` | Silberschatz, Korth, Sudarshan, "Database System Concepts" (McGraw-Hill) | Relational databases, SQL, transactions, indexing | 1,376 | 2,206 |
| 4 | `dive_into_deep_learning.pdf` | Zhang, Lipton, Li, Smola, "Dive into Deep Learning" (d2l.ai, open-source) | Deep learning, neural networks, CNNs, RNNs, attention | 2,114 | 2,133 |
| 5 | `intro_hpc.pdf` | "Introduction to High Performance Computing" (open textbook) | HPC, parallel computing, MPI, OpenMP | 510 | 766 |
| 6 | `open_data_structures.pdf` | Pat Morin, "Open Data Structures" (ODS, open-source) | Data structures: arrays, linked lists, trees, hash tables | 325 | 385 |
| 7 | `os_three_easy_pieces.pdf` | Arpaci-Dusseau, "Operating Systems: Three Easy Pieces" (pages.cs.wisc.edu, open-source) | OS concepts: virtualization, concurrency, persistence | 609 | 1,008 |

### Classic Literature from Project Gutenberg (4 PDFs)

| # | Filename | Source | Topic | Pages | Chunks |
|---|----------|--------|-------|-------|--------|
| 8 | `gutenberg_count_of_monte_cristo.pdf` | Project Gutenberg (gutenberg.org) | Alexandre Dumas, "The Count of Monte Cristo" (1844) | 1,122 | 1,935 |
| 9 | `gutenberg_les_miserables.pdf` | Project Gutenberg | Victor Hugo, "Les Misérables" (1862) | 1,338 | 2,232 |
| 10 | `gutenberg_moby_dick.pdf` | Project Gutenberg | Herman Melville, "Moby-Dick" (1851) | 406 | 806 |
| 11 | `gutenberg_war_and_peace.pdf` | Project Gutenberg | Leo Tolstoy, "War and Peace" (1869) | 1,201 | 2,307 |

### Statistics / Data Science (1 PDF)

| # | Filename | Source | Topic | Pages | Chunks |
|---|----------|--------|-------|-------|--------|
| 12 | `think_bayes.pdf` | Allen Downey, "Think Bayes" (greenteapress.com, open-source) | Bayesian statistics, probability, Python | 212 | 223 |

## Source Distribution

| Source | Count | Pages |
|--------|-------|-------|
| CS Textbooks (open/published) | 7 | 5,937 |
| Project Gutenberg | 4 | 4,067 |
| Statistics textbook | 1 | 212 |

## How the System Handles Scale

- **16,137 vectors** in a FAISS IndexFlatIP (exact cosine similarity)
- **Retrieval latency**: ~4.3s at 16K vectors (linear scan). At the original ~600 vectors, latency was ~23ms. For production scale beyond 50K vectors, switching to `IndexIVFFlat` would restore sub-100ms search with minimal code changes.
- **Memory**: ~24 MB index + ~3 MB metadata JSON = ~27 MB total — well within the container's 5.9 GB RAM.
- **Ingestion**: 10,266 pages processed in 34.5 minutes (~5 pages/second including embedding). No errors, no memory issues.

## Technology Stack (All Open Source)

| Component | Technology | License |
|-----------|-----------|---------|
| PDF extraction | PyMuPDF (fitz) 1.27 | AGPL/Commercial |
| OCR | Tesseract 5.5 + RapidOCR fallback | Apache 2.0 |
| Text cleaning | Custom (Unicode NFC, ligatures, de-hyphenation) | — |
| Language detection | langdetect 1.0.9 | MIT |
| Tokenizer | HuggingFace tokenizers (BGE) | Apache 2.0 |
| Chunking | Custom token-aware recursive splitter (512 tokens / 64 overlap) | — |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) | MIT |
| Vector DB | FAISS-cpu (IndexFlatIP) | MIT |
| LLM (answer gen) | drytis/MiniMax-M3 via OpenAI-compatible API | Open weights |
| Backend | FastAPI + Uvicorn | MIT |
| Frontend | Vanilla HTML/CSS/JS | — |

## Demo Questions for Interview

**CS / Data Structures:**
- "What is a B-tree?" → cites database_system_concepts.pdf + open_data_structures.pdf
- "Explain dynamic programming" → cites algorithm_design_manual.pdf

**Deep Learning:**
- "What is backpropagation?" → cites dive_into_deep_learning.pdf
- "Explain the attention mechanism" → cites dive_into_deep_learning.pdf

**Literature:**
- "Who is Jean Valjean?" → cites gutenberg_les_miserables.pdf
- "What is the theme of Moby-Dick?" → cites gutenberg_moby_dick.pdf

**Cryptography:**
- "How does RSA encryption work?" → cites cryptography_network_security.pdf

**Operating Systems:**
- "What is virtual memory?" → cites os_three_easy_pieces.pdf
