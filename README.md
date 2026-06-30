# FindWithin: AI-Powered PDF Semantic Search Engine

![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b.svg?style=flat&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16_pgvector-blue.svg?style=flat&logo=postgresql&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-all--MiniLM--L6--v2-orange.svg?style=flat)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)

---

**Tech Stack:** FastAPI, Streamlit, PostgreSQL, pgvector, Sentence Transformers, PyMuPDF, Docker, SQLAlchemy (Async), Pytest

---

## Project Overview
FindWithin is an AI-powered semantic search engine for PDFs. Unlike keyword search, FindWithin understands the meaning of a user's query and retrieves the most relevant passages from uploaded documents.

The system uses FastAPI for the backend, Streamlit for the frontend, PyMuPDF for PDF text extraction, Sentence Transformers for local embeddings, PostgreSQL with pgvector for vector storage, async PostgreSQL operations with SQLAlchemy async and asyncpg, and HNSW indexing for faster similarity search.

## Project Background
FindWithin was developed as a complete full-stack portfolio project demonstrating advanced capabilities in semantic search, vector databases, and async microservices. Rather than using third-party paid embedding APIs, it utilizes local embedding models baked into Docker containers to perform fast, offline, and cost-effective text embeddings.

---

## Portfolio Highlights
This project demonstrates proficiency in:
*   **Vector Database Indexing**: Implementing PostgreSQL with the `pgvector` extension and optimizing query latency with a Hierarchical Navigable Small World (HNSW) index using cosine similarity.
*   **Baking ML Models into Build Pipelines**: Configuring Docker build stages to download and cache Hugging Face transformer models (`all-MiniLM-L6-v2`) in `/app/model_cache`, ensuring instant container boot times.
*   **Page-Aware PDF Slicing**: Designing a page-by-page token-based sliding window chunking algorithm using the Transformers `AutoTokenizer`, ensuring that search results map directly back to a specific page number.
*   **Fully Asynchronous Database Layer**: Executing database reads, duplicate deletions, and batch inserts asynchronously using SQLAlchemy with the `asyncpg` driver.
*   **Unit & Integration Test Coverage**: Validating text cleaning, scanned document flags, vector sizing, validation schemas, and REST endpoints using `pytest` and `pytest-asyncio`.

---

## Business Problem
Corporate knowledge repositories, employee handbooks, technical manuals, and policy documents are typically distributed as large PDF files. Finding specific sections inside them via keyword search (e.g. `Ctrl+F`) requires users to know the exact phrasing. If a user searches for "reimbursement guidelines" but the document uses the word "refunds," standard search fails, causing lost productivity.

## Project Objective
To build a semantic indexing and search platform that:
1.  **Extracts and Cleans PDF Text**: Parses multi-page PDF documents page by page while stripping excess whitespace and newlines.
2.  **Identifies Scanned Files**: Automatically flags image-only PDFs containing fewer than 50 characters of text to prevent empty indices.
3.  **Generates Vector Embeddings**: Dynamically maps cleaned token blocks to 384-dimensional dense vectors using a local SentenceTransformer model.
4.  **Enforces API Authentication**: Validates requests with secure headers and rejects unauthorized queries.
5.  **Performs High-Speed Semantic Search**: Returns the top matching text blocks sorted by cosine similarity alongside filename, page number, and chunk identifiers.

---

## Dataset Description
The platform operates on a vector database table mapped through SQLAlchemy, containing a single core table:

*   **Documents Table (`documents`)**:
    *   *Attributes*: `id` (Auto-increment PK), `filename` (Text), `page_number` (Integer), `chunk_index` (Integer), `chunk_text` (Text), `embedding` (Vector of size 384), `created_at` (Timestamp).
    *   *Index*: HNSW cosine similarity index (`documents_embedding_hnsw_idx`) configured on the `embedding` column.

---

## Tools and Technologies
*   **Backend API**: FastAPI, Python 3.11-slim, Pydantic (v2), Pydantic Settings
*   **Vector Engine**: PostgreSQL 16 with `pgvector`
*   **Embeddings & Tokenizer**: SentenceTransformers (`all-MiniLM-L6-v2`), Hugging Face `transformers`
*   **Document Parser**: PyMuPDF (`fitz`)
*   **ORM & Driver**: SQLAlchemy (Async), `asyncpg`, `greenlet`
*   **Frontend UI**: Streamlit, `requests`
*   **Containerization**: Docker, Docker Compose
*   **Testing**: Pytest, Pytest-Asyncio

---

## Folder Structure
```
FindWithin/
├── backend/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                  # Pytest configuration and asyncio loop settings
│   │   └── test_backend.py              # Unit tests for chunking, extraction, and schemas
│   ├── Dockerfile                       # Python 3.11-slim container with pre-baked model cache
│   ├── chunker.py                       # Token-based sliding window chunking algorithm
│   ├── config.py                        # Pydantic environment configurations
│   ├── database.py                      # SQLAlchemy async engine, session, and ORM model
│   ├── embedding.py                     # Singleton wrapper for local SentenceTransformers model
│   ├── main.py                          # FastAPI routers, middleware, and exception handlers
│   ├── pdf_processor.py                 # PyMuPDF parser and scanned PDF detector
│   └── requirements.txt                 # Backend dependency list
├── benchmarks/
│   ├── benchmark_ingestion.py           # Ingestion metrics script (pages, chunks, embeddings/sec)
│   ├── benchmark_search_latency.py      # Search speed metric script (100, 500, 1000 chunks)
│   ├── benchmark_results.md             # Ingestion and latency benchmarks markdown logs
│   └── evaluation_queries.json          # 20 hand-written test queries checking recall accuracy
├── database/
│   └── init.sql                         # PostgreSQL pgvector schemas and HNSW index setup
├── frontend/
│   ├── app.py                           # Streamlit UI dashboard
│   ├── Dockerfile                       # Streamlit deployment container config
│   └── requirements.txt                 # Frontend dependency list
├── .env.example                         # Environment template config
├── .env                                 # Local environment variables
├── .gitignore                           # Git ignore definitions
├── docker-compose.yml                   # Container orchestration spec
└── README.md                            # Documentation
```

---

## Methodology
1.  **Text Clean & Extraction**: PDF streams are parsed page-by-page. Content is stripped of double spacing, tabs, and line break formatting.
2.  **Page-Level Token-Based Slicing**: Cleaned page strings are parsed using a token sliding window (384 token window, 64 token overlap) to ensure no chunk crosses page margins.
3.  **Local Index Ingestion**: Generated token chunks are converted to dense vector formats. Existing records matching the uploaded filename are deleted first to avoid duplication.
4.  **Database Integration**: Chunks are loaded asynchronously in batch mode to postgres.
5.  **Cosine Similarity Matching**: Search prompts are embedded and evaluated against indexed vectors via `1 - (embedding <=> query_embedding)`.

---

## How to Run the Project

### Prerequisites
*   Docker and Docker Compose installed.
*   Python 3.11 (if running tests and benchmarks locally on the host).
*   **OpenAI API Key (Optional)**: Required only for the "Ask with AI" RAG feature. Semantic search functions without it.

### Setup and Deployment
1.  **Build and launch the containers**:
    ```bash
    docker-compose up --build
    ```
    *(Note: The first build will take longer since it downloads the SentenceTransformers weights and bakes them into the backend image).*
2.  **Verify local running services**:
    *   Streamlit UI: [http://localhost:8501](http://localhost:8501)
    *   FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
    *   PostgreSQL: `localhost:5432`
3.  **API Authorization & LLM Settings**: Ensure you set the `x-api-key` header to the key defined in your `.env` (default is `change-me`).
    To enable the RAG Q&A feature, specify these keys in your `.env`:
    ```env
    OPENAI_API_KEY=your-api-key-here
    LLM_MODEL=gpt-4o-mini
    RAG_TOP_K=5
    ```
    *Note: The OpenAI API key is optional. It is only required for the Ask with AI feature. Semantic search works without it.*

### Running Tests
To run unit tests inside the backend container:
```bash
docker-compose exec backend pytest tests/
```

### Running Performance Benchmarks
To run the metrics scripts inside the backend container:
```bash
# Measure document upload and embedding speeds
docker-compose exec backend python benchmarks/benchmark_ingestion.py

# Measure cosine similarity search latencies at 100/500/1000 chunks
docker-compose exec backend python benchmarks/benchmark_search_latency.py
```

---

## Core API Endpoints

### Health Route
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Verify if the API is running |

### Ingestion Route
| Method | Endpoint | Description | Headers |
|---|---|---|---|
| POST | `/upload` | Parse PDF, chunk, embed, and index | `x-api-key: your-secret-api-key` |

### Search Route
| Method | Endpoint | Description | Headers |
|---|---|---|---|
| POST | `/search` | Query semantic search index for top-k results | `x-api-key: your-secret-api-key` |

---

## Demo

![FindWithin Demo](assets/findwithin-demo.webp)

> The demo GIF should show uploading a PDF, successful ingestion, searching a natural language query, and viewing source-aware results.

---

## Benchmarks
See the detailed ingestion speeds and search response times in [benchmarks/benchmark_results.md](file:///c:/Users/karth/OneDrive/Documents/Projects/FindWithin/benchmarks/benchmark_results.md).

## Evaluation
A 20-query evaluation set containing expected pages and search keywords is configured inside [benchmarks/evaluation_queries.json](file:///c:/Users/karth/OneDrive/Documents/Projects/FindWithin/benchmarks/evaluation_queries.json).

---

## Error Handling
FindWithin handles error states gracefully and returns clear messaging:
- **Non-PDF files**: Rejects files without `.pdf` extension with `status: failed`.
- **Scanned PDFs**: If total text extracted is < 50 chars, returns a clear error informing the user that OCR is not supported in v1.
- **Empty Queries**: The Pydantic schema validation rejects empty inputs and returns a structured validation error.
- **Empty Database**: If the documents table contains no data, searches are aborted immediately returning `"reason": "No documents have been indexed yet..."`.
- **Invalid API Key**: Access is rejected with a `401 Unauthorized` response.

---

## Project Limitations
*   **Text-Based PDFs Only**: FindWithin v1 extracts digital text only. Document scans, images, or protected PDF files without embedded texts will trigger the scanned PDF error message.
*   **Memory-Bound ML Model**: Embedding generation runs locally inside the container on CPUs. High concurrency workloads might hit CPU bottlenecks.

## Version 2: RAG With Citations
FindWithin can optionally generate cohesive answers using retrieved PDF chunks as context. The answers include source citations citing the filename and page number. 

If no OpenAI API key is configured, semantic search still works and the Ask with AI interface returns a clear, helpful message informing the user that RAG answer generation requires the key.

## Search Scope (Document Filtering)
By default, FindWithin scopes queries to the **currently uploaded document only** to provide a targeted context. Users can switch the scope to **all indexed documents** globally via the radio selectors on the Streamlit dashboard. This scoping filter applies to both the **Semantic Search** and **Ask with AI (RAG)** features. When the `filename` parameter is omitted or set to null, the API automatically falls back to global search across all documents to preserve backward compatibility.

## Future Improvements
*   **OCR Engine Integration**: Incorporate an OCR framework (like Tesseract) to parse text from scanned images.
*   **Index Administration**: Add deletion routes to let users purge specific documents.

---

---
**Author: Sharadha Karthikeyan**
