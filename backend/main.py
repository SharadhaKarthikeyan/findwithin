import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from config import settings
from database import get_db, Document
from schemas import SearchRequest, SearchResponse, SearchResult, UploadResponse, AskRequest, AskResponse, Citation
from pdf_processor import extract_pdf_pages
from chunker import chunk_page
from embedding import EmbeddingEngine
from rag import generate_rag_answer

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Verifies that the provided API key matches the configured value."""
    if not api_key_header or api_key_header != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key_header

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the embedding engine once on startup
    # This will load the pre-downloaded sentence-transformers model from cache
    EmbeddingEngine.get_instance()
    yield

app = FastAPI(
    title="FindWithin API",
    description="AI-powered semantic search engine for PDFs",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """
    Custom exception handler to format validation errors.
    In particular, empty or invalid queries/parameters return 400 with status: failed.
    """
    for error in exc.errors():
        loc = error.get("loc", [])
        if "query" in loc:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "reason": "Search query cannot be empty."
                }
            )
        if "question" in loc:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "reason": "Question cannot be empty."
                }
            )
        if "top_k" in loc:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "reason": "top_k must be between 1 and 10."
                }
            )
            
    return JSONResponse(
        status_code=400,
        content={
            "status": "failed",
            "reason": str(exc.errors())
        }
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "FindWithin API"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Uploads a PDF file, extracts and cleans text page-by-page, chunks it,
    generates embeddings, and stores them in PostgreSQL.
    If a file with the same name exists, it is replaced to prevent duplication.
    """
    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "filename": filename,
                "reason": "Only PDF files are supported."
            }
        )
        
    try:
        file_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "filename": filename,
                "reason": f"Failed to read file: {str(e)}"
            }
        )
        
    # Extract text and run scanned PDF detection
    pages_data, is_valid, msg = extract_pdf_pages(file_bytes)
    if not is_valid:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "filename": filename,
                "reason": msg
            }
        )
        
    # Process pages to generate sliding window token chunks
    all_chunks = []
    for page in pages_data:
        page_chunks = chunk_page(
            page["text"],
            page["page_number"],
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens
        )
        all_chunks.extend(page_chunks)
        
    if not all_chunks:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "filename": filename,
                "reason": "No meaningful text could be extracted or chunked from the PDF."
            }
        )
        
    # Generate embeddings
    texts_to_embed = [chunk["chunk_text"] for chunk in all_chunks]
    embeddings = EmbeddingEngine.get_instance().embed_texts(texts_to_embed)
    
    # Delete existing entries with the same filename to avoid duplicates
    delete_stmt = delete(Document).where(Document.filename == filename)
    await db.execute(delete_stmt)
    await db.commit()
    
    # Insert new chunks and embeddings
    db_documents = []
    for chunk, embedding in zip(all_chunks, embeddings):
        db_doc = Document(
            filename=filename,
            page_number=chunk["page_number"],
            chunk_index=chunk["chunk_index"],
            chunk_text=chunk["chunk_text"],
            embedding=embedding
        )
        db_documents.append(db_doc)
        
    db.add_all(db_documents)
    await db.commit()
    
    return {
        "status": "success",
        "filename": filename,
        "pages_processed": len(pages_data),
        "chunks_stored": len(db_documents),
        "message": "PDF processed successfully."
    }

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Embeds the natural language query and performs a pgvector cosine similarity
    search to retrieve the top-k relevant PDF chunks.
    """
    # 1. Pre-search document count check
    count_stmt = select(func.count(Document.id))
    result = await db.execute(count_stmt)
    count = result.scalar()
    
    if count == 0:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "reason": "No documents have been indexed yet. Please upload a PDF first."
            }
        )
        
    # 2. Embed the query text
    query_vector = EmbeddingEngine.get_instance().embed_text(request.query)
    
    # 3. Calculate cosine similarity: 1 - cosine_distance
    similarity_expr = 1 - Document.embedding.cosine_distance(query_vector)
    
    # 4. Perform vector search order by cosine distance ascending
    stmt = (
        select(
            Document.filename,
            Document.page_number,
            Document.chunk_index,
            Document.chunk_text,
            similarity_expr.label("similarity")
        )
        .order_by(Document.embedding.cosine_distance(query_vector))
        .limit(request.top_k)
    )
    
    res = await db.execute(stmt)
    rows = res.all()
    
    results = []
    for row in rows:
        results.append(SearchResult(
            filename=row.filename,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            similarity=float(row.similarity),
            chunk_text=row.chunk_text
        ))
        
    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=results
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Accepts a natural language question, retrieves the top-k relevant PDF chunks
    using pgvector semantic search, builds context, and generates a grounded
    answer with citations using OpenAI.
    """
    # 1. Document presence check
    count_stmt = select(func.count(Document.id))
    result = await db.execute(count_stmt)
    count = result.scalar()
    
    if count == 0:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "reason": "No documents have been indexed yet. Please upload a PDF first."
            }
        )
        
    # 2. Check if OPENAI_API_KEY is configured
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "reason": "OPENAI_API_KEY is not configured. Semantic search is still available, but RAG answer generation requires an LLM API key."
            }
        )
        
    # 3. Retrieve top-k chunks using existing vector search
    query_vector = EmbeddingEngine.get_instance().embed_text(request.question)
    similarity_expr = 1 - Document.embedding.cosine_distance(query_vector)
    
    stmt = (
        select(
            Document.filename,
            Document.page_number,
            Document.chunk_index,
            Document.chunk_text,
            similarity_expr.label("similarity")
        )
        .order_by(Document.embedding.cosine_distance(query_vector))
        .limit(request.top_k)
    )
    
    res = await db.execute(stmt)
    rows = res.all()
    
    if not rows:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "reason": "No relevant context chunks found in the database."
            }
        )
        
    # 4. Format context chunks and prepare citations
    chunks_dict_list = []
    citations = []
    for row in rows:
        chunks_dict_list.append({
            "filename": row.filename,
            "page_number": row.page_number,
            "chunk_index": row.chunk_index,
            "chunk_text": row.chunk_text
        })
        citations.append(Citation(
            filename=row.filename,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
            similarity=float(row.similarity)
        ))
        
    # 5. Call LLM to generate answer
    try:
        answer = generate_rag_answer(request.question, chunks_dict_list)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "reason": f"Failed to generate answer from LLM: {str(e)}"
            }
        )
        
    return AskResponse(
        question=request.question,
        answer=answer,
        citations=citations
    )
