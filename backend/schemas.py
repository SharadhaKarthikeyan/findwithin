from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)

class SearchResult(BaseModel):
    filename: str
    page_number: int
    chunk_index: int
    similarity: float
    chunk_text: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]

class UploadResponse(BaseModel):
    status: str
    filename: str
    pages_processed: Optional[int] = None
    chunks_stored: Optional[int] = None
    message: Optional[str] = None
    reason: Optional[str] = None
