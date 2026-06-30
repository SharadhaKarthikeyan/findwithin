import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from pdf_processor import clean_text, extract_pdf_pages
from chunker import chunk_page
from schemas import SearchRequest, SearchResult, SearchResponse

def test_text_cleaning():
    """
    Test that the clean_text helper replaces newlines, multiple spaces,
    and tab characters with a single space, and strips outer whitespace.
    """
    raw_text = "\n  Hello \t  world!\n  This is a   new line. \n"
    expected = "Hello world! This is a new line."
    assert clean_text(raw_text) == expected
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_token_chunking_per_page():
    """
    Test that chunking correctly tokenizes text per page, respects sizes,
    and keeps chunks restricted to the single input page.
    """
    # A text with some repeated content to make it long enough
    page_text = "Sample sentence " * 50
    page_num = 3
    
    chunks = chunk_page(page_text, page_number=page_num, chunk_size=30, chunk_overlap=10)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["page_number"] == page_num
        assert len(chunk["chunk_text"]) > 0
        assert "chunk_index" in chunk

def test_scanned_pdf_detection(monkeypatch):
    """
    Test that extract_pdf_pages detects scanned PDFs (total text under 50 characters)
    and returns the correct failure status.
    """
    # Mock PyMuPDF's fitz.open to return a document with short text
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Hello"  # Only 5 characters
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__len__.return_value = 1
    
    import fitz
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: mock_doc)
    
    pages_data, is_valid, msg = extract_pdf_pages(b"dummy pdf content")
    assert not is_valid
    assert "scanned or image-only" in msg

def test_pydantic_schema_validation():
    """
    Test that:
    1. Empty search queries are rejected.
    2. top_k is validated to be between 1 and 10.
    """
    # Empty query should raise ValidationError
    with pytest.raises(ValidationError):
        SearchRequest(query="", top_k=5)
        
    # top_k < 1 should raise ValidationError
    with pytest.raises(ValidationError):
        SearchRequest(query="refund policy", top_k=0)
        
    # top_k > 10 should raise ValidationError
    with pytest.raises(ValidationError):
        SearchRequest(query="refund policy", top_k=11)
        
    # Valid model setup
    req = SearchRequest(query="refund policy", top_k=3)
    assert req.query == "refund policy"
    assert req.top_k == 3

def test_search_response_format():
    """
    Test that SearchResult and SearchResponse formats contain all expected fields.
    """
    result = SearchResult(
        filename="terms.pdf",
        page_number=2,
        chunk_index=0,
        similarity=0.895,
        chunk_text="You can request reimbursement inside 30 days."
    )
    
    response = SearchResponse(
        query="reimbursement",
        top_k=1,
        results=[result]
    )
    
    assert response.query == "reimbursement"
    assert len(response.results) == 1
    assert response.results[0].filename == "terms.pdf"
    assert response.results[0].page_number == 2
    assert response.results[0].chunk_index == 0
    assert response.results[0].similarity == 0.895
    assert response.results[0].chunk_text == "You can request reimbursement inside 30 days."

def test_embedding_dimension():
    """
    Verify that the EmbeddingEngine generates vectors of size 384.
    """
    from embedding import EmbeddingEngine
    # The EmbeddingEngine should load the model and return a 384 dimensional list
    engine = EmbeddingEngine.get_instance()
    emb = engine.embed_text("Verify embedding output length")
    assert len(emb) == 384
