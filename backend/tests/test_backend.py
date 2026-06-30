import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from pdf_processor import clean_text, extract_pdf_pages
from chunker import chunk_page
from schemas import SearchRequest, SearchResult, SearchResponse, AskRequest, Citation, AskResponse
from main import app, get_db

client = TestClient(app)

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
    engine = EmbeddingEngine.get_instance()
    emb = engine.embed_text("Verify embedding output length")
    assert len(emb) == 384

# ==================== Version 2 RAG Q&A Tests ====================

def test_ask_request_validation():
    """
    Test AskRequest schema validations: empty question, lower/upper top_k bounds.
    """
    # Empty question
    with pytest.raises(ValidationError):
        AskRequest(question="", top_k=5)
        
    # top_k < 1
    with pytest.raises(ValidationError):
        AskRequest(question="What is this?", top_k=0)
        
    # top_k > 10
    with pytest.raises(ValidationError):
        AskRequest(question="What is this?", top_k=11)
        
    # Valid
    req = AskRequest(question="What is this?", top_k=5)
    assert req.question == "What is this?"
    assert req.top_k == 5

def test_citation_format():
    """
    Verify Citation format and required fields.
    """
    cit = Citation(
        filename="manual.pdf",
        page_number=14,
        chunk_index=3,
        similarity=0.887
    )
    assert cit.filename == "manual.pdf"
    assert cit.page_number == 14
    assert cit.chunk_index == 3
    assert cit.similarity == 0.887

def test_rag_context_builder():
    """
    Verify that rag.build_context builds a string that includes
    the filename, page number, chunk index, and chunk text.
    """
    from rag import build_context
    chunks = [
        {
            "filename": "guide.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "chunk_text": "Sample text content."
        }
    ]
    context = build_context(chunks)
    assert "guide.pdf" in context
    assert "Page: 2" in context
    assert "Chunk Index: 0" in context
    assert "Sample text content." in context

@patch("config.settings.openai_api_key", "")
def test_rag_missing_api_key_error():
    """
    Verify generate_rag_answer raises ValueError when OPENAI_API_KEY is missing.
    """
    from rag import generate_rag_answer
    with pytest.raises(ValueError) as exc:
        generate_rag_answer("question", [{"filename": "doc.pdf", "page_number": 1, "chunk_index": 0, "chunk_text": "text"}])
    assert "OPENAI_API_KEY is not configured" in str(exc.value)

@patch("config.settings.openai_api_key", "mock-key")
@patch("openai.resources.chat.completions.Completions.create")
def test_rag_openai_call_mocked(mock_create):
    """
    Test generate_rag_answer runs successfully using a mocked OpenAI client completion call.
    """
    from rag import generate_rag_answer
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked grounded LLM answer."
    mock_response.choices = [mock_choice]
    mock_create.return_value = mock_response
    
    chunks = [{"filename": "policy.pdf", "page_number": 3, "chunk_index": 2, "chunk_text": "reimbursement within 30 days"}]
    answer = generate_rag_answer("What is the refund policy?", chunks)
    
    assert answer == "Mocked grounded LLM answer."
    mock_create.assert_called_once()

@patch("config.settings.openai_api_key", "mock-key")
@patch("config.settings.api_key", "test-key")
def test_ask_checks_document_presence_empty_db():
    """
    Verify /ask endpoint returns error when no documents are indexed.
    """
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    
    async def mock_execute(*args, **kwargs):
        return mock_result
    mock_session.execute = mock_execute
    
    app.dependency_overrides[get_db] = lambda: mock_session
    
    response = client.post(
        "/ask",
        json={"question": "Where is my refund?", "top_k": 5},
        headers={"x-api-key": "test-key"}
    )
    
    assert response.status_code == 400
    assert "No documents have been indexed yet" in response.json()["reason"]
    
    app.dependency_overrides.clear()

@patch("config.settings.openai_api_key", "")
@patch("config.settings.api_key", "test-key")
def test_ask_handles_missing_openai_key():
    """
    Verify /ask returns graceful error when OPENAI_API_KEY is missing.
    """
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 5 # Documents present
    
    async def mock_execute(*args, **kwargs):
        return mock_result
    mock_session.execute = mock_execute
    
    app.dependency_overrides[get_db] = lambda: mock_session
    
    response = client.post(
        "/ask",
        json={"question": "Where is my refund?", "top_k": 5},
        headers={"x-api-key": "test-key"}
    )
    
    assert response.status_code == 400
    assert "OPENAI_API_KEY is not configured" in response.json()["reason"]
    
    app.dependency_overrides.clear()

@patch("config.settings.api_key", "test-key")
def test_search_with_filename_filter_sql():
    """
    Verify that when filename is provided to /search, the database query
    filters results by filename.
    """
    mock_session = MagicMock()
    captured_stmts = []
    
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 1 # Chunks exist
    
    mock_search_result = MagicMock()
    mock_row = MagicMock()
    mock_row.filename = "policy.pdf"
    mock_row.page_number = 3
    mock_row.chunk_index = 2
    mock_row.chunk_text = "refund policy content"
    mock_row.similarity = 0.95
    mock_search_result.all.return_value = [mock_row]
    
    async def mock_execute(stmt, *args, **kwargs):
        captured_stmts.append(stmt)
        if len(captured_stmts) == 1:
            return mock_count_result
        return mock_search_result
        
    mock_session.execute = mock_execute
    app.dependency_overrides[get_db] = lambda: mock_session
    
    response = client.post(
        "/search",
        json={"query": "refund policy", "top_k": 3, "filename": "policy.pdf"},
        headers={"x-api-key": "test-key"}
    )
    
    assert response.status_code == 200
    assert len(captured_stmts) == 2
    
    # Verify the second statement (search query) filters by filename
    search_stmt_str = str(captured_stmts[1])
    assert "documents.filename =" in search_stmt_str
    
    app.dependency_overrides.clear()

@patch("config.settings.api_key", "test-key")
def test_search_without_filename_global_sql():
    """
    Verify that when filename is omitted or null, the database search query
    does NOT filter by filename (global search).
    """
    mock_session = MagicMock()
    captured_stmts = []
    
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 1
    
    mock_search_result = MagicMock()
    mock_search_result.all.return_value = []
    
    async def mock_execute(stmt, *args, **kwargs):
        captured_stmts.append(stmt)
        if len(captured_stmts) == 1:
            return mock_count_result
        return mock_search_result
        
    mock_session.execute = mock_execute
    app.dependency_overrides[get_db] = lambda: mock_session
    
    response = client.post(
        "/search",
        json={"query": "refund policy", "top_k": 3, "filename": None},
        headers={"x-api-key": "test-key"}
    )
    
    assert response.status_code == 200
    assert len(captured_stmts) == 2
    
    # Verify the search query statement does NOT contain the filename filter
    search_stmt_str = str(captured_stmts[1])
    assert "documents.filename =" not in search_stmt_str
    
    app.dependency_overrides.clear()

@patch("config.settings.openai_api_key", "mock-key")
@patch("config.settings.api_key", "test-key")
@patch("openai.resources.chat.completions.Completions.create")
def test_ask_passes_filename_filter_to_retrieval(mock_openai_create):
    """
    Verify that when filename is provided to /ask, the database query
    filters the retrieved chunks by filename.
    """
    mock_session = MagicMock()
    captured_stmts = []
    
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 1
    
    mock_search_result = MagicMock()
    mock_row = MagicMock()
    mock_row.filename = "policy.pdf"
    mock_row.page_number = 3
    mock_row.chunk_index = 2
    mock_row.chunk_text = "refund policy content"
    mock_row.similarity = 0.95
    mock_search_result.all.return_value = [mock_row]
    
    async def mock_execute(stmt, *args, **kwargs):
        captured_stmts.append(stmt)
        if len(captured_stmts) == 1:
            return mock_count_result
        return mock_search_result
        
    mock_session.execute = mock_execute
    app.dependency_overrides[get_db] = lambda: mock_session
    
    # Mock OpenAI
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer content."
    mock_response.choices = [mock_choice]
    mock_openai_create.return_value = mock_response
    
    response = client.post(
        "/ask",
        json={"question": "What is the policy?", "top_k": 3, "filename": "policy.pdf"},
        headers={"x-api-key": "test-key"}
    )
    
    assert response.status_code == 200
    assert len(captured_stmts) == 2
    
    # Verify the retrieval query statement filters by filename
    search_stmt_str = str(captured_stmts[1])
    assert "documents.filename =" in search_stmt_str
    
    app.dependency_overrides.clear()
