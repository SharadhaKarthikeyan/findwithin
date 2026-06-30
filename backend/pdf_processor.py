import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple

def clean_text(text: str) -> str:
    """
    Cleans extra spaces, line breaks, and repeated whitespace from the text.
    """
    if not text:
        return ""
    # Replace any sequence of whitespace with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def extract_pdf_pages(file_bytes: bytes) -> Tuple[List[Dict], bool, str]:
    """
    Extracts text page by page from PDF bytes.
    Returns:
        Tuple of:
        - List of dicts: [{"page_number": int, "text": str}]
        - bool: is_valid (True if valid text found, False if scanned/image-only or other failure)
        - str: message or reason for failure
    """
    pages_data = []
    total_length = 0
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return [], False, f"Could not parse PDF file: {str(e)}"
    
    if len(doc) == 0:
        return [], False, "The PDF file is empty."
        
    for i, page in enumerate(doc):
        page_num = i + 1
        try:
            text = page.get_text("text") or ""
            cleaned = clean_text(text)
            total_length += len(cleaned)
            pages_data.append({
                "page_number": page_num,
                "text": cleaned
            })
        except Exception as e:
            # Handle page extraction errors gracefully
            pages_data.append({
                "page_number": page_num,
                "text": ""
            })
            
    # Check for scanned/image-only PDF (fewer than 50 characters across all pages)
    if total_length < 50:
        return [], False, "No extractable text found. This PDF may be scanned or image-only. OCR support is not enabled in version 1."
        
    return pages_data, True, "Success"
