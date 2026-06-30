import os
import time
import requests
import fitz  # PyMuPDF is used to generate the synthetic PDF

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "change-me")

def create_synthetic_pdf(page_count: int = 20) -> bytes:
    """Generates a multi-page text-based PDF using PyMuPDF."""
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        # Insert some repetitive and semantically meaningful text
        text = f"This is page {i+1} of the benchmark document. " + \
               "The refund policy states that customers may request reimbursement within 30 days of purchase. " + \
               "Eligible candidates must submit their application before the deadline. " + \
               "We secure user accounts with encryption and multi-factor authentication. " * 15
        page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def run_benchmark():
    print("--- FindWithin Ingestion Benchmark ---")
    page_count = 20
    print(f"Generating synthetic PDF with {page_count} pages...")
    pdf_data = create_synthetic_pdf(page_count)
    
    filename = f"benchmark_{page_count}p.pdf"
    
    headers = {"x-api-key": API_KEY}
    files = {"file": (filename, pdf_data, "application/pdf")}
    
    print(f"Uploading {filename} to {BACKEND_URL}/upload ...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BACKEND_URL}/upload", files=files, headers=headers)
    except Exception as e:
        print(f"CRITICAL: Failed to connect to the backend at {BACKEND_URL}. Is it running?")
        print(f"Error: {e}")
        return
        
    duration = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        chunks_stored = data.get("chunks_stored", 0)
        pages_processed = data.get("pages_processed", 0)
        
        print("\n--- Ingestion Metrics ---")
        print(f"Status: {data.get('status')}")
        print(f"Pages Processed: {pages_processed}")
        print(f"Chunks Generated: {chunks_stored}")
        print(f"Total Ingestion Time: {duration:.4f} seconds")
        
        pages_per_sec = pages_processed / duration if duration > 0 else 0
        chunks_per_sec = chunks_stored / duration if duration > 0 else 0
        embeddings_per_sec = chunks_stored / duration if duration > 0 else 0 # 1 embedding per chunk
        
        print(f"Pages/sec: {pages_per_sec:.2f}")
        print(f"Chunks/sec: {chunks_per_sec:.2f}")
        print(f"Embeddings/sec: {embeddings_per_sec:.2f}")
    else:
        print(f"Failed to ingest document. Status code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    run_benchmark()
