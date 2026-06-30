import os
import time
import requests
import fitz

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "change-me")

def create_synthetic_pdf(page_count: int) -> bytes:
    """Generates a text-based PDF with exactly page_count pages.
    Each page contains enough text to yield 1 chunk."""
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        # Add about 300 words to make it roughly 1 chunk per page
        text = f"This is page {i+1} of the latency benchmark document. " + \
               "We are evaluating semantic search retrieval speeds. " + \
               "The database stores vector embeddings and text chunks. " * 35
        page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def get_stats(latencies):
    if not latencies:
        return 0.0, 0.0
    avg = sum(latencies) / len(latencies)
    sorted_lats = sorted(latencies)
    p95_idx = int(len(sorted_lats) * 0.95)
    p95 = sorted_lats[min(p95_idx, len(sorted_lats)-1)]
    return avg * 1000, p95 * 1000  # Convert to milliseconds

def run_queries(num_queries=20):
    queries = [
        "What is the search latency?",
        "How fast does the HNSW index respond?",
        "Vector database benchmark query",
        "FastAPI search response time",
        "pgvector cosine similarity retrieval",
        "Evaluating semantic query speeds",
        "Performance testing under load",
        "How are documents indexed?",
        "Retrieving the most relevant passages",
        "What is the top_k parameter limit?"
    ]
    # Reuse list if we need more queries
    extended_queries = (queries * (num_queries // len(queries) + 1))[:num_queries]
    
    headers = {"x-api-key": API_KEY}
    latencies = []
    
    for q in extended_queries:
        payload = {"query": q, "top_k": 5}
        start = time.time()
        try:
            res = requests.post(f"{BACKEND_URL}/search", json=payload, headers=headers)
            if res.status_code == 200:
                latencies.append(time.time() - start)
            else:
                print(f"Warning: search request failed with status {res.status_code}")
        except Exception as e:
            print(f"Warning: search request exception: {e}")
            
    return latencies

def run_benchmark():
    print("--- FindWithin Search Latency Benchmark ---")
    headers = {"x-api-key": API_KEY}
    
    # Check health first
    try:
        requests.get(f"{BACKEND_URL}/health")
    except Exception as e:
        print(f"CRITICAL: Cannot reach backend at {BACKEND_URL}. Exception: {e}")
        return
        
    sizes = [100, 500, 1000]
    results = {}
    
    # We will upload documents sequentially to increase corpus size
    current_chunks = 0
    uploaded_files = []
    
    for size in sizes:
        needed_pages = size - current_chunks
        if needed_pages > 0:
            print(f"Generating and uploading PDF with {needed_pages} pages to reach {size} chunks...")
            pdf_bytes = create_synthetic_pdf(needed_pages)
            filename = f"latency_test_{size}.pdf"
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    chunks_added = data.get("chunks_stored", 0)
                    current_chunks += chunks_added
                    uploaded_files.append(filename)
                    print(f"Successfully uploaded. DB now contains approximately {current_chunks} chunks.")
                else:
                    print(f"Error uploading PDF: {res.text}")
                    return
            except Exception as e:
                print(f"Exception during upload: {e}")
                return
                
        print(f"Running {20} search benchmark queries at {current_chunks} chunks...")
        latencies = run_queries(num_queries=20)
        
        if latencies:
            avg_ms, p95_ms = get_stats(latencies)
            results[size] = (avg_ms, p95_ms)
            print(f"Corpus Size: {size} chunks -> Avg Latency: {avg_ms:.2f} ms | p95 Latency: {p95_ms:.2f} ms")
        else:
            print("Failed to collect latency data.")
            
    # Cleanup uploaded files from database
    print("Cleaning up benchmark documents...")
    # By uploading empty PDF or just deleting (uploading non-pdf file deletes it or since we can re-upload to delete)
    # Wait, the best way to delete is if we delete via database or just upload a small pdf with the same filename to minimize size.
    # Actually, we can just leave it, or upload a 1-page PDF for each filename to clean up.
    # Uploading a 1-page PDF with same filename deletes the old massive pages!
    tiny_pdf = create_synthetic_pdf(1)
    for filename in uploaded_files:
        files = {"file": (filename, tiny_pdf, "application/pdf")}
        requests.post(f"{BACKEND_URL}/upload", files=files, headers=headers)
        
    print("\nBenchmark Complete!")
    print("| Chunk Count | Average Latency | p95 Latency |")
    print("|---|---:|---:|")
    for size in sizes:
        if size in results:
            avg_ms, p95_ms = results[size]
            print(f"| {size} chunks | {avg_ms:.2f} ms | {p95_ms:.2f} ms |")

if __name__ == "__main__":
    run_benchmark()
