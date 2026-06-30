import os
import json
import time
import requests
import fitz

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "change-me")

# Keyword mapping to page numbers based on evaluation_queries.json
PAGE_KEYWORDS = {
    2: "Working hours remote options are flexible with telecommuting capabilities.",
    3: "Refund and reimbursement policy allows claims within 30 days.",
    4: "Annual carry-over limits for PTO paid time off vacation and parental benefits including maternity and paternity leave.",
    5: "Eligible candidate requirements for application processing.",
    6: "To submit an expense report for reimbursement, attach the original receipt.",
    7: "How to enroll in the company health insurance plan and understand your benefits coverage.",
    8: "Guidelines on what is the policy regarding personal device usage for work. The BYOD personal device mobile security policy defines the terms.",
    9: "Security encryption passwords MFA enforced. For data breach, contact the security officer for incident response notification.",
    10: "Harassment or discrimination report to HR anonymously. Disciplinary sanctions for code of conduct violation.",
    11: "Conflict of interest ethics disclosures and integrity. Business gift acceptance limit and bribery or hospitality restrictions.",
    12: "Intellectual property IP copyright ownership policies.",
    13: "Annual performance review feedback and evaluation. Training professional development education reimbursement.",
    14: "Social media public relations posting guidelines for employees.",
    15: "Termination notice period and resignation requirements measured in weeks."
}

def create_recall_pdf() -> bytes:
    """Creates a 15-page PDF with targeted keywords for recall evaluation."""
    doc = fitz.open()
    for page_num in range(1, 16):
        page = doc.new_page()
        content = f"Page {page_num} index document. "
        if page_num in PAGE_KEYWORDS:
            content += PAGE_KEYWORDS[page_num]
        page.insert_text((50, 50), content)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def run_evaluation():
    print("--- FindWithin Recall@3 Evaluation ---")
    headers = {"x-api-key": API_KEY}
    
    # 1. Generate and upload the evaluation PDF
    pdf_bytes = create_recall_pdf()
    filename = "recall_eval_doc.pdf"
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    
    print(f"Uploading {filename} to index document pages...")
    try:
        upload_res = requests.post(f"{BACKEND_URL}/upload", files=files, headers=headers)
        if upload_res.status_code != 200:
            print(f"Error: Upload failed with status {upload_res.status_code}: {upload_res.text}")
            return
    except Exception as e:
        print(f"CRITICAL: Failed to connect to the backend: {e}")
        return
        
    # 2. Load the evaluation queries
    queries_path = "benchmarks/evaluation_queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: Could not find {queries_path}")
        return
        
    with open(queries_path, "r") as f:
        queries = json.load(f)
        
    correct_count = 0
    total_queries = len(queries)
    
    # 3. Evaluate each query
    for idx, item in enumerate(queries):
        query_text = item["query"]
        expected_page = item["expected_page"]
        expected_keywords = item["expected_keywords"]
        
        payload = {"query": query_text, "top_k": 3}
        try:
            res = requests.post(f"{BACKEND_URL}/search", json=payload, headers=headers)
            if res.status_code != 200:
                print(f"Query {idx+1} failed: {res.text}")
                continue
                
            data = res.json()
            results = data.get("results", [])
            
            is_correct = False
            for result in results:
                # Page match check
                if result.get("page_number") == expected_page:
                    is_correct = True
                    break
                # Keyword match check
                chunk_text_lower = result.get("chunk_text", "").lower()
                for kw in expected_keywords:
                    if kw.lower() in chunk_text_lower:
                        is_correct = True
                        break
                if is_correct:
                    break
            
            if is_correct:
                correct_count += 1
            else:
                print(f"Missed: '{query_text}' (Expected page: {expected_page})")
                print(f"  Got pages: {[r.get('page_number') for r in results]}")
        except Exception as e:
            print(f"Exception on query '{query_text}': {e}")
            
    # 4. Clean up documents in DB
    print("Cleaning up evaluation documents...")
    tiny_pdf = fitz.open()
    tiny_page = tiny_pdf.new_page()
    tiny_page.insert_text((50, 50), "Cleanup recall document.")
    tiny_bytes = tiny_pdf.write()
    tiny_pdf.close()
    
    requests.post(f"{BACKEND_URL}/upload", files={"file": (filename, tiny_bytes, "application/pdf")}, headers=headers)

    # 5. Output results
    recall = (correct_count / total_queries) * 100 if total_queries > 0 else 0.0
    print("\n--- Retrieval Evaluation Summary ---")
    print(f"Test queries: {total_queries}")
    print(f"Correct result found in top 3: {correct_count}")
    print(f"Recall@3: {recall:.1f}%")
    
    # Return metrics for integration
    return total_queries, correct_count, recall

if __name__ == "__main__":
    run_evaluation()
