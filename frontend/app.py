import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
API_KEY = os.getenv("API_KEY", "change-me")

st.set_page_config(
    page_title="FindWithin - PDF Semantic Search",
    page_icon="🔍",
    layout="centered"
)

# Custom Styling for rich visual aesthetics
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background-color: #f9fafb;
        border-left: 5px solid #6366f1;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .result-score {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">FindWithin</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered semantic search inside your PDFs.</p>', unsafe_allow_html=True)

# API Authentication headers
headers = {"x-api-key": API_KEY}

# File Upload Section
st.subheader("1. Upload PDF Document")
uploaded_file = st.file_uploader("Upload a PDF file to index chunks and generate embeddings", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload & Index PDF", use_container_width=True):
        with st.spinner("Extracting text, cleaning, and generating embeddings..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"🎉 Success! {data.get('message', 'PDF processed successfully.')} "
                               f"Stored {data.get('chunks_stored', 0)} chunks from {data.get('pages_processed', 0)} pages.")
                else:
                    try:
                        data = response.json()
                        reason = data.get("reason", "Unknown error")
                    except Exception:
                        reason = response.text
                        
                    if "scanned or image-only" in reason.lower() or "scanned" in reason.lower() or "no extractable text" in reason.lower():
                        st.error("This PDF appears to be scanned or image-only. FindWithin v1 supports text-based PDFs only. OCR support is planned for v2.")
                    else:
                        st.error(f"Upload failed: {reason}")
            except Exception as e:
                st.error(f"Could not connect to backend service at {BACKEND_URL}: {str(e)}")

st.divider()

# Search Section
st.subheader("2. Search inside your documents")
query = st.text_input("Enter natural language query:", placeholder="e.g. What is the refund policy?")
top_k = st.selectbox("Top Results (K)", options=[3, 5, 10], index=1)

if st.button("Search Index", use_container_width=True):
    if not query.strip():
        st.warning("Search query cannot be empty.")
    else:
        with st.spinner("Searching semantic index..."):
            payload = {
                "query": query,
                "top_k": top_k
            }
            try:
                response = requests.post(f"{BACKEND_URL}/search", json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if not results:
                        st.info("No matching passages found.")
                    else:
                        st.write(f"Showing top {len(results)} relevant passages:")
                        for idx, result in enumerate(results):
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">
                                    <span><strong>Result {idx + 1}</strong> | Source: <i>{result['filename']}</i> (Page {result['page_number']})</span>
                                    <span class="result-score">Similarity: {result['similarity']:.4f}</span>
                                </div>
                                <p style="margin: 0; font-size: 0.95rem; color: #1f2937; line-height: 1.5;">
                                    "{result['chunk_text']}"
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    try:
                        data = response.json()
                        reason = data.get("reason", "Unknown error")
                    except Exception:
                        reason = response.text
                    st.error(f"Search failed: {reason}")
            except Exception as e:
                st.error(f"Could not connect to backend service at {BACKEND_URL}: {str(e)}")
