import os
from typing import List, Dict
from transformers import AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Initialize the tokenizer
# It will load from cache if SENTENCE_TRANSFORMERS_HOME is set and the model is pre-downloaded
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    # Fallback to local model path if cache directory is structured differently
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_page(text: str, page_number: int, chunk_size: int = 384, chunk_overlap: int = 64) -> List[Dict]:
    """
    Splits text of a single page into token-based windows with overlap.
    Every chunk maps to exactly one page number. Empty chunks are excluded.
    """
    if not text or not text.strip():
        return []
        
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be strictly greater than chunk_overlap")
        
    # Encode text into token IDs without special tokens (like [CLS], [SEP])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(tokens)
    
    if num_tokens == 0:
        return []
        
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        chunk_tokens = tokens[start:end]
        
        # Decode token IDs back to a string
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True).strip()
        
        if chunk_text:
            chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text
            })
            chunk_index += 1
            
        if end == num_tokens:
            break
            
        start += (chunk_size - chunk_overlap)
        
    return chunks
