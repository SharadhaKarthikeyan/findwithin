from openai import OpenAI
from config import settings
from typing import List, Dict

def build_context(chunks: List[Dict]) -> str:
    """
    Builds context string from retrieved chunks with source labels.
    """
    context_parts = []
    for idx, chunk in enumerate(chunks):
        source_idx = idx + 1
        part = (
            f"[Source {source_idx}]\n"
            f"Filename: {chunk['filename']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Chunk Index: {chunk['chunk_index']}\n"
            f"Text:\n{chunk['chunk_text']}\n"
        )
        context_parts.append(part)
    return "\n".join(context_parts)

def generate_rag_answer(question: str, chunks: List[Dict]) -> str:
    """
    Sends the question and context to OpenAI to generate a grounded answer.
    Enforces strict grounding bounds.
    """
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is not configured.")
        
    if not chunks:
        return "I could not find this information in the uploaded documents."
        
    context_str = build_context(chunks)
    
    system_prompt = (
        "You are FindWithin, a document question-answering assistant.\n"
        "Answer the user's question using only the provided context.\n"
        "If the answer is not present in the context, say: \"I could not find this information in the uploaded documents.\"\n"
        "Do not make up facts.\n"
        "Keep the answer concise.\n"
        "Include source citations using the provided filename and page number."
    )
    
    user_prompt = (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )
    
    # Modern OpenAI Client instantiation
    client = OpenAI(api_key=settings.openai_api_key)
    
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()
