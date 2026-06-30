import os
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingEngine:
    """
    Singleton class to manage the SentenceTransformer embedding model.
    Loads the model once on startup and keeps it in memory.
    """
    _instance = None

    def __init__(self):
        if EmbeddingEngine._instance is not None:
            raise Exception("EmbeddingEngine is a singleton! Use get_instance().")
            
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        # Loads from SENTENCE_TRANSFORMERS_HOME environment variable path automatically
        self.model = SentenceTransformer(model_name)
        EmbeddingEngine._instance = self

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """
        Generates embedding for a single text string.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings in batch.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
