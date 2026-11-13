import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from app.config import Config

logger = logging.getLogger("DevKnow.Embedding")

class EmbeddingService:
    _instance = None
    _local_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.provider = Config.EMBEDDING_PROVIDER  # "LOCAL" or "GEMINI"
        self.local_model_name = Config.EMBEDDING_MODEL
        self.gemini_model_name = Config.GEMINI_EMBEDDING_MODEL
        self._local_model = None
        logger.info(f"Initialized EmbeddingService configuration (Mode: [{self.provider}])")

    @property
    def local_model(self):
        """Lazy loading property: only loads SentenceTransformer when first needed."""
        if self._local_model is None:
            logger.info(f"Lazily loading local embedding model [{self.local_model_name}]...")
            try:
                self._local_model = SentenceTransformer(self.local_model_name)
                logger.info("Local SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load local model {self.local_model_name}: {e}")
                raise
        return self._local_model

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generates embedding vector using either GEMINI mode or LOCAL mode.
        """
        if not text:
            return []

        # Mode 1: Google Gemini text-embedding-004
        if self.provider == "GEMINI" and Config.GEMINI_API_KEY:
            try:
                task_type = "retrieval_query" if is_query else "retrieval_document"
                result = genai.embed_content(
                    model=self.gemini_model_name,
                    content=text,
                    task_type=task_type
                )
                if "embedding" in result:
                    return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini embedding API call failed: {e}. Falling back to local model.")

        # Mode 2: Local SentenceTransformers (Default / Offline / Free)
        try:
            model = self.local_model
            if model:
                embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch document embedding"""
        if not texts:
            return []

        # Gemini mode
        if self.provider == "GEMINI" and Config.GEMINI_API_KEY:
            try:
                result = genai.embed_content(
                    model=self.gemini_model_name,
                    content=texts,
                    task_type="retrieval_document"
                )
                if "embedding" in result:
                    return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini batch embedding failed: {e}. Using local model.")

        # Local mode
        try:
            model = self.local_model
            if model:
                embeddings = model.encode(
                    texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
                )
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            return embeddings.tolist()

        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Search query embedding with is_query=True flag"""
        return self.embed_text(query, is_query=True)

    @property
    def dimension(self) -> int:
        """Returns dimensionality: 768 for Gemini text-embedding-004, 384 for MiniLM"""
        if self.provider == "GEMINI" and Config.GEMINI_API_KEY:
            return 768
        if self._local_model:
            return self._local_model.get_sentence_embedding_dimension()
        return 384

embedding_service = EmbeddingService.get_instance()
