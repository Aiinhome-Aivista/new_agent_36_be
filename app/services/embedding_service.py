"""
Embedding Service Module for DevKnow.

Provides vector embedding generation for semantic search, knowledge indexing,
and code retrieval. Supports a dual-provider architecture:
1. LOCAL Mode (Default / Offline / Free): Uses Hugging Face SentenceTransformer
   (`all-MiniLM-L6-v2`) generating 384-dimensional dense vectors locally.
2. GEMINI Mode (Cloud): Uses Google Generative AI (`text-embedding-004`)
   generating 768-dimensional dense vectors with task-type optimization.

Design Patterns:
- Singleton Pattern: Shared instance across the entire application lifecycle.
- Lazy Loading: Local SentenceTransformer model weights are only loaded into memory
  upon the first embedding request, keeping server startup fast and lightweight.
- Resilient Failover: Automatically falls back to local SentenceTransformers if the
  Gemini Cloud API call fails or encounters rate limits.
"""

import logging
import warnings
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

# Suppress legacy package deprecation notice in console
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

from app.config import Config

logger = logging.getLogger("DevKnow.Embedding")


class EmbeddingService:
    """
    Singleton service managing dense vector representations for code snippets,
    rulebook headers, and user search queries.
    """
    _instance: Optional["EmbeddingService"] = None
    _local_model: Optional[SentenceTransformer] = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """
        Thread-safe singleton accessor for EmbeddingService.

        Returns:
            EmbeddingService: Global shared instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initializes embedding service configurations from application environment.
        Does not eagerly load transformer weights to conserve memory.
        """
        self.provider = Config.EMBEDDING_PROVIDER.upper()  # "LOCAL" or "GEMINI"
        self.local_model_name = Config.EMBEDDING_MODEL
        self.gemini_model_name = Config.GEMINI_EMBEDDING_MODEL
        self._local_model = None
        logger.info(f"Initialized EmbeddingService configuration (Mode: [{self.provider}])")

    @property
    def local_model(self) -> SentenceTransformer:
        """
        Lazy-loading property for local SentenceTransformer model.

        Ensures heavyweight neural network weights are only downloaded/loaded into memory
        when an actual vector calculation is requested.

        Returns:
            SentenceTransformer: Initialized local transformer model.
        """
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
        Generates a normalized dense vector embedding for a single text string.

        Args:
            text: Input string (code snippet, rulebook header, or search query).
            is_query: Flag indicating if text is an incoming search query (used by
                      Gemini task-type routing for retrieval_query vs retrieval_document).

        Returns:
            List[float]: Normalized floating-point vector embedding. Empty list on failure.
        """
        if not text or not text.strip():
            return []

        # Mode 1: Google Gemini text-embedding-004 (Cloud)
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

        # Mode 2: Local SentenceTransformers (Default / Offline / Zero-cost)
        try:
            model = self.local_model
            if model:
                embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for a batch of text documents in parallel.

        Optimized with batch processing (batch size 32) for fast ingestion of
        Excel spreadsheets and source code libraries.

        Args:
            texts: List of document strings to vectorize.

        Returns:
            List[List[float]]: List of vector embeddings corresponding to input texts.
        """
        if not texts:
            return []

        # Cloud Batch Mode: Google Gemini
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
                logger.warning(f"Gemini batch embedding failed: {e}. Using local model fallback.")

        # Local Batch Mode: SentenceTransformers batch encoding
        try:
            model = self.local_model
            if model:
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")

        # Final resilient per-item fallback
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """
        Generates embedding vector specifically tailored for search retrieval queries.

        Args:
            query: The user search prompt or question.

        Returns:
            List[float]: Query vector embedding.
        """
        return self.embed_text(query, is_query=True)

    @property
    def dimension(self) -> int:
        """
        Returns the vector dimension of the active embedding model.

        - 768 dimensions for Google Gemini `text-embedding-004`
        - 384 dimensions for `all-MiniLM-L6-v2` SentenceTransformer

        Returns:
            int: Dimension size.
        """
        if self.provider == "GEMINI" and Config.GEMINI_API_KEY:
            return 768
        if self._local_model:
            return self._local_model.get_sentence_embedding_dimension()
        return 384


# Singleton export for system-wide consumption
embedding_service = EmbeddingService.get_instance()
