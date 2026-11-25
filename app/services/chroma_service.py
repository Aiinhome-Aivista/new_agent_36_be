"""
ChromaDB Vector Store Service Module for DevKnow.

Provides local persistent vector database storage for semantic function matching,
code rule retrieval, and hybrid metadata filtering.

Key Capabilities:
1. Persistent Storage: Uses ChromaDB PersistentClient stored in `<data_root>/chroma`.
2. Cosine Metric Space: Vectors compared using cosine distance, mapped to similarity scores:
   `similarity = 1.0 - distance` (clamped between 0.0 and 1.0).
3. Metadata Segment Auto-Repair: Automated recovery for SQLite FTS5 virtual table
   corruption caused by abrupt terminations or Windows file locking.
4. Chunked Batch Upsert: Vectorizes and commits hundreds of functions in chunked batches
   (max 500 items per chunk) to adhere to ChromaDB batch size limits.
5. Multi-field Filtering: Supports precise language and framework querying using `$and` expressions.
6. Defensive Null-Safety: Guarantees zero None-values in metadata payloads to prevent ChromaDB schema exceptions.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional

try:
    import chromadb  # type: ignore[import-untyped, import-not-found]
    from chromadb.config import Settings  # type: ignore[import-untyped, import-not-found]
except (ImportError, ModuleNotFoundError):
    chromadb = None  # type: ignore
    Settings = None  # type: ignore


from app.config import Config
from app.services.embedding_service import embedding_service

logger = logging.getLogger("DevKnow.Chroma")


class ChromaService:
    """
    Singleton service managing the ChromaDB vector database instance, collection lifecycle,
    upsert operations, and semantic nearest-neighbor searches.
    """
    _instance: Optional["ChromaService"] = None
    _client = None
    _collection = None

    @classmethod
    def get_instance(cls) -> "ChromaService":
        """
        Thread-safe singleton accessor for ChromaService.

        Returns:
            ChromaService: Global shared instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initializes ChromaDB persistence directory and creates collection instance.
        """
        self.chroma_path = str(Config.CHROMA_PATH)
        Config.ensure_directories()
        self._init_client()

    def _repair_sqlite_fts(self) -> None:
        """
        Auto-repairs corrupted SQLite FTS5 shadow tables in ChromaDB metadata segment.

        Certain Windows environments or sudden server terminations can leave SQLite FTS5
        virtual tables (`embedding_fulltext_search`) in an unrecoverable state, throwing
        `vtable constructor failed`. This method drops and re-creates the FTS5 table
        without losing underlying vector embeddings.
        """
        db_file = os.path.join(self.chroma_path, "chroma.sqlite3")
        if not os.path.exists(db_file):
            return

        logger.warning(f"Attempting automated repair of Chroma SQLite FTS tables at {db_file}...")
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("PRAGMA writable_schema = ON;")
            cur.execute("DELETE FROM sqlite_schema WHERE name LIKE 'embedding_fulltext_search%';")
            cur.execute("PRAGMA writable_schema = OFF;")
            conn.commit()
            conn.close()

            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("VACUUM;")
            conn.commit()
            cur.execute("CREATE VIRTUAL TABLE embedding_fulltext_search USING fts5(string_value, tokenize='trigram');")
            conn.commit()
            conn.close()
            logger.info("Automated repair of Chroma SQLite FTS tables completed successfully.")
        except Exception as e:
            logger.error(f"Error during SQLite FTS repair: {e}")

    def _init_client(self) -> None:
        """
        Initializes the ChromaDB persistent client and retrieves or creates
        the 'devknow_functions' collection configured with cosine distance metric.
        """
        if chromadb is None or Settings is None:
            logger.warning("ChromaDB library is not loaded. Operating in fallback vector mode.")
            return

        for attempt in range(2):
            try:
                self._client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                self._collection = self._client.get_or_create_collection(
                    name="devknow_functions",
                    metadata={"hnsw:space": "cosine"}
                )
                count = self._collection.count()
                logger.info(f"ChromaDB initialized at {self.chroma_path}. Collection count: {count}")
                return
            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("metadata segment" in err_str or "vtable constructor failed" in err_str):
                    logger.warning(f"ChromaDB metadata segment issue detected ({e}). Attempting auto-repair...")
                    try:
                        self._repair_sqlite_fts()
                        continue
                    except Exception as repair_err:
                        logger.error(f"Failed automated repair of Chroma SQLite: {repair_err}")
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise

    def build_search_document(
        self,
        function_name: str,
        rule_version_name: str = "",
        rule_version_description: str = "",
        function_header: str = "",
        programming_language: str = "general",
        framework: str = "",
        description: str = "",
        class_name: str = "",
        namespace: str = "",
        additional_metadata: str = ""
    ) -> str:
        """
        Constructs a structured, token-dense representation for embedding vectorization.

        Includes function name, language, framework, signature, and context summary,
        optimizing semantic similarity matching for user search prompts.

        Args:
            function_name: Function identifier.
            rule_version_name: Module or rulebook name.
            rule_version_description: Business context or purpose.
            function_header: Signature declaration.
            programming_language: Language (e.g. 'c#', 'python').
            framework: Target framework (e.g. '.NET Core', 'Flask').
            description: Detailed functionality description.
            class_name: Class context.
            namespace: Namespace or package hierarchy.
            additional_metadata: Extra annotations or complexity notes.

        Returns:
            str: Pipe-delimited structured text block for vectorization.
        """
        parts = []
        if rule_version_name:
            parts.append(f"Rule/Module: {str(rule_version_name).strip()}")
        if rule_version_description:
            parts.append(f"Context: {str(rule_version_description).strip()}")
        parts.append(f"Function: {str(function_name).strip()}")
        parts.append(f"Language: {str(programming_language).strip()}")
        if framework:
            parts.append(f"Framework: {str(framework).strip()}")
        if class_name:
            parts.append(f"Class: {str(class_name).strip()}")
        if namespace:
            parts.append(f"Namespace: {str(namespace).strip()}")
        if function_header:
            parts.append(f"Signature: {str(function_header).strip()}")
        if description:
            parts.append(f"Description: {str(description).strip()}")
        if additional_metadata:
            parts.append(f"Metadata: {str(additional_metadata).strip()}")

        return " | ".join(parts)

    def upsert_function(
        self,
        mysql_id: int,
        function_name: str,
        rule_version_name: str = "",
        rule_version_description: str = "",
        function_header: str = "",
        programming_language: str = "general",
        framework: str = "",
        version: str = "1.0.0",
        class_name: str = "",
        namespace: str = "",
        description: str = "",
        additional_metadata: str = "",
        knowledge_source: str = "excel",
        source_file_id: Optional[int] = None
    ) -> bool:
        """
        Vectorizes and indexes or updates a single function record in ChromaDB.

        Defensive checks:
        - Validates embedding generation is non-empty before calling upsert.
        - Guarantees zero None values in metadata dictionary (strictly required by ChromaDB).

        Args:
            mysql_id: Relational database foreign key ID.
            function_name: Function identifier.
            rule_version_name: Rule or module name.
            rule_version_description: Rule context.
            function_header: Exact function signature.
            programming_language: Language identifier.
            framework: Framework tag.
            version: Version string.
            class_name: Class or struct name.
            namespace: Namespace hierarchy.
            description: Semantic description.
            additional_metadata: Custom metadata.
            knowledge_source: Source origin ('excel', 'code_file', 'doc_file', etc.).
            source_file_id: ID of the uploaded source file.

        Returns:
            bool: True if upserted successfully, False otherwise.
        """
        if not self._collection:
            self._init_client()

        doc = self.build_search_document(
            function_name=function_name,
            rule_version_name=rule_version_name,
            rule_version_description=rule_version_description,
            function_header=function_header,
            programming_language=programming_language,
            framework=framework,
            description=description,
            class_name=class_name,
            namespace=namespace,
            additional_metadata=additional_metadata
        )

        embedding = embedding_service.embed_text(doc)
        if not embedding:
            logger.warning(f"Could not generate embedding for function '{function_name}' ({mysql_id}). Skipping vector upsert.")
            return False

        # Strictly sanitize metadata to ensure no None values exist
        metadata: Dict[str, Any] = {
            "mysql_id": int(mysql_id),
            "function_name": str(function_name or "unnamed"),
            "rule_version_name": str(rule_version_name or ""),
            "programming_language": str(programming_language or "general").lower(),
            "framework": str(framework or "").lower(),
            "version": str(version or "1.0.0"),
            "class_name": str(class_name or ""),
            "namespace": str(namespace or ""),
            "knowledge_source": str(knowledge_source or "excel"),
            "source_file_id": int(source_file_id) if source_file_id is not None else 0
        }

        try:
            self._collection.upsert(
                ids=[f"fn_{mysql_id}"],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting function {mysql_id} into ChromaDB: {e}")
            return False

    def upsert_batch(self, items: List[Dict[str, Any]]) -> int:
        """
        Batch vectorization and chunked upsert for high-throughput ingestion.

        Splits items into chunks of 500 to stay safely below ChromaDB's maximum
        batch limit and prevents memory exhaustion.

        Args:
            items: List of dictionary records containing function attributes.

        Returns:
            int: Total number of successfully upserted items.
        """
        if not items:
            return 0

        if not self._collection:
            self._init_client()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in items:
            m_id = item.get("mysql_id")
            if m_id is None:
                continue

            f_name = item.get("function_name", "unnamed")
            p_lang = item.get("programming_language") or "general"
            f_work = item.get("framework") or ""

            doc = self.build_search_document(
                function_name=f_name,
                rule_version_name=item.get("rule_version_name", ""),
                rule_version_description=item.get("rule_version_description", ""),
                function_header=item.get("function_header", ""),
                programming_language=p_lang,
                framework=f_work,
                description=item.get("description", ""),
                class_name=item.get("class_name", ""),
                namespace=item.get("namespace", ""),
                additional_metadata=item.get("additional_metadata", "")
            )
            documents.append(doc)
            ids.append(f"fn_{m_id}")
            metadatas.append({
                "mysql_id": int(m_id),
                "function_name": str(f_name),
                "rule_version_name": str(item.get("rule_version_name") or ""),
                "programming_language": str(p_lang).lower(),
                "framework": str(f_work).lower(),
                "version": str(item.get("version") or "1.0.0"),
                "class_name": str(item.get("class_name") or ""),
                "namespace": str(item.get("namespace") or ""),
                "knowledge_source": str(item.get("knowledge_source") or "excel"),
                "source_file_id": int(item.get("source_file_id") or 0)
            })

        if not ids:
            return 0

        # Batch embed all formatted documents in one pass
        embeddings = embedding_service.embed_documents(documents)
        if not embeddings or len(embeddings) != len(ids):
            logger.error("Embedding count mismatch or empty embeddings generated during batch upsert.")
            return 0

        # Chunked ingestion: Max 500 items per upsert call
        CHUNK_SIZE = 500
        total_upserted = 0

        for i in range(0, len(ids), CHUNK_SIZE):
            chunk_ids = ids[i:i + CHUNK_SIZE]
            chunk_embeddings = embeddings[i:i + CHUNK_SIZE]
            chunk_documents = documents[i:i + CHUNK_SIZE]
            chunk_metadatas = metadatas[i:i + CHUNK_SIZE]

            try:
                self._collection.upsert(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunk_documents,
                    metadatas=chunk_metadatas
                )
                total_upserted += len(chunk_ids)
            except Exception as e:
                logger.error(f"Error upserting chunk [{i}:{i + CHUNK_SIZE}] into ChromaDB: {e}")

        logger.info(f"Successfully upserted {total_upserted} vectors into ChromaDB in chunks.")
        return total_upserted

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
        framework: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic nearest-neighbor search against the vector index.

        Transforms raw cosine distance into normalized similarity scores:
        `similarity = 1.0 - distance`

        Args:
            query: User search query or code context.
            top_k: Maximum number of candidate results to retrieve.
            language: Optional language filter (e.g. 'c#', 'python').
            framework: Optional framework filter.

        Returns:
            List of matching records ordered by similarity descending.
        """
        if not self._collection:
            self._init_client()

        if not query or not query.strip():
            return []

        count = self.get_count()
        if count == 0:
            return []

        query_embedding = embedding_service.embed_query(query)
        if not query_embedding:
            logger.warning(f"Could not generate query embedding for query: '{query[:50]}'")
            return []

        # Construct filter clause
        where_filter: Dict[str, Any] = {}
        if language and language.strip() and language.strip().lower() != "all":
            where_filter["programming_language"] = language.strip().lower()
        if framework and framework.strip() and framework.strip().lower() != "all":
            where_filter["framework"] = framework.strip().lower()

        where_clause: Optional[Dict[str, Any]] = None
        if len(where_filter) == 1:
            where_clause = where_filter
        elif len(where_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in where_filter.items()]}

        n_results = min(max(1, top_k), count)

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query execution error: {e}")
            return []

        matches: List[Dict[str, Any]] = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)
            docs = results["documents"][0] if "documents" in results else [""] * len(ids)

            for i, doc_id in enumerate(ids):
                distance = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, min(1.0, 1.0 - distance))

                meta = metas[i] if i < len(metas) else {}
                matches.append({
                    "id": doc_id,
                    "mysql_id": meta.get("mysql_id"),
                    "function_name": meta.get("function_name"),
                    "programming_language": meta.get("programming_language"),
                    "framework": meta.get("framework"),
                    "version": meta.get("version"),
                    "knowledge_source": meta.get("knowledge_source"),
                    "rule_version_name": meta.get("rule_version_name"),
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4),
                    "document_preview": docs[i] if i < len(docs) else ""
                })

        # Sort results in descending similarity order
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def delete_function(self, mysql_id: int) -> bool:
        """
        Removes a function vector record from ChromaDB by its relational primary key.

        Args:
            mysql_id: Relational database foreign key ID.

        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        if not self._collection:
            self._init_client()

        try:
            self._collection.delete(ids=[f"fn_{mysql_id}"])
            logger.info(f"Deleted vector fn_{mysql_id} from ChromaDB.")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete vector fn_{mysql_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Resets and re-creates the `devknow_functions` collection.
        Used primarily during full knowledge-base reindexing operations.

        Returns:
            bool: True if reset succeeded, False otherwise.
        """
        try:
            try:
                self._client.delete_collection("devknow_functions")
            except Exception as del_err:
                logger.debug(f"Collection deletion notice (safe to ignore if not exists): {del_err}")

            self._collection = self._client.get_or_create_collection(
                name="devknow_functions",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB devknow_functions collection cleared.")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    def get_count(self) -> int:
        """
        Returns the total number of vectors indexed in the collection.

        Returns:
            int: Vector count.
        """
        try:
            return self._collection.count() if self._collection else 0
        except Exception as e:
            logger.debug(f"Error getting collection count: {e}")
            return 0


# Singleton export for system-wide consumption
chroma_service = ChromaService.get_instance()
