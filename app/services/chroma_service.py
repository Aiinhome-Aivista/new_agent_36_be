import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from app.config import Config
from app.services.embedding_service import embedding_service

logger = logging.getLogger("DevKnow.Chroma")

class ChromaService:
    _instance = None
    _client = None
    _collection = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.chroma_path = str(Config.CHROMA_PATH)
        Config.ensure_directories()
        self._init_client()

    def _init_client(self):
        try:
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            # Collection name for functions
            self._collection = self._client.get_or_create_collection(
                name="devknow_functions",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {self.chroma_path}. Collection count: {self._collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def build_search_document(
        self,
        function_name: str,
        rule_version_name: str,
        rule_version_description: str,
        function_header: str,
        programming_language: str,
        framework: str = "",
        description: str = "",
        class_name: str = "",
        namespace: str = "",
        additional_metadata: str = ""
    ) -> str:
        """Constructs a clean semantic representation for vector embedding"""
        parts = []
        if rule_version_name:
            parts.append(f"Rule/Module: {rule_version_name}")
        if rule_version_description:
            parts.append(f"Context: {rule_version_description}")
        parts.append(f"Function: {function_name}")
        parts.append(f"Language: {programming_language}")
        if framework:
            parts.append(f"Framework: {framework}")
        if class_name:
            parts.append(f"Class: {class_name}")
        if namespace:
            parts.append(f"Namespace: {namespace}")
        if function_header:
            parts.append(f"Signature: {function_header}")
        if description:
            parts.append(f"Description: {description}")
        if additional_metadata:
            parts.append(f"Metadata: {additional_metadata}")

        return " | ".join(parts)

    def upsert_function(
        self,
        mysql_id: int,
        function_name: str,
        rule_version_name: str,
        rule_version_description: str,
        function_header: str,
        programming_language: str,
        framework: str = "",
        version: str = "1.0.0",
        class_name: str = "",
        namespace: str = "",
        description: str = "",
        additional_metadata: str = "",
        knowledge_source: str = "excel",
        source_file_id: Optional[int] = None
    ):
        """Indexes or updates a single function vector in ChromaDB"""
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

        metadata = {
            "mysql_id": int(mysql_id),
            "function_name": str(function_name),
            "rule_version_name": str(rule_version_name or ""),
            "programming_language": str(programming_language).lower(),
            "framework": str(framework or "").lower(),
            "version": str(version or "1.0.0"),
            "class_name": str(class_name or ""),
            "namespace": str(namespace or ""),
            "knowledge_source": str(knowledge_source),
            "source_file_id": int(source_file_id) if source_file_id is not None else 0
        }

        self._collection.upsert(
            ids=[f"fn_{mysql_id}"],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[metadata]
        )

    def upsert_batch(self, items: List[Dict[str, Any]]):
        """Batch upsert for fast Excel ingestion"""
        if not items:
            return

        ids = []
        documents = []
        metadatas = []

        for item in items:
            doc = self.build_search_document(
                function_name=item["function_name"],
                rule_version_name=item.get("rule_version_name", ""),
                rule_version_description=item.get("rule_version_description", ""),
                function_header=item["function_header"],
                programming_language=item["programming_language"],
                framework=item.get("framework", ""),
                description=item.get("description", ""),
                class_name=item.get("class_name", ""),
                namespace=item.get("namespace", ""),
                additional_metadata=item.get("additional_metadata", "")
            )
            documents.append(doc)
            ids.append(f"fn_{item['mysql_id']}")
            metadatas.append({
                "mysql_id": int(item["mysql_id"]),
                "function_name": str(item["function_name"]),
                "rule_version_name": str(item.get("rule_version_name", "")),
                "programming_language": str(item["programming_language"]).lower(),
                "framework": str(item.get("framework", "")).lower(),
                "version": str(item.get("version", "1.0.0")),
                "class_name": str(item.get("class_name", "")),
                "namespace": str(item.get("namespace", "")),
                "knowledge_source": str(item.get("knowledge_source", "excel")),
                "source_file_id": int(item["source_file_id"]) if item.get("source_file_id") is not None else 0
            })

        # Batch embed documents
        embeddings = embedding_service.embed_documents(documents)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully upserted {len(ids)} vectors into ChromaDB.")

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
        framework: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against ChromaDB.
        Returns list of matched results with mysql_id, distance, similarity score, and metadata.
        """
        query_embedding = embedding_service.embed_query(query)

        where_filter = {}
        if language and language.strip() and language.strip().lower() != "all":
            where_filter["programming_language"] = language.strip().lower()
        if framework and framework.strip() and framework.strip().lower() != "all":
            where_filter["framework"] = framework.strip().lower()

        where_clause = None
        if len(where_filter) == 1:
            where_clause = where_filter
        elif len(where_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in where_filter.items()]}

        count = self._collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["metadatas", "distances", "documents"]
        )

        matches = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)
            docs = results["documents"][0] if "documents" in results else [""] * len(ids)

            for i, doc_id in enumerate(ids):
                # Cosine distance to similarity: similarity = 1 - distance
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

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def delete_function(self, mysql_id: int):
        """Removes a vector record from ChromaDB"""
        try:
            self._collection.delete(ids=[f"fn_{mysql_id}"])
            logger.info(f"Deleted vector fn_{mysql_id} from ChromaDB.")
        except Exception as e:
            logger.warning(f"Failed to delete vector fn_{mysql_id}: {e}")

    def clear_collection(self):
        """Clears all records in ChromaDB collection"""
        try:
            self._client.delete_collection("devknow_functions")
            self._collection = self._client.get_or_create_collection(
                name="devknow_functions",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB devknow_functions collection cleared.")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")

    def get_count(self) -> int:
        return self._collection.count() if self._collection else 0

chroma_service = ChromaService.get_instance()

