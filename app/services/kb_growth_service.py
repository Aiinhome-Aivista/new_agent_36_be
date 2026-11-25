"""
Knowledge Base Growth & Self-Learning Module for DevKnow.

Enables the DevKnow platform to continuously capture, stage, and evolve its organizational
knowledge from high-quality user-accepted chat solutions.

Design & Policy:
1. Markdown Code Block Extraction: Parses code blocks out of LLM answers.
2. Zero-Direct-Ingestion Gate: Saves to the database with `verification_status='pending_review'`.
   Unverified solutions are never automatically pushed to ChromaDB vector search.
3. Conflict & Version Tracking: Checks if a function with the same name exists. If so,
   detects collision and proposes an auto-incremented version (e.g. 1.0.0 -> 1.1.0).
4. Provenance & Audit Trail: Persists prompt and metadata link in `llm_generated_knowledge`.
5. Human Verification: Once a senior developer approves the item via the UI, it is promoted
   to `verified` and indexed into the ChromaDB vector store.
"""

import re
import datetime
import logging
from typing import Dict, Any, List, Optional

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.models.database_models import FunctionRecord, LLMGeneratedKnowledge

logger = logging.getLogger("DevKnow.KBGrowth")


class KBGrowthService:
    """
    Service responsible for capturing approved solutions from chat interactions,
    staging them into the verification queue, and maintaining version lineages.
    """

    @staticmethod
    def extract_code_blocks(markdown_text: str) -> List[Dict[str, str]]:
        """
        Extracts fenced code blocks from markdown text.

        Args:
            markdown_text: Markdown formatted string containing ```lang ... ``` blocks.

        Returns:
            List[Dict[str, str]]: List of dicts with 'language' and 'code' keys.
        """
        pattern = r"```([a-zA-Z0-9_#+]*)\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        blocks = []
        for lang, code in matches:
            blocks.append({
                "language": lang.strip() or "text",
                "code": code.strip()
            })
        return blocks

    @classmethod
    def save_llm_answer_to_kb(
        cls,
        original_query: str,
        llm_answer: str,
        function_name: str,
        programming_language: str,
        function_header: str = "",
        function_body: str = "",
        framework: str = "",
        version: str = "1.0.0",
        rule_version_name: str = "AI_Learned_Knowledge",
        description: str = "",
        similarity_score_at_query: float = 0.0,
        user_decision: str = "accepted"
    ) -> Dict[str, Any]:
        """
        Stages an accepted LLM solution into the Knowledge Base Review Queue.

        Persists into both `functions` and `llm_generated_knowledge` tables.
        In accordance with the zero-direct-ingestion policy, the record is created with
        `verification_status='pending_review'` and is NOT added to ChromaDB until approved.

        Args:
            original_query: The question that prompted the answer.
            llm_answer: Full text of the generated response.
            function_name: Proposed function identifier.
            programming_language: Language (e.g. 'python', 'c#').
            function_header: Signature line of the function.
            function_body: Implementation body of the function.
            framework: Associated framework or runtime.
            version: Target version string.
            rule_version_name: Module or rule classification name.
            description: Purpose explanation.
            similarity_score_at_query: The vector similarity score observed during the original query.
            user_decision: Feedback status ('accepted', 'saved').

        Returns:
            Dict containing staging record ID, version, conflict status, and confirmation.
        """
        # If code body wasn't explicitly supplied, extract the first markdown code block
        if not function_body:
            blocks = cls.extract_code_blocks(llm_answer)
            if blocks:
                function_body = blocks[0]["code"]
                if not programming_language or programming_language.lower() in ["text", "code"]:
                    programming_language = blocks[0]["language"]

        if not function_body:
            function_body = llm_answer

        if not function_header:
            first_line = function_body.split("\n")[0]
            function_header = first_line[:150]

        if not function_name:
            words = re.findall(r"\w+", original_query.lower())
            function_name = "_".join(words[:4]) or "ai_generated_function"

        if not description:
            description = f"Knowledge learned from user query: '{original_query}'"

        programming_language = programming_language.lower().strip() or "general"

        with db_service.get_session() as session:
            # Conflict & Versioning Check
            existing = (
                session.query(FunctionRecord)
                .filter_by(function_name=function_name)
                .order_by(FunctionRecord.created_at.desc())
                .first()
            )
            if existing:
                conflict_status = "conflict_detected"
                parent_id = existing.id
                parts = (existing.version or "1.0.0").replace("v", "").split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    parts[1] = str(int(parts[1]) + 1)
                    version = ".".join(parts)
                else:
                    version = "1.1.0"
            else:
                conflict_status = "clean"
                parent_id = None
                version = version or "1.0.0"

            # Create FunctionRecord in STAGING (verification_status='pending_review')
            func = FunctionRecord(
                rule_version_name=rule_version_name,
                rule_version_description=f"Auto-learned function from user question: {original_query}",
                function_name=function_name,
                function_header=function_header,
                function_body=function_body,
                programming_language=programming_language,
                framework=framework,
                version=version,
                description=description,
                knowledge_source="llm_generated",
                verification_status="pending_review",
                conflict_status=conflict_status,
                parent_function_id=parent_id,
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow()
            )
            session.add(func)
            session.flush()

            # Record origin in llm_generated_knowledge audit table
            llm_record = LLMGeneratedKnowledge(
                function_id=func.id,
                original_query=original_query,
                llm_answer=llm_answer,
                user_decision=user_decision,
                similarity_score_at_query=similarity_score_at_query,
                stored_at=datetime.datetime.utcnow()
            )
            session.add(llm_record)
            session.flush()

            record_id = func.id

        logger.info(
            f"Staged LLM solution for human verification. Function ID: {record_id}, Name: {function_name}, "
            f"Version: {version}, Conflict: {conflict_status}."
        )

        return {
            "success": True,
            "function_id": record_id,
            "function_name": function_name,
            "version": version,
            "conflict_status": conflict_status,
            "verification_status": "pending_review",
            "programming_language": programming_language,
            "knowledge_source": "llm_generated",
            "message": f"Solution submitted to Staging Queue (Version {version}). A senior developer or admin will verify it before it becomes active in the Knowledge Base."
        }

    @classmethod
    def list_llm_generated(cls) -> List[Dict[str, Any]]:
        """
        Returns all functions originally created via LLM chat interactions.

        Returns:
            List[Dict[str, Any]]: List of serialized function records.
        """
        with db_service.get_session() as session:
            records = (
                session.query(FunctionRecord)
                .filter(FunctionRecord.knowledge_source == "llm_generated")
                .order_by(FunctionRecord.created_at.desc())
                .all()
            )
            return [r.to_dict() for r in records]

    @classmethod
    def delete_llm_generated(cls, func_id: int) -> bool:
        """
        Removes an LLM-generated function from the database and vector store.

        Args:
            func_id: Primary key id of the FunctionRecord to remove.

        Returns:
            bool: True if found and deleted, False otherwise.
        """
        with db_service.get_session() as session:
            record = (
                session.query(FunctionRecord)
                .filter_by(id=func_id, knowledge_source="llm_generated")
                .first()
            )
            if not record:
                return False

            # Delete associated LLM origin metadata if exists
            session.query(LLMGeneratedKnowledge).filter_by(function_id=func_id).delete()
            session.delete(record)
            session.commit()

        # Remove from vector store
        try:
            chroma_service.delete_function(func_id)
        except Exception as e:
            logger.debug(f"Notice deleting vector for LLM function {func_id}: {e}")

        logger.info(f"Successfully deleted LLM-generated function {func_id} from KB.")
        return True

