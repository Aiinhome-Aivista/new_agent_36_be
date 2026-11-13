import re
import datetime
import logging
from typing import Dict, Any, List

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.models.database_models import FunctionRecord, LLMGeneratedKnowledge

logger = logging.getLogger("DevKnow.KBGrowth")

class KBGrowthService:
    @staticmethod
    def extract_code_blocks(markdown_text: str) -> List[Dict[str, str]]:
        """Extracts code blocks from markdown answer"""
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
        Stores an accepted LLM-generated solution into the Global Knowledge Base.
        Inserts into MySQL (functions & llm_generated_knowledge) and updates ChromaDB.
        """
        # If code body wasn't explicitly supplied, try extracting the first code block
        if not function_body:
            blocks = cls.extract_code_blocks(llm_answer)
            if blocks:
                function_body = blocks[0]["code"]
                if not programming_language or programming_language.lower() in ["text", "code"]:
                    programming_language = blocks[0]["language"]

        if not function_body:
            # Fall back to storing full answer
            function_body = llm_answer

        if not function_header:
            first_line = function_body.split("\n")[0]
            function_header = first_line[:150]

        if not function_name:
            # Generate sensible name from query
            words = re.findall(r"\w+", original_query.lower())
            function_name = "_".join(words[:4]) or "ai_generated_function"

        if not description:
            description = f"Knowledge learned from user query: '{original_query}'"

        programming_language = programming_language.lower().strip() or "general"

        with db_service.get_session() as session:
            # Create FunctionRecord
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
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow()
            )
            session.add(func)
            session.flush()

            # Record origin in llm_generated_knowledge
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

        # Upsert into ChromaDB so it's immediately searchable
        chroma_service.upsert_function(
            mysql_id=record_id,
            function_name=function_name,
            rule_version_name=rule_version_name,
            rule_version_description=description,
            function_header=function_header,
            programming_language=programming_language,
            framework=framework,
            version=version,
            description=description,
            knowledge_source="llm_generated"
        )

        logger.info(f"Successfully saved LLM solution into KB. Function ID: {record_id}, Name: {function_name}")

        return {
            "success": True,
            "function_id": record_id,
            "function_name": function_name,
            "programming_language": programming_language,
            "knowledge_source": "llm_generated",
            "message": "Solution successfully added to organizational knowledge base."
        }

    @classmethod
    def list_llm_generated(cls) -> List[Dict[str, Any]]:
        """Lists all functions learned from LLM answers"""
        with db_service.get_session() as session:
            records = (
                session.query(FunctionRecord)
                .filter_by(knowledge_source="llm_generated")
                .order_by(FunctionRecord.created_at.desc())
                .all()
            )
            return [r.to_dict() for r in records]

    @classmethod
    def delete_llm_generated(cls, function_id: int) -> bool:
        """Deletes an LLM-generated entry from both MySQL and ChromaDB"""
        with db_service.get_session() as session:
            func = session.query(FunctionRecord).filter_by(id=function_id, knowledge_source="llm_generated").first()
            if not func:
                return False
            # Delete from MySQL
            session.query(LLMGeneratedKnowledge).filter_by(function_id=function_id).delete()
            session.delete(func)

        # Delete from ChromaDB
        chroma_service.delete_function(function_id)
        logger.info(f"Purged LLM-generated function {function_id} from KB.")
        return True

