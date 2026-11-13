import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class KnowledgeBaseFile(Base):
    __tablename__ = "knowledge_base_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    total_records = Column(Integer, default=0)
    successful_records = Column(Integer, default=0)
    failed_records = Column(Integer, default=0)
    skipped_records = Column(Integer, default=0)
    languages_detected = Column(String(500), default="")
    ingestion_status = Column(String(50), default="pending")  # completed, failed, partial

    functions = relationship("FunctionRecord", back_populates="source_file", cascade="all, delete-orphan")
    logs = relationship("IngestionLog", back_populates="file", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "total_records": self.total_records,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "skipped_records": self.skipped_records,
            "languages_detected": self.languages_detected.split(",") if self.languages_detected else [],
            "ingestion_status": self.ingestion_status
        }


class FunctionRecord(Base):
    __tablename__ = "functions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_version_name = Column(String(255), index=True, default="")
    rule_version_description = Column(Text, default="")
    function_name = Column(String(255), nullable=False, index=True)
    function_header = Column(Text, nullable=False)
    function_body = Column(Text, nullable=False)
    programming_language = Column(String(100), nullable=False, index=True)
    framework = Column(String(100), index=True, default="")
    version = Column(String(50), index=True, default="1.0.0")
    class_name = Column(String(255), index=True, default="")
    namespace = Column(String(255), index=True, default="")
    description = Column(Text, default="")
    additional_metadata = Column(Text, default="")
    knowledge_source = Column(String(50), default="excel", index=True)  # 'excel' or 'llm_generated'
    source_file_id = Column(Integer, ForeignKey("knowledge_base_files.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    source_file = relationship("KnowledgeBaseFile", back_populates="functions")
    llm_origin = relationship("LLMGeneratedKnowledge", back_populates="function_record", uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "rule_version_name": self.rule_version_name,
            "rule_version_description": self.rule_version_description,
            "function_name": self.function_name,
            "function_header": self.function_header,
            "function_body": self.function_body,
            "programming_language": self.programming_language,
            "framework": self.framework,
            "version": self.version,
            "class_name": self.class_name,
            "namespace": self.namespace,
            "description": self.description,
            "additional_metadata": self.additional_metadata,
            "knowledge_source": self.knowledge_source,
            "source_file_id": self.source_file_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class LLMGeneratedKnowledge(Base):
    __tablename__ = "llm_generated_knowledge"

    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(Integer, ForeignKey("functions.id"), nullable=False)
    original_query = Column(Text, nullable=False)
    llm_answer = Column(Text, nullable=False)
    user_decision = Column(String(50), default="accepted")  # 'accepted', 'edited', 'manual_save'
    similarity_score_at_query = Column(Float, default=0.0)
    stored_at = Column(DateTime, default=datetime.datetime.utcnow)

    function_record = relationship("FunctionRecord", back_populates="llm_origin")

    def to_dict(self):
        return {
            "id": self.id,
            "function_id": self.function_id,
            "original_query": self.original_query,
            "llm_answer": self.llm_answer,
            "user_decision": self.user_decision,
            "similarity_score_at_query": self.similarity_score_at_query,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None
        }


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("knowledge_base_files.id"), nullable=False)
    record_identifier = Column(String(255), default="")
    status = Column(String(50), nullable=False)  # 'success', 'failed', 'duplicate'
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    file = relationship("KnowledgeBaseFile", back_populates="logs")

    def to_dict(self):
        return {
            "id": self.id,
            "file_id": self.file_id,
            "record_identifier": self.record_identifier,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="ConversationMessage.created_at.asc()")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0
        }


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user', 'assistant'
    message = Column(Text, nullable=False)
    attached_code = Column(Text, nullable=True)
    answer_source = Column(String(50), default="none")  # 'kb_hit', 'llm_fallback', 'none'
    retrieved_function_ids = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)
    query_intent = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "message": self.message,
            "attached_code": self.attached_code,
            "answer_source": self.answer_source,
            "retrieved_function_ids": self.retrieved_function_ids,
            "similarity_score": self.similarity_score,
            "query_intent": self.query_intent,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

