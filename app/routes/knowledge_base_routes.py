"""
Knowledge Base Management Routes Blueprint for DevKnow.

Provides management, observability, and administrative review endpoints:
1. `GET /api/knowledge-base/stats`: Aggregated dashboard metrics (counts, language breakdown).
2. `GET /api/knowledge-base/files`: List of all uploaded knowledge base files.
3. `POST /api/knowledge-base/reindex`: Re-vectorizes all relational functions into ChromaDB.
4. `POST /api/knowledge/save-llm-answer`: Self-growing KB: stages accepted chat solutions.
5. `GET /api/knowledge/llm-generated`: Lists auto-learned functions in the KB.
6. `DELETE /api/knowledge/llm-generated/<id>`: Removes an auto-learned function.
7. `GET /api/knowledge-base/staging-queue`: Fetches functions awaiting human verification.
8. `POST /api/knowledge-base/staging-queue/<id>/approve`: Human approval -> verified status & vector indexing.
9. `POST /api/knowledge-base/staging-queue/<id>/reject`: Human rejection -> marked rejected, excluded from index.
"""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import func

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.kb_growth_service import KBGrowthService
from app.services.code_doc_ingestion_service import CodeDocIngestionService
from app.models.database_models import KnowledgeBaseFile, FunctionRecord, LLMGeneratedKnowledge

kb_bp = Blueprint("knowledge_base", __name__)
logger = logging.getLogger("DevKnow.KBRoutes")


@kb_bp.route("/knowledge-base/stats", methods=["GET"])
def get_stats():
    """
    Returns high-level statistics and dashboard metrics for the platform.

    Metrics Returned:
    - Total functions stored in relational database.
    - Breakdown by knowledge source ('excel' vs 'llm_generated').
    - Total uploaded files.
    - Distribution of functions across programming languages.
    - ChromaDB vector collection count.
    - 5 most recently uploaded knowledge files.

    Responses:
        200: JSON object containing dashboard counts and language distribution.
    """
    with db_service.get_session() as session:
        total_functions = session.query(func.count(FunctionRecord.id)).scalar() or 0
        excel_functions = session.query(func.count(FunctionRecord.id)).filter_by(knowledge_source="excel").scalar() or 0
        llm_functions = session.query(func.count(FunctionRecord.id)).filter_by(knowledge_source="llm_generated").scalar() or 0
        total_files = session.query(func.count(KnowledgeBaseFile.id)).scalar() or 0

        # Language distribution
        lang_counts = (
            session.query(FunctionRecord.programming_language, func.count(FunctionRecord.id))
            .group_by(FunctionRecord.programming_language)
            .all()
        )
        languages_distribution = [{"language": lang, "count": count} for lang, count in lang_counts if lang]

        # Recent files
        recent_files = (
            session.query(KnowledgeBaseFile)
            .order_by(KnowledgeBaseFile.uploaded_at.desc())
            .limit(5)
            .all()
        )

        vector_count = chroma_service.get_count()

        if vector_count != total_functions:
            logger.warning(
                f"[Database Discrepancy] MySQL function count ({total_functions}) != ChromaDB vector count ({vector_count}). "
                f"Sync recommended via POST /api/knowledge-base/reindex."
            )

        return jsonify({
            "success": True,
            "total_functions": total_functions,
            "excel_functions": excel_functions,
            "llm_functions": llm_functions,
            "total_files": total_files,
            "vector_index_count": vector_count,
            "languages_distribution": languages_distribution,
            "recent_files": [f.to_dict() for f in recent_files]
        })


@kb_bp.route("/knowledge-base/files", methods=["GET"])
def list_files():
    """
    Returns a list of all uploaded knowledge base files and spreadsheets.

    Responses:
        200: JSON list of KnowledgeBaseFile records ordered by upload date descending.
    """
    with db_service.get_session() as session:
        files = session.query(KnowledgeBaseFile).order_by(KnowledgeBaseFile.uploaded_at.desc()).all()
        return jsonify({
            "success": True,
            "files": [f.to_dict() for f in files]
        })


@kb_bp.route("/knowledge-base/reindex", methods=["POST"])
def reindex_knowledge_base():
    """
    Re-synchronizes the ChromaDB vector collection from scratch.

    Workflow:
    1. Resets the `devknow_functions` ChromaDB collection.
    2. Fetches all function records from the database.
    3. Batch vectorizes and commits all records into the collection.

    Responses:
        200: Successfully reindexed with count of synced vectors.
        500: Reindexing failure details.
    """
    try:
        logger.info("[KnowledgeBase] Starting full search index re-synchronization...")
        chroma_service.clear_collection()
        with db_service.get_session() as session:
            records = session.query(FunctionRecord).all()
            batch = [
                {
                    "mysql_id": r.id,
                    "function_name": r.function_name,
                    "rule_version_name": r.rule_version_name,
                    "rule_version_description": r.rule_version_description,
                    "function_header": r.function_header,
                    "programming_language": r.programming_language,
                    "framework": r.framework,
                    "version": r.version,
                    "class_name": r.class_name,
                    "namespace": r.namespace,
                    "description": r.description,
                    "additional_metadata": r.additional_metadata,
                    "knowledge_source": r.knowledge_source,
                    "source_file_id": r.source_file_id
                }
                for r in records
            ]
            if batch:
                chroma_service.upsert_batch(batch)

        logger.info(f"[KnowledgeBase] Re-indexing complete. Synchronized {len(batch)} functions into ChromaDB.")
        return jsonify({
            "success": True,
            "message": f"Successfully re-indexed {len(batch)} functions into ChromaDB.",
            "total_indexed": len(batch)
        })
    except Exception as e:
        logger.exception(f"[KnowledgeBase] Re-indexing failed with error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kb_bp.route("/knowledge/save-llm-answer", methods=["POST"])
def save_llm_answer():
    """
    Self-growing KB endpoint: Stages an accepted AI solution for human verification.

    Payload:
        - query: Original user prompt.
        - llm_answer: AI synthesized response.
        - function_name: (Optional) Extracted function name.
        - programming_language: (Optional) Programming language.
        - function_body: (Optional) Code block.

    Responses:
        200: Successfully staged into Review Queue.
        400: Missing query or llm_answer.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    llm_answer = data.get("llm_answer", "").strip()

    if not query or not llm_answer:
        return jsonify({"success": False, "error": "Query and LLM answer are required"}), 400

    result = KBGrowthService.save_llm_answer_to_kb(
        original_query=query,
        llm_answer=llm_answer,
        function_name=data.get("function_name", ""),
        programming_language=data.get("programming_language", "general"),
        function_header=data.get("function_header", ""),
        function_body=data.get("function_body", ""),
        framework=data.get("framework", ""),
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        similarity_score_at_query=float(data.get("similarity_score", 0.0)),
        user_decision="accepted"
    )

    return jsonify(result), 200


@kb_bp.route("/knowledge/llm-generated", methods=["GET"])
def list_llm_generated():
    """
    Returns all functions originally created via LLM chat interactions.

    Responses:
        200: JSON list of LLM-generated function records.
    """
    records = KBGrowthService.list_llm_generated()
    return jsonify({"success": True, "count": len(records), "records": records})


@kb_bp.route("/knowledge/llm-generated/<int:func_id>", methods=["DELETE"])
def delete_llm_generated(func_id):
    """
    Removes an LLM-generated record from the database and vector store.

    Args:
        func_id: Primary key id of the function.

    Responses:
        200: Deleted successfully.
        404: Function not found or not LLM-generated.
    """
    success = KBGrowthService.delete_llm_generated(func_id)
    if success:
        return jsonify({"success": True, "message": f"Record {func_id} removed from KB."})
    return jsonify({"success": False, "error": "Record not found or not LLM-generated"}), 404


# ── Human-in-the-Loop Review Queue Endpoints ─────────────────────────────────

@kb_bp.route("/knowledge-base/staging-queue", methods=["GET"])
def get_staging_queue():
    """
    Returns all functions waiting in the staging queue for human review.

    Query Parameters:
        status: (Optional) Filter status, defaults to 'pending_review'.

    Responses:
        200: List of staged function dictionaries with conflict comparisons.
    """
    status = request.args.get("status", "pending_review")
    items = CodeDocIngestionService.list_staging_queue(status=status)
    return jsonify({
        "success": True,
        "count": len(items),
        "items": items
    }), 200


@kb_bp.route("/knowledge-base/staging-queue/<int:func_id>/approve", methods=["POST"])
def approve_staged_function(func_id):
    """
    Human verification action: Approves a function in the staging queue.
    Updates `verification_status` to 'verified' and indexes into ChromaDB.

    Path Parameters:
        func_id: ID of the function record to approve.

    JSON Payload:
        version: (Optional) Version override string.
        notes: (Optional) Reviewer notes.

    Responses:
        200: Function approved and indexed.
        400: Function not found or approval failed.
    """
    data = request.get_json() or {}
    approved_version = data.get("version")
    notes = data.get("notes")

    res = CodeDocIngestionService.approve_staged_function(
        function_id=func_id,
        approved_version=approved_version,
        notes=notes
    )
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code


@kb_bp.route("/knowledge-base/staging-queue/<int:func_id>/reject", methods=["POST"])
def reject_staged_function(func_id):
    """
    Human verification action: Rejects a function in the staging queue.
    Marks `verification_status` as 'rejected' and excludes it from ChromaDB.

    Path Parameters:
        func_id: ID of the function record to reject.

    JSON Payload:
        reason: (Optional) Rationale for rejection.

    Responses:
        200: Function rejected successfully.
        400: Function not found.
    """
    data = request.get_json() or {}
    reason = data.get("reason", "Rejected during human review")

    res = CodeDocIngestionService.reject_staged_function(
        function_id=func_id,
        reason=reason
    )
    status_code = 200 if res.get("success") else 400
    return jsonify(res), status_code
