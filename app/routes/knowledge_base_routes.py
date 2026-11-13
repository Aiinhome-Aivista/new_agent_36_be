import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import func

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.kb_growth_service import KBGrowthService
from app.models.database_models import KnowledgeBaseFile, FunctionRecord, LLMGeneratedKnowledge

kb_bp = Blueprint("knowledge_base", __name__)
logger = logging.getLogger("DevKnow.KBRoutes")

@kb_bp.route("/knowledge-base/stats", methods=["GET"])
def get_stats():
    """Returns overall knowledge base and dashboard statistics"""
    with db_service.get_session() as session:
        total_functions = session.query(func.count(FunctionRecord.id)).scalar() or 0
        excel_functions = session.query(func.count(FunctionRecord.id)).filter_by(knowledge_source="excel").scalar() or 0
        llm_functions = session.query(func.count(FunctionRecord.id)).filter_by(knowledge_source="llm_generated").scalar() or 0
        total_files = session.query(func.count(KnowledgeBaseFile.id)).scalar() or 0

        # Unique languages with count
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
    """Returns list of all uploaded knowledge base files"""
    with db_service.get_session() as session:
        files = session.query(KnowledgeBaseFile).order_by(KnowledgeBaseFile.uploaded_at.desc()).all()
        return jsonify({
            "success": True,
            "files": [f.to_dict() for f in files]
        })

@kb_bp.route("/knowledge-base/reindex", methods=["POST"])
def reindex_knowledge_base():
    """Re-indexes all MySQL function records into ChromaDB vector store"""
    try:
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
            chroma_service.upsert_batch(batch)

        return jsonify({
            "success": True,
            "message": f"Successfully re-indexed {len(batch)} functions into ChromaDB.",
            "total_indexed": len(batch)
        })
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@kb_bp.route("/knowledge/save-llm-answer", methods=["POST"])
def save_llm_answer():
    """
    Self-growing KB endpoint: Saves an LLM-fallback answer back into the global KB.
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
    """Lists all auto-learned functions in the KB"""
    records = KBGrowthService.list_llm_generated()
    return jsonify({"success": True, "count": len(records), "records": records})

@kb_bp.route("/knowledge/llm-generated/<int:func_id>", methods=["DELETE"])
def delete_llm_generated(func_id):
    """Admin endpoint to remove an LLM-generated record from KB"""
    success = KBGrowthService.delete_llm_generated(func_id)
    if success:
        return jsonify({"success": True, "message": f"Record {func_id} removed from KB."})
    return jsonify({"success": False, "error": "Record not found or not LLM-generated"}), 404

