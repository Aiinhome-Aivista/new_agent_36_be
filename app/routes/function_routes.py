import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import or_

from app.services.mysql_service import db_service
from app.models.database_models import FunctionRecord

function_bp = Blueprint("functions", __name__)
logger = logging.getLogger("DevKnow.FunctionRoutes")

@function_bp.route("/functions", methods=["GET"])
def list_functions():
    """
    Searchable function browser endpoint.
    Filters: query, language, framework, source (excel/llm_generated), page, limit.
    """
    query_str = request.args.get("q", "").strip()
    language = request.args.get("language", "").strip().lower()
    framework = request.args.get("framework", "").strip().lower()
    source = request.args.get("source", "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    offset = (page - 1) * limit

    with db_service.get_session() as session:
        query = session.query(FunctionRecord)

        if query_str:
            search_pattern = f"%{query_str}%"
            query = query.filter(
                or_(
                    FunctionRecord.function_name.ilike(search_pattern),
                    FunctionRecord.rule_version_name.ilike(search_pattern),
                    FunctionRecord.description.ilike(search_pattern),
                    FunctionRecord.function_header.ilike(search_pattern)
                )
            )

        if language and language != "all":
            query = query.filter(FunctionRecord.programming_language == language)

        if framework and framework != "all":
            query = query.filter(FunctionRecord.framework == framework)

        if source and source != "all":
            query = query.filter(FunctionRecord.knowledge_source == source)

        total_count = query.count()
        records = query.order_by(FunctionRecord.created_at.desc()).offset(offset).limit(limit).all()

        return jsonify({
            "success": True,
            "total": total_count,
            "page": page,
            "limit": limit,
            "pages": (total_count + limit - 1) // limit,
            "functions": [r.to_dict() for r in records]
        })

@function_bp.route("/functions/<int:func_id>", methods=["GET"])
def get_function(func_id):
    """Retrieves exact original function implementation and metadata from MySQL"""
    with db_service.get_session() as session:
        record = session.query(FunctionRecord).filter_by(id=func_id).first()
        if not record:
            return jsonify({"success": False, "error": f"Function with ID {func_id} not found"}), 404

        return jsonify({
            "success": True,
            "function": record.to_dict()
        })

