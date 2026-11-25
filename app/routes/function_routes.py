"""
Function Catalog Routes Blueprint for DevKnow.

Provides searchable catalog and detail endpoints for indexed organizational functions:
1. `GET /api/functions`:
   Paginated, multi-field search over functions by keyword, language, framework, or knowledge source.
2. `GET /api/functions/<id>`:
   Retrieves exact original function implementation, metadata, and syntax parameters.
"""

import logging
from flask import Blueprint, request
from sqlalchemy import or_

from app.services.mysql_service import db_service
from app.models.database_models import FunctionRecord
from app.utils.response_helper import success_response, error_response

function_bp = Blueprint("functions", __name__)
logger = logging.getLogger("DevKnow.FunctionRoutes")


@function_bp.route("/functions", methods=["GET"])
def list_functions():
    """
    Searchable and paginated function browser endpoint.

    Query Parameters:
        - q: (Optional) Search query matched across function_name, rule_version_name,
             description, and function_header.
        - language: (Optional) Programming language filter (e.g. 'c#', 'python', 'all').
        - framework: (Optional) Framework filter (e.g. '.NET Core', 'Flask', 'all').
        - source: (Optional) Knowledge source filter ('excel', 'llm_generated', 'code_file', 'all').
        - page: (Optional) Page number (1-based, default: 1).
        - limit: (Optional) Items per page (default: 20, max: 100).

    Responses:
        200: Standardized JSON envelope with paginated functions list.
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
        pages = (total_count + limit - 1) // limit

        return success_response(
            message=f"Retrieved {len(records)} function(s)",
            status_code=200,
            total=total_count,
            page=page,
            limit=limit,
            pages=pages,
            functions=[r.to_dict() for r in records]
        )


@function_bp.route("/functions/<int:func_id>", methods=["GET"])
def get_function(func_id: int):
    """
    Retrieves exact original function implementation and metadata from relational storage.

    Path Parameters:
        func_id: Primary key ID of the function.

    Responses:
        200: Standardized JSON envelope with FunctionRecord details.
        404: Function not found error.
    """
    with db_service.get_session() as session:
        record = session.query(FunctionRecord).filter_by(id=func_id).first()
        if not record:
            return error_response(
                message=f"Function with ID {func_id} not found",
                status_code=404
            )

        return success_response(
            message=f"Function '{record.function_name}' retrieved successfully",
            status_code=200,
            function=record.to_dict()
        )
