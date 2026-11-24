"""
Admin & Diagnostics Routes Blueprint for DevKnow.

Provides operational monitoring and system diagnostic endpoints:
1. `GET /api/health`:
   Validates server operational state, active database provider, embedding model,
   LLM model, and vector store collection size.
"""

import sys
import logging
from flask import Blueprint

from app.config import Config
from app.services.chroma_service import chroma_service
from app.utils.response_helper import success_response

admin_bp = Blueprint("admin", __name__)
logger = logging.getLogger("DevKnow.AdminRoutes")


@admin_bp.route("/health", methods=["GET"])
def health_check():
    """
    System health check and diagnostic information endpoint.

    Used by uptime checkers, load balancers, and developer diagnostics to verify
    application availability, active configurations, and connection counts.

    Responses:
        200: Standardized JSON response envelope containing health attributes.
    """
    return success_response(
        message="DevKnow service is healthy and operational",
        status_code=200,
        status="healthy",  # Keep status='healthy' for existing monitoring tools
        app_name=Config.APP_NAME,
        environment=Config.APP_ENV,
        db_provider=Config.DB_PROVIDER,
        gemini_model=Config.GEMINI_MODEL,
        embedding_model=Config.EMBEDDING_MODEL,
        vector_count=chroma_service.get_count(),
        python_version=sys.version
    )
