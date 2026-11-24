"""
Admin & Diagnostics Routes Blueprint for DevKnow.

Provides operational monitoring and system diagnostic endpoints:
1. `GET /api/health`:
   Validates server operational state, active database provider, embedding model,
   LLM model, and vector store collection size.
"""

import sys
import logging
from flask import Blueprint, jsonify

from app.config import Config
from app.services.chroma_service import chroma_service

admin_bp = Blueprint("admin", __name__)
logger = logging.getLogger("DevKnow.AdminRoutes")


@admin_bp.route("/health", methods=["GET"])
def health_check():
    """
    System health check and diagnostic information endpoint.

    Used by uptime checkers, load balancers, and developer diagnostics to verify
    application availability, active configurations, and connection counts.

    Responses:
        200: JSON containing:
            - status: 'healthy'
            - app_name: Configured application name.
            - environment: Deployment environment (e.g. 'development', 'production').
            - db_provider: Active database backend ('mysql' or 'sqlite').
            - gemini_model: Active generative AI model name.
            - embedding_model: Active embedding model.
            - vector_count: Total functions currently indexed in ChromaDB.
            - python_version: Runtime Python version.
    """
    return jsonify({
        "status": "healthy",
        "app_name": Config.APP_NAME,
        "environment": Config.APP_ENV,
        "db_provider": Config.DB_PROVIDER,
        "gemini_model": Config.GEMINI_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL,
        "vector_count": chroma_service.get_count(),
        "python_version": sys.version
    })
