import os
import sys
import logging
from flask import Blueprint, jsonify

from app.config import Config
from app.services.chroma_service import chroma_service

admin_bp = Blueprint("admin", __name__)
logger = logging.getLogger("DevKnow.AdminRoutes")

@admin_bp.route("/health", methods=["GET"])
def health_check():
    """System health check and diagnostic information"""
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

