"""
DevKnow Flask Application Factory.
Sets up structured logging, configures CORS, registers all blueprint route modules,
and handles global HTTP error states.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.utils.response_helper import error_response
from app.routes.upload_routes import upload_bp
from app.routes.knowledge_base_routes import kb_bp
from app.routes.function_routes import function_bp
from app.routes.chat_routes import chat_bp
from app.routes.admin_routes import admin_bp
from app.routes.codebase_routes import codebase_bp

def create_app() -> Flask:
    """
    Application Factory Pattern for DevKnow:
    1. Ensures required disk storage directories exist.
    2. Initializes both console and rolling file logging.
    3. Creates the Flask app instance and loads configuration.
    4. Configures Cross-Origin Resource Sharing (CORS) for frontend clients.
    5. Registers modular API blueprints under the configured API prefix (e.g. /api).
    6. Registers standardized JSON error handlers (404, 500).
    
    Returns:
        Flask: The fully configured Flask application instance.
    """
    # Ensure all directories (uploads, chroma_db, logs, treesitter) exist on disk
    Config.ensure_directories()

    # Setup structured logging format for readability and automated parsing
    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 1. Console Stream Handler (real-time terminal logging)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    # 2. Persistent Rotating File Handler (stores audit trace logs in data/logs/devknow.log)
    log_file = Config.LOGS_PATH / "devknow.log"
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB per file before rotating
            backupCount=5,              # Retains the last 5 rotated log files
            encoding="utf-8"
        )
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logger = logging.getLogger("DevKnow")
    logger.info("Initializing DevKnow Flask Application...")
    logger.info(f"Persistent trace logs active at: {log_file}")

    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for frontend clients (React Vite dev server and production origins)
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    # Register API Blueprints with the common prefix (e.g., /api/chat, /api/upload)
    prefix = Config.API_PREFIX
    app.register_blueprint(upload_bp, url_prefix=prefix)
    app.register_blueprint(kb_bp, url_prefix=prefix)
    app.register_blueprint(function_bp, url_prefix=prefix)
    app.register_blueprint(chat_bp, url_prefix=prefix)
    app.register_blueprint(admin_bp, url_prefix=prefix)
    app.register_blueprint(codebase_bp, url_prefix=prefix)

    # Standardized 404 handler returning JSON instead of default HTML
    @app.errorhandler(404)
    def not_found(e):
        return error_response(message="Endpoint not found", status_code=404)

    # Standardized 500 handler logging exception details and returning clean JSON
    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return error_response(message="Internal server error", status_code=500)

    logger.info("DevKnow Flask Application ready.")
    return app

