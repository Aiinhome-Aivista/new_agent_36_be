import logging
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.routes.upload_routes import upload_bp
from app.routes.knowledge_base_routes import kb_bp
from app.routes.function_routes import function_bp
from app.routes.chat_routes import chat_bp
from app.routes.admin_routes import admin_bp

def create_app():
    # Setup structured logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("DevKnow")
    logger.info("Initializing DevKnow Flask Application...")

    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for all routes under API_PREFIX
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    # Register Blueprints
    prefix = Config.API_PREFIX
    app.register_blueprint(upload_bp, url_prefix=prefix)
    app.register_blueprint(kb_bp, url_prefix=prefix)
    app.register_blueprint(function_bp, url_prefix=prefix)
    app.register_blueprint(chat_bp, url_prefix=prefix)
    app.register_blueprint(admin_bp, url_prefix=prefix)

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

    logger.info("DevKnow Flask Application ready.")
    return app

