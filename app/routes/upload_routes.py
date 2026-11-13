import os
import uuid
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.config import Config
from app.utils.excel_validator import ExcelValidator
from app.services.excel_ingestion_service import ExcelIngestionService

upload_bp = Blueprint("upload", __name__)
logger = logging.getLogger("DevKnow.UploadRoutes")

@upload_bp.route("/knowledge-base/upload", methods=["POST"])
def upload_knowledge_base():
    """
    Handles Excel knowledge base file upload.
    Validates file, writes to disk, runs ingestion, returns summary.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not ExcelValidator.allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file extension. Only .xlsx and .xls files are supported."
        }), 400

    Config.ensure_directories()
    original_filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_filename}"
    saved_path = os.path.join(str(Config.UPLOAD_PATH), unique_name)

    try:
        file.save(saved_path)
        logger.info(f"File saved to {saved_path}. Starting ingestion...")

        summary = ExcelIngestionService.ingest_excel_file(
            file_path=saved_path,
            original_filename=original_filename
        )

        status_code = 200 if summary.get("success") else 400
        return jsonify(summary), status_code

    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        return jsonify({"success": False, "error": f"Upload processing error: {str(e)}"}), 500

