import os
import uuid
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.config import Config
from app.utils.excel_validator import ExcelValidator
from app.services.excel_ingestion_service import ExcelIngestionService
from app.services.code_doc_ingestion_service import CodeDocIngestionService

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


@upload_bp.route("/knowledge-base/upload-code-or-doc", methods=["POST"])
def upload_code_or_doc():
    """
    Handles upload of source code files (.py, .cs, .ts, .js) or
    developer documentation (.md, .txt).
    Parses via AST + LLM, runs conflict check, and stages for human verification.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file attached in upload"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not CodeDocIngestionService.is_supported_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Supported: .py, .cs, .ts, .js, .jsx, .tsx, .java, .go, .cpp, .md, .txt"
        }), 400

    author_notes = request.form.get("notes", "").strip()

    Config.ensure_directories()
    original_filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_filename}"
    saved_path = os.path.join(str(Config.UPLOAD_PATH), unique_name)

    try:
        file.save(saved_path)
        logger.info(f"Source code/doc saved to {saved_path}. Running AST extraction and staging...")

        result = CodeDocIngestionService.ingest_code_or_doc_file(
            file_path=saved_path,
            original_filename=original_filename,
            author_notes=author_notes
        )

        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    except Exception as e:
        logger.exception(f"Error handling code/doc upload: {e}")
        return jsonify({"success": False, "error": f"Upload processing error: {str(e)}"}), 500

