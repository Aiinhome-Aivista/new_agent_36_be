"""
Upload Routes Blueprint for DevKnow.

Provides HTTP endpoints for ingesting knowledge files:
1. `POST /api/knowledge-base/upload`:
   Ingests structured Excel spreadsheets (.xlsx, .xls) containing rulebook scripts,
   function headers, and code definitions.
2. `POST /api/knowledge-base/upload-code-or-doc`:
   Ingests raw source code files (.py, .cs, .ts, .js) or markdown/text documentation (.md, .txt),
   parses them using Tree-sitter AST and LLM summarization, and stages them for human verification.
"""

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
    Handles Excel knowledge base workbook uploads.

    Workflow:
    1. Validates multipart/form-data payload contains a valid file with allowed extension.
    2. Generates collision-resistant unique filename and saves to `data/uploads/`.
    3. Invokes ExcelIngestionService to validate columns, parse rows, detect deduplication,
       and batch index vectors in ChromaDB.
    4. Returns ingestion statistics (total, successful, failed, skipped, detected languages).

    Request:
        multipart/form-data with key 'file' (.xlsx or .xls)

    Responses:
        200: Successfully parsed and ingested file with detailed statistics.
        400: Missing file, invalid extension, or malformed spreadsheet columns.
        500: Server-side I/O or ingestion processing error.
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
        logger.info(f"File saved to {saved_path}. Starting Excel ingestion pipeline...")

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
    Handles upload of source code files or developer documentation.

    Supported Formats:
    - Code: .py, .cs, .ts, .js, .jsx, .tsx, .java, .go, .cpp
    - Documentation: .md, .txt

    Workflow:
    1. Saves file to disk with unique prefix.
    2. Uses Tree-sitter AST to parse and extract function signatures, lines, and complexity.
    3. Uses Gemini LLM to generate concise business logic descriptions.
    4. Enforces the Zero-Direct-Ingestion policy: stages extracted functions with
       `verification_status='pending_review'` without immediately indexing into ChromaDB.
    5. Returns staging list, conflict warnings, and generated Tree-sitter log filenames.

    Request:
        multipart/form-data:
        - file: File binary
        - notes: (Optional) Author/developer notes for context

    Responses:
        200: Successfully extracted and staged code functions or doc rules.
        400: Unsupported format or empty file.
        500: Server processing error.
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
        logger.info(f"Source code/doc saved to {saved_path}. Running Tree-sitter AST extraction and staging...")

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
