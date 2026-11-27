"""
Codebase Upload API Routes for DevKnow.

Provides endpoints for uploading a whole codebase .zip archive and polling
the async ingestion job status. All responses follow the standard DevKnow
API envelope: { success, status, status_code, message, data }.

Endpoints:
    POST /api/codebase/upload
        Accepts a .zip archive of source code. Submits an async background
        ingestion job and returns a job_id for polling.

    GET  /api/codebase/status/<job_id>
        Returns real-time progress and results for a given ingestion job.

    GET  /api/codebase/jobs
        Lists recent codebase ingestion jobs (newest first).
"""

import os
import logging
from pathlib import Path
from flask import Blueprint, request

from app.config import Config
from app.services.codebase_ingestion_service import CodebaseIngestionService
from app.utils.response_helper import success_response, error_response

logger = logging.getLogger("DevKnow.CodebaseRoutes")

codebase_bp = Blueprint("codebase", __name__)

# Supported archive MIME types and extensions
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}
MAX_ZIP_SIZE_MB = 200  # Maximum .zip size in megabytes


@codebase_bp.route("/codebase/upload", methods=["POST"])
def upload_codebase():
    """
    Accepts a whole codebase .zip archive and starts an async ingestion job.

    The file is saved to disk immediately and a background thread handles all
    extraction, Tree-sitter parsing, and staging. The HTTP response returns
    within milliseconds with a job_id the frontend uses to poll progress.

    Request:
        multipart/form-data with field 'file' containing the .zip archive.

    Returns:
        202 Accepted: { job_id, status, estimated_files, message }
        400 Bad Request: missing file, wrong extension, or no code files found
        413 Content Too Large: archive exceeds 200 MB
    """
    if "file" not in request.files:
        return error_response(
            message="No file provided. Upload a .zip archive with field name 'file'.",
            status_code=400
        )

    file = request.files["file"]

    if not file.filename:
        return error_response(message="Uploaded file has no filename.", status_code=400)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_ARCHIVE_EXTENSIONS:
        return error_response(
            message=f"Unsupported archive format '{ext}'. Only .zip archives are accepted.",
            status_code=400
        )

    # Save the archive to the uploads directory for the background thread to read
    safe_name = f"codebase_{os.urandom(6).hex()}{ext}"
    save_path = Config.UPLOAD_PATH / safe_name
    Config.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)

    try:
        file.save(str(save_path))
    except OSError as e:
        logger.error(f"Failed to save uploaded archive '{file.filename}': {e}")
        return error_response(message=f"Failed to save uploaded file: {e}", status_code=500)

    # Reject archives that exceed the configured maximum size
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_ZIP_SIZE_MB:
        try:
            save_path.unlink(missing_ok=True)
        except OSError:
            pass
        return error_response(
            message=f"Archive too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_ZIP_SIZE_MB} MB.",
            status_code=413
        )

    logger.info(f"Received codebase archive '{file.filename}' ({file_size_mb:.2f} MB). Submitting job...")

    # Submit async ingestion job
    result = CodebaseIngestionService.submit_codebase_job(
        zip_path=str(save_path),
        archive_name=file.filename
    )

    if not result.get("success"):
        # Clean up the saved file if job submission failed
        try:
            save_path.unlink(missing_ok=True)
        except OSError:
            pass
        return error_response(
            message=result.get("error", "Failed to process the codebase archive."),
            status_code=400
        )

    return success_response(
        message=result["message"],
        status_code=202,
        status="accepted",
        job_id=result["job_id"],
        archive_name=result["archive_name"],
        estimated_files=result["estimated_files"],
        poll_url=f"/api/codebase/status/{result['job_id']}"
    )


@codebase_bp.route("/codebase/pick-folder", methods=["POST", "GET"])
def pick_folder():
    """
    Opens the native OS folder picker dialog on the host system to let the user
    select a local directory. Returns the absolute path of the selected folder.
    Prevents browser upload popups ('Upload 50k+ files?') and auto-fills the input.
    """
    import subprocess
    import sys

    try:
        # Script to open native Windows/Tk folder selection dialog
        script = (
            "import tkinter as tk; "
            "from tkinter import filedialog; "
            "root = tk.Tk(); "
            "root.withdraw(); "
            "root.wm_attributes('-topmost', 1); "
            "path = filedialog.askdirectory(title='Select Codebase Folder for DevKnow'); "
            "root.destroy(); "
            "print(path)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120
        )
        selected_path = proc.stdout.strip()
        if not selected_path:
            return success_response(
                message="Folder selection canceled.",
                folder_path="",
                canceled=True
            )

        norm_path = os.path.normcase(os.path.normpath(selected_path)) if os.name == 'nt' else os.path.normpath(selected_path)
        # Ensure proper casing for Windows drive letter if available
        if len(norm_path) >= 2 and norm_path[1] == ':':
            norm_path = norm_path[0].upper() + norm_path[1:]

        return success_response(
            message=f"Folder selected: {norm_path}",
            folder_path=norm_path,
            canceled=False
        )
    except subprocess.TimeoutExpired:
        return error_response(message="Folder selection timed out.", status_code=408)
    except Exception as e:
        logger.warning(f"Native folder picker notice: {e}")
        return error_response(message=f"Could not open native folder dialog: {e}", status_code=500)


@codebase_bp.route("/codebase/scan-local", methods=["POST"])
def scan_local_folder():
    """
    Directly scans a local codebase directory on the host machine.
    Enforces the immutable safety blacklist and optional user filter instructions.

    Clean code functions are directly verified and indexed in ChromaDB.
    Risky code (broken syntax, secrets, severe drift) is quarantined with an
    elaborated negative impact report for the admin.

    Request:
        JSON or Form-Data:
        {
            "folder_path": "C:\\path\\to\\project",
            "user_instructions": "only read python files" (optional)
        }
    """
    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}
    if not data and request.data:
        try:
            import json
            data = json.loads(request.data.decode('utf-8', errors='ignore'))
        except Exception:
            pass
    folder_path = (data.get("folder_path") or request.form.get("folder_path") or "").strip()
    user_instructions = (data.get("user_instructions") or request.form.get("user_instructions") or "").strip()

    if not folder_path:
        return error_response(
            message="No folder path provided. Please specify 'folder_path' (e.g. 'C:\\Projects\\MyBackend').",
            status_code=400
        )

    result = CodebaseIngestionService.submit_local_folder_job(
        folder_path=folder_path,
        user_instructions=user_instructions
    )

    if not result.get("success"):
        return error_response(
            message=result.get("error", "Failed to scan local folder."),
            status_code=400
        )

    return success_response(
        message=result["message"],
        status_code=202,
        status="accepted",
        job_id=result["job_id"],
        archive_name=result["archive_name"],
        folder_path=result["folder_path"],
        estimated_files=result["estimated_files"],
        poll_url=f"/api/codebase/status/{result['job_id']}"
    )


@codebase_bp.route("/codebase/upload-folder", methods=["POST"])
def upload_folder_directory():
    """
    Accepts files selected via the browser's directory picker (webkitdirectory).
    Saves them preserving folder structure and submits a local scan job.
    """
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or len(uploaded_files) == 0:
        return error_response(message="No files selected.", status_code=400)

    user_instructions = (request.form.get("user_instructions") or "").strip()
    folder_name = (request.form.get("folder_name") or "Uploaded_Folder").strip()

    # Save to a dedicated staging folder
    import tempfile
    import uuid
    dest_dir = Config.UPLOAD_PATH / f"folder_{uuid.uuid4().hex[:8]}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for f in uploaded_files:
        # Browser sends relative path in form or filename
        rel_path_str = f.filename.replace("\\", "/")
        target_path = dest_dir / rel_path_str
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            f.save(str(target_path))
            saved_count += 1
        except Exception as e:
            logger.debug(f"Save file notice: {e}")

    result = CodebaseIngestionService.submit_local_folder_job(
        folder_path=str(dest_dir),
        user_instructions=user_instructions
    )

    if not result.get("success"):
        return error_response(message=result.get("error", "Failed to scan uploaded folder."), status_code=400)

    return success_response(
        message=result["message"],
        status_code=202,
        status="accepted",
        job_id=result["job_id"],
        archive_name=f"Folder: {folder_name}",
        estimated_files=result["estimated_files"],
        poll_url=f"/api/codebase/status/{result['job_id']}"
    )


@codebase_bp.route("/codebase/status/<string:job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Returns real-time status and progress of a codebase ingestion job.

    Poll this endpoint every 2-3 seconds while status is 'queued' or 'processing'.
    Stop polling when status is 'completed' or 'failed'.

    Args:
        job_id: Unique job identifier returned by POST /api/codebase/upload.

    Returns:
        200 OK:  { job_id, status, progress_percent, total_files, processed_files,
                   skipped_files, total_functions_staged, total_conflicts, file_results }
        404:     job_id not found
    """
    job_data = CodebaseIngestionService.get_job_status(job_id)
    if job_data is None:
        return error_response(
            message=f"Ingestion job '{job_id}' not found.",
            status_code=404
        )

    return success_response(
        message=f"Job '{job_id}' is {job_data.get('status', 'unknown')}.",
        **job_data
    )


@codebase_bp.route("/codebase/jobs", methods=["GET"])
def list_jobs():
    """
    Lists recent codebase ingestion jobs with status and summary metrics.

    Query Parameters:
        limit (int): Maximum number of jobs to return (default: 20, max: 100).

    Returns:
        200 OK: { jobs: [...], total: N }
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except (ValueError, TypeError):
        limit = 20

    jobs = CodebaseIngestionService.list_jobs(limit=limit)

    return success_response(
        message=f"Retrieved {len(jobs)} ingestion job(s).",
        jobs=jobs,
        total=len(jobs)
    )


@codebase_bp.route("/codebase/logs/<string:job_id>", methods=["GET"])
def get_job_log(job_id: str):
    """
    Returns the contents of a dedicated audit log file for an ingestion job.

    Query Parameters:
        type: 'mysql' | 'vectordb' | 'summary' | 'json' (default: 'summary')
    """
    log_type = request.args.get("type", "summary").lower()
    logs_dir = getattr(Config, "INGESTION_LOGS_PATH", Config.LOGS_PATH / "ingestion")

    filename_map = {
        "mysql": f"ingestion_{job_id}_mysql.log",
        "vectordb": f"ingestion_{job_id}_vectordb.log",
        "summary": f"ingestion_{job_id}_summary.log",
        "json": f"ingestion_{job_id}_audit.json"
    }

    target_name = filename_map.get(log_type, f"ingestion_{job_id}_summary.log")
    target_path = logs_dir / target_name

    if not target_path.exists():
        return error_response(
            message=f"Log file '{target_name}' not found for job '{job_id}'.",
            status_code=404
        )

    try:
        content = target_path.read_text(encoding="utf-8")
        if log_type == "json":
            import json
            return success_response(
                message=f"Audit JSON manifest for '{job_id}'.",
                filename=target_name,
                job_id=job_id,
                manifest=json.loads(content)
            )

        return success_response(
            message=f"Audit log '{target_name}' retrieved successfully.",
            filename=target_name,
            job_id=job_id,
            log_type=log_type,
            content=content
        )
    except Exception as e:
        logger.error(f"Error reading log file '{target_name}': {e}")
        return error_response(message=f"Error reading log file: {e}", status_code=500)

