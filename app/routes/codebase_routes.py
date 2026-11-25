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
