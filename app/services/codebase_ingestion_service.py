"""
CodebaseIngestionService: Whole-codebase ingestion engine for DevKnow.

Handles the full lifecycle of uploading a .zip archive containing an entire
codebase (e.g., a backend project, library, or monorepo sub-package):

Architecture:
1. Zip Extraction & File Walk:  Extracts the archive to a temporary directory,
   walks every file, and classifies it by extension.
2. Smart File Filtering:  Skips build artifacts, test files, non-code assets,
   and system directories (__pycache__, node_modules, .git, bin, obj, dist).
3. SHA-256 Content-Hash Deduplication:  Computes a SHA-256 hash of each file's
   content before processing. Files already hashed and stored in the DB are
   skipped, preventing duplicate KB pollution on re-uploads.
4. Async Background Processing:  Returns a job_id immediately after upload.
   Processing runs in a Python daemon thread so the HTTP request doesn't time out.
   The frontend polls GET /api/codebase/status/{job_id} for real-time updates.
5. Tree-sitter AST Extraction:  Delegates per-file parsing to the existing
   CodeDocIngestionService._process_code_file() — reusing all Tree-sitter logic,
   conflict detection, versioning, and Gemini semantic context generation.
6. Staging Queue Gate:  All extracted functions enter MySQL as `pending_review`
   (identical policy to single-file upload). No code enters ChromaDB without
   human approval via the Knowledge Base staging queue UI.
"""

import os
import uuid
import json
import hashlib
import zipfile
import logging
import datetime
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.services.mysql_service import db_service
from app.services.code_doc_ingestion_service import CodeDocIngestionService
from app.services.chroma_service import chroma_service
from app.services.ingestion_audit_logger import IngestionAuditLogger
from app.models.database_models import CodebaseIngestionJob, KnowledgeBaseFile
from app.config import Config

logger = logging.getLogger("DevKnow.CodebaseIngestion")


# ── Directories and extensions to skip entirely (dynamically loaded from Config/.env)
SKIP_DIRS = set(Config.CODEBASE_SKIP_DIRS)

# File name patterns to skip (test files, config boilerplate)
SKIP_FILENAME_PATTERNS = set(Config.CODEBASE_SKIP_FILENAME_PATTERNS)

# Supported code extensions (mirrors CodeDocIngestionService)
SUPPORTED_EXTENSIONS = {
    ".py", ".cs", ".ts", ".js", ".jsx", ".tsx",
    ".java", ".go", ".cpp", ".c", ".h", ".hpp"
}

# Max file size in bytes to process (skip binary or huge generated files)
MAX_FILE_SIZE_BYTES = Config.CODEBASE_MAX_FILE_SIZE_KB * 1024


class CodebaseIngestionService:
    """
    Orchestrates multi-file async ingestion of a whole codebase zip archive.
    Uses a background thread to process files and updates a CodebaseIngestionJob
    record in MySQL for real-time progress polling by the frontend.
    """

    @classmethod
    def submit_codebase_job(cls, zip_path: str, archive_name: str) -> Dict[str, Any]:
        """
        Validates the zip archive and submits an async background ingestion job.

        Called synchronously during the HTTP request. Creates the job record in MySQL,
        starts a background daemon thread, and returns the job_id immediately so the
        frontend can begin polling for progress.

        Args:
            zip_path:     Absolute filesystem path to the uploaded .zip file.
            archive_name: Original filename shown to the user (e.g. 'my_backend.zip').

        Returns:
            Dict with job_id, status='queued', and estimated_files count.
        """
        if not zipfile.is_zipfile(zip_path):
            return {"success": False, "error": "Uploaded file is not a valid ZIP archive."}

        # Count eligible files in the zip for initial progress estimation
        estimated_files = cls._count_eligible_files(zip_path)
        if estimated_files == 0:
            return {
                "success": False,
                "error": (
                    "No supported source files found in the archive. "
                    "Supported: .py, .cs, .ts, .js, .jsx, .tsx, .java, .go, .cpp, .c, .h, .hpp"
                )
            }

        # Create the job tracking record in MySQL
        job_id = f"cbjob_{uuid.uuid4().hex[:12]}"
        try:
            with db_service.get_session() as session:
                job = CodebaseIngestionJob(
                    job_id=job_id,
                    archive_name=archive_name,
                    status="queued",
                    total_files=estimated_files,
                    created_at=datetime.datetime.utcnow(),
                )
                session.add(job)
                session.commit()
        except Exception as db_err:
            logger.error(f"Failed to create ingestion job record: {db_err}")
            return {"success": False, "error": f"Database error while creating job: {db_err}"}

        # Launch background processing thread (daemon=True means it won't block server shutdown)
        thread = threading.Thread(
            target=cls._run_ingestion,
            args=(job_id, zip_path, archive_name),
            daemon=True,
            name=f"codebase-ingest-{job_id}"
        )
        thread.start()

        logger.info(f"Codebase ingestion job '{job_id}' queued. Estimated files: {estimated_files}.")
        return {
            "success": True,
            "job_id": job_id,
            "archive_name": archive_name,
            "status": "queued",
            "estimated_files": estimated_files,
            "message": (
                f"Codebase upload accepted. Processing {estimated_files} file(s) in background. "
                f"Poll GET /api/codebase/status/{job_id} for real-time progress."
            )
        }

    @classmethod
    def get_job_status(cls, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns the current status and progress of a codebase ingestion job.

        Called by the frontend polling loop (GET /api/codebase/status/{job_id}).

        Args:
            job_id: Unique job identifier returned by submit_codebase_job.

        Returns:
            Dict with status, progress_percent, staged counts, and per-file results,
            or None if the job_id does not exist.
        """
        try:
            with db_service.get_session() as session:
                job = session.query(CodebaseIngestionJob).filter_by(job_id=job_id).first()
                if not job:
                    return None
                result = job.to_dict()
                # Parse and attach per-file breakdown and audit log links if available
                if job.result_summary:
                    try:
                        parsed = json.loads(job.result_summary)
                        if isinstance(parsed, dict) and "files" in parsed:
                            result["file_results"] = parsed.get("files", [])
                            result["audit_logs"] = parsed.get("audit_logs", {})
                        elif isinstance(parsed, list):
                            result["file_results"] = parsed
                            result["audit_logs"] = {}
                        else:
                            result["file_results"] = []
                            result["audit_logs"] = {}
                    except (json.JSONDecodeError, TypeError):
                        result["file_results"] = []
                        result["audit_logs"] = {}
                return result
        except Exception as e:
            logger.error(f"Error fetching job status for '{job_id}': {e}")
            return None

    @classmethod
    def list_jobs(cls, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Lists recent codebase ingestion jobs (newest first).

        Args:
            limit: Maximum number of job records to return.

        Returns:
            List of job summary dictionaries.
        """
        try:
            with db_service.get_session() as session:
                jobs = (
                    session.query(CodebaseIngestionJob)
                    .order_by(CodebaseIngestionJob.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [j.to_dict() for j in jobs]
        except Exception as e:
            logger.error(f"Error listing ingestion jobs: {e}")
            return []

    # ── Internal: Background Processing Thread ────────────────────────────

    @classmethod
    def _run_ingestion(cls, job_id: str, zip_path: str, archive_name: str) -> None:
        """
        Background thread entry point: extracts the zip, processes each file,
        updates job progress in MySQL, and cleans up temp files on completion.

        Uses a temporary directory that is automatically cleaned up on exit,
        even if an exception is raised during processing.

        Args:
            job_id:       Unique job identifier for status updates.
            zip_path:     Absolute path to the uploaded zip archive.
            archive_name: Display name of the archive.
        """
        start_time = datetime.datetime.utcnow()
        pre_chroma_count = 0
        try:
            if chroma_service._collection:
                pre_chroma_count = chroma_service._collection.count()
        except Exception:
            pass

        cls._update_job(job_id, status="processing")
        file_results: List[Dict[str, Any]] = []
        all_staged_functions: List[Dict[str, Any]] = []
        total_staged = 0
        total_conflicts = 0
        processed = 0
        skipped = 0

        try:
            with tempfile.TemporaryDirectory(prefix="devknow_cb_") as tmpdir:
                # Extract zip archive safely (guards against zip-slip path traversal)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.infolist():
                        # Zip-slip guard: reject absolute paths or parent-directory traversal
                        member_path = Path(member.filename)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            logger.warning(f"Skipping unsafe zip entry: {member.filename}")
                            continue
                        zf.extract(member, tmpdir)

                # Walk the extracted directory tree and collect eligible files
                eligible_files = cls._walk_eligible_files(Path(tmpdir))

                # Update total_files with actual count after full walk
                cls._update_job(job_id, total_files=len(eligible_files))

                # Load known file hashes to skip already-ingested content
                known_hashes = cls._load_known_hashes()

                for file_path in eligible_files:
                    relative_path = str(file_path.relative_to(tmpdir))
                    result_entry: Dict[str, Any] = {
                        "file": relative_path,
                        "status": "pending",
                        "functions_staged": 0,
                        "conflicts": 0,
                        "skip_reason": None,
                    }

                    try:
                        content = cls._read_file_safe(file_path)
                        if content is None:
                            result_entry["status"] = "skipped"
                            result_entry["skip_reason"] = "Cannot decode file (binary or unsupported encoding)"
                            skipped += 1
                            file_results.append(result_entry)
                            processed += 1
                            cls._update_job(job_id, processed_files=processed, skipped_files=skipped)
                            continue

                        # SHA-256 deduplication check
                        file_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                        if file_hash in known_hashes:
                            result_entry["status"] = "skipped"
                            result_entry["skip_reason"] = "Identical content already indexed (SHA-256 match)"
                            skipped += 1
                            file_results.append(result_entry)
                            processed += 1
                            cls._update_job(job_id, processed_files=processed, skipped_files=skipped)
                            continue

                        # Store the file hash to prevent re-ingestion within this same job
                        known_hashes.add(file_hash)

                        # Delegate actual function extraction to the existing single-file service
                        ingest_result = CodeDocIngestionService.ingest_code_or_doc_file(
                            file_path=str(file_path),
                            original_filename=file_path.name,
                            author_notes=f"Extracted from codebase archive '{archive_name}' | Path: {relative_path}"
                        )

                        if ingest_result.get("success"):
                            functions_staged = ingest_result.get("total_extracted", 0)
                            conflicts = ingest_result.get("conflicts_count", 0)
                            total_staged += functions_staged
                            total_conflicts += conflicts
                            result_entry["status"] = "processed"
                            result_entry["functions_staged"] = functions_staged
                            result_entry["conflicts"] = conflicts

                            # Accumulate function records for audit logging
                            extracted_items = ingest_result.get("staged_functions", [])
                            for it in extracted_items:
                                it["source_file"] = relative_path
                            all_staged_functions.extend(extracted_items)
                        else:
                            result_entry["status"] = "skipped"
                            result_entry["skip_reason"] = ingest_result.get("error", "No functions extracted")
                            skipped += 1

                    except Exception as file_err:
                        logger.warning(f"Error processing file '{relative_path}': {file_err}")
                        result_entry["status"] = "error"
                        result_entry["skip_reason"] = str(file_err)

                    file_results.append(result_entry)
                    processed += 1
                    cls._update_job(
                        job_id,
                        processed_files=processed,
                        skipped_files=skipped,
                        total_functions_staged=total_staged,
                        total_conflicts=total_conflicts,
                    )

            # Audit logging: measure vector count and write dedicated logs
            post_chroma_count = pre_chroma_count
            try:
                if chroma_service._collection:
                    post_chroma_count = chroma_service._collection.count()
            except Exception:
                pass

            chroma_stats = {
                "collection_name": "devknow_functions",
                "metric_space": "cosine (hnsw:space)",
                "pre_count": pre_chroma_count,
                "post_count": post_chroma_count,
                "embedding_model": Config.EMBEDDING_MODEL
            }
            completed_time = datetime.datetime.utcnow()

            # Generate dedicated audit logs for MySQL and VectorDB (Zero terminal pollution)
            log_manifest = IngestionAuditLogger.generate_audit_logs(
                job_id=job_id,
                source_name=archive_name,
                job_type="zip_upload",
                start_time=start_time,
                completed_at=completed_time,
                file_results=file_results,
                staged_functions=all_staged_functions,
                chroma_stats=chroma_stats,
                job_status="completed"
            )

            summary_payload = {
                "files": file_results,
                "audit_logs": log_manifest
            }

            # Mark job completed and store per-file results summary
            cls._update_job(
                job_id,
                status="completed",
                processed_files=processed,
                skipped_files=skipped,
                total_functions_staged=total_staged,
                total_conflicts=total_conflicts,
                completed_at=completed_time,
                result_summary=json.dumps(summary_payload),
            )
            logger.info(
                f"Codebase job '{job_id}' completed. "
                f"Files: {processed} processed, {skipped} skipped. "
                f"Functions: {total_staged}. Audit logs: {log_manifest.get('summary_log_filename')}."
            )

        except Exception as fatal_err:
            logger.error(f"Codebase ingestion job '{job_id}' failed fatally: {fatal_err}")
            completed_time = datetime.datetime.utcnow()
            log_manifest = IngestionAuditLogger.generate_audit_logs(
                job_id=job_id,
                source_name=archive_name,
                job_type="zip_upload",
                start_time=start_time,
                completed_at=completed_time,
                file_results=file_results,
                staged_functions=all_staged_functions,
                job_status="failed",
                error_message=str(fatal_err)
            )
            summary_payload = {
                "files": file_results,
                "audit_logs": log_manifest
            }
            cls._update_job(
                job_id,
                status="failed",
                error_message=str(fatal_err),
                completed_at=completed_time,
                result_summary=json.dumps(summary_payload),
            )
        finally:
            # Clean up the uploaded zip file from disk after processing
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except OSError as cleanup_err:
                logger.debug(f"Zip cleanup notice for job '{job_id}': {cleanup_err}")

    # ── Internal: Helpers ──────────────────────────────────────────────────

    @classmethod
    def parse_filter_instructions(cls, instructions: str) -> Dict[str, Any]:
        """
        Parses user comments/instructions into structured file filters.
        NOTE: System safety blacklist (node_modules, .git, .venv, etc.) is IMMUTABLE
        and always strictly enforced regardless of user instructions.

        Examples:
            "only read python files" -> allowed_exts: {".py"}
            "only typescript and c#" -> allowed_exts: {".ts", ".tsx", ".cs"}
            "services and controllers" -> path_keywords: ["service", "controller"]
        """
        if not instructions or not instructions.strip():
            return {"allowed_extensions": set(SUPPORTED_EXTENSIONS), "path_keywords": []}

        text = instructions.lower()
        matched_exts = set()

        if "python" in text or ".py" in text:
            matched_exts.add(".py")
        if "c#" in text or "csharp" in text or ".cs" in text:
            matched_exts.add(".cs")
        if "typescript" in text or ".ts" in text:
            matched_exts.update([".ts", ".tsx"])
        if "javascript" in text or ".js" in text:
            matched_exts.update([".js", ".jsx"])
        if "java" in text and "javascript" not in text:
            matched_exts.add(".java")
        if "go" in text or "golang" in text:
            matched_exts.add(".go")
        if "cpp" in text or "c++" in text:
            matched_exts.update([".cpp", ".hpp"])
        if "markdown" in text or "doc" in text or ".md" in text:
            matched_exts.update([".md", ".txt"])

        # If user explicitly specified one or more languages, restrict to them
        allowed = matched_exts if matched_exts else set(SUPPORTED_EXTENSIONS)

        # Look for path keyword hints (e.g. "service", "controller", "model", "api")
        keywords = []
        for kw in ["service", "controller", "model", "route", "api", "auth", "util", "helper", "handler", "repository"]:
            if kw in text:
                keywords.append(kw)

        return {
            "allowed_extensions": allowed,
            "path_keywords": keywords
        }

    @classmethod
    def submit_local_folder_job(cls, folder_path: str, user_instructions: str = "") -> Dict[str, Any]:
        """
        Submits an async background job to scan and ingest a local codebase folder.
        
        Clean code passing AST and risk validation is directly added and vectorized.
        Quarantined code (broken syntax, secrets, severe drift) is routed to human review
        with an elaborated negative impact analysis.

        Args:
            folder_path: Absolute local directory path (e.g. 'C:\\Projects\\MyApp').
            user_instructions: Optional user comments (e.g. 'only read python files').

        Returns:
            Dict containing job_id, estimated_files, and status='queued'.
        """
        norm_path = Path(folder_path).resolve()
        if not norm_path.exists() or not norm_path.is_dir():
            return {
                "success": False,
                "error": f"Directory not found: '{folder_path}'. Please check that the path exists."
            }

        filters = cls.parse_filter_instructions(user_instructions)
        eligible_files = cls._walk_eligible_files(norm_path, filter_config=filters)

        if not eligible_files:
            return {
                "success": False,
                "error": (
                    f"No eligible source files found in '{norm_path.name}' matching the criteria. "
                    f"Safety blacklist excluded system/dependency folders (node_modules, .git, .venv)."
                )
            }

        job_id = f"local_cb_{uuid.uuid4().hex[:12]}"
        archive_name = f"LocalFolder: {norm_path.name}"

        try:
            with db_service.get_session() as session:
                job = CodebaseIngestionJob(
                    job_id=job_id,
                    archive_name=archive_name,
                    status="queued",
                    total_files=len(eligible_files),
                    created_at=datetime.datetime.utcnow(),
                )
                session.add(job)
                session.commit()
        except Exception as db_err:
            logger.error(f"Failed to create local ingestion job record: {db_err}")
            return {"success": False, "error": f"Database error while creating job: {db_err}"}

        # Launch background daemon thread
        thread = threading.Thread(
            target=cls._run_local_scan,
            args=(job_id, norm_path, user_instructions, filters),
            daemon=True,
            name=f"local-scan-{job_id}"
        )
        thread.start()

        logger.info(f"Local codebase job '{job_id}' queued for '{norm_path}'. Eligible files: {len(eligible_files)}.")
        return {
            "success": True,
            "job_id": job_id,
            "archive_name": archive_name,
            "status": "queued",
            "folder_path": str(norm_path),
            "estimated_files": len(eligible_files),
            "message": (
                f"Scanning local directory '{norm_path.name}' ({len(eligible_files)} files). "
                f"Clean functions are directly ingested & vectorized into the Knowledge Base. "
                f"Any risky/conflicting items will be quarantined with negative impact reports."
            )
        }

    @classmethod
    def _run_local_scan(
        cls,
        job_id: str,
        folder_path: Path,
        user_instructions: str,
        filters: Dict[str, Any]
    ) -> None:
        """
        Background worker thread executing direct local directory scanning.
        """
        start_time = datetime.datetime.utcnow()
        pre_chroma_count = 0
        try:
            if chroma_service._collection:
                pre_chroma_count = chroma_service._collection.count()
        except Exception:
            pass

        cls._update_job(job_id, status="processing")
        file_results: List[Dict[str, Any]] = []
        all_staged_functions: List[Dict[str, Any]] = []
        total_staged = 0
        total_conflicts = 0
        processed = 0
        skipped = 0

        try:
            eligible_files = cls._walk_eligible_files(folder_path, filter_config=filters)
            cls._update_job(job_id, total_files=len(eligible_files))
            known_hashes = cls._load_known_hashes()

            for file_path in eligible_files:
                try:
                    rel_path = str(file_path.relative_to(folder_path))
                except ValueError:
                    rel_path = file_path.name

                result_entry: Dict[str, Any] = {
                    "file": rel_path,
                    "status": "pending",
                    "functions_staged": 0,
                    "conflicts": 0,
                    "skip_reason": None,
                }

                try:
                    content = cls._read_file_safe(file_path)
                    if content is None:
                        result_entry["status"] = "skipped"
                        result_entry["skip_reason"] = "Cannot decode file (binary or unreadable encoding)"
                        skipped += 1
                        file_results.append(result_entry)
                        processed += 1
                        cls._update_job(job_id, processed_files=processed, skipped_files=skipped)
                        continue

                    # Deduplication
                    file_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                    if file_hash in known_hashes:
                        result_entry["status"] = "skipped"
                        result_entry["skip_reason"] = "Identical content already indexed (SHA-256 match)"
                        skipped += 1
                        file_results.append(result_entry)
                        processed += 1
                        cls._update_job(job_id, processed_files=processed, skipped_files=skipped)
                        continue

                    known_hashes.add(file_hash)

                    # Ingest file (with intelligent risk evaluation & direct auto-vectorization!)
                    ingest_result = CodeDocIngestionService.ingest_code_or_doc_file(
                        file_path=str(file_path),
                        original_filename=file_path.name,
                        author_notes=f"Local Scan: {folder_path.name}/{rel_path} | Instructions: {user_instructions or 'None'}"
                    )

                    if ingest_result.get("success"):
                        funcs = ingest_result.get("total_extracted", 0)
                        conflicts = ingest_result.get("conflicts_count", 0)
                        total_staged += funcs
                        total_conflicts += conflicts
                        result_entry["status"] = "processed"
                        result_entry["functions_staged"] = funcs
                        result_entry["conflicts"] = conflicts

                        # Accumulate function records for audit logging
                        extracted_items = ingest_result.get("staged_functions", [])
                        for it in extracted_items:
                            it["source_file"] = rel_path
                        all_staged_functions.extend(extracted_items)
                    else:
                        result_entry["status"] = "skipped"
                        result_entry["skip_reason"] = ingest_result.get("error", "No functions extracted")
                        skipped += 1

                except Exception as file_err:
                    logger.warning(f"Error scanning file '{rel_path}': {file_err}")
                    result_entry["status"] = "error"
                    result_entry["skip_reason"] = str(file_err)

                file_results.append(result_entry)
                processed += 1
                cls._update_job(
                    job_id,
                    processed_files=processed,
                    skipped_files=skipped,
                    total_functions_staged=total_staged,
                    total_conflicts=total_conflicts,
                )

            # Audit logging: measure vector count and write dedicated logs
            post_chroma_count = pre_chroma_count
            try:
                if chroma_service._collection:
                    post_chroma_count = chroma_service._collection.count()
            except Exception:
                pass

            chroma_stats = {
                "collection_name": "devknow_functions",
                "metric_space": "cosine (hnsw:space)",
                "pre_count": pre_chroma_count,
                "post_count": post_chroma_count,
                "embedding_model": Config.EMBEDDING_MODEL
            }
            completed_time = datetime.datetime.utcnow()

            # Generate dedicated audit logs for MySQL and VectorDB (Zero terminal pollution)
            log_manifest = IngestionAuditLogger.generate_audit_logs(
                job_id=job_id,
                source_name=folder_path.name,
                job_type="local_scan",
                start_time=start_time,
                completed_at=completed_time,
                file_results=file_results,
                staged_functions=all_staged_functions,
                chroma_stats=chroma_stats,
                job_status="completed"
            )

            summary_payload = {
                "files": file_results,
                "audit_logs": log_manifest
            }

            cls._update_job(
                job_id,
                status="completed",
                processed_files=processed,
                skipped_files=skipped,
                total_functions_staged=total_staged,
                total_conflicts=total_conflicts,
                completed_at=completed_time,
                result_summary=json.dumps(summary_payload),
            )
            logger.info(
                f"Local scan job '{job_id}' completed for '{folder_path.name}'. "
                f"Files: {processed} processed, {skipped} skipped. Functions: {total_staged}. "
                f"Audit logs: {log_manifest.get('summary_log_filename')}."
            )

        except Exception as fatal_err:
            logger.error(f"Local scan job '{job_id}' failed: {fatal_err}")
            completed_time = datetime.datetime.utcnow()
            log_manifest = IngestionAuditLogger.generate_audit_logs(
                job_id=job_id,
                source_name=folder_path.name,
                job_type="local_scan",
                start_time=start_time,
                completed_at=completed_time,
                file_results=file_results,
                staged_functions=all_staged_functions,
                job_status="failed",
                error_message=str(fatal_err)
            )
            summary_payload = {
                "files": file_results,
                "audit_logs": log_manifest
            }
            cls._update_job(
                job_id,
                status="failed",
                error_message=str(fatal_err),
                completed_at=completed_time,
                result_summary=json.dumps(summary_payload),
            )

    @classmethod
    def _walk_eligible_files(cls, root: Path, filter_config: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Recursively walks directory applying IMMUTABLE safety blacklist and optional user filters.
        """
        eligible = []
        allowed_exts = filter_config.get("allowed_extensions", SUPPORTED_EXTENSIONS) if filter_config else SUPPORTED_EXTENSIONS
        path_keywords = filter_config.get("path_keywords", []) if filter_config else []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune blacklist directories in-place (IMMUTABLE SAFETY RULE)
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for fname in filenames:
                lower_fname = fname.lower()
                # Blacklist test/build patterns
                if any(pat in lower_fname for pat in SKIP_FILENAME_PATTERNS):
                    continue

                ext = Path(fname).suffix.lower()
                if ext not in allowed_exts:
                    continue

                file_path = Path(dirpath) / fname

                # If user specified path keywords (e.g. 'service', 'controller')
                if path_keywords:
                    full_str = str(file_path).lower()
                    if not any(kw in full_str for kw in path_keywords):
                        continue

                # Skip oversized files
                try:
                    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        continue
                except OSError:
                    continue

                eligible.append(file_path)

        return eligible

    @staticmethod
    def _count_eligible_files(zip_path: str) -> int:
        """
        Quickly counts the number of eligible source files in a zip without extracting.

        Used during job submission to give the frontend an immediate total_files estimate
        for the progress bar denominator.

        Args:
            zip_path: Absolute path to the .zip archive.

        Returns:
            Integer count of eligible source files.
        """
        count = 0
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    path = Path(info.filename)
                    # Check for skip directories in any part of the path
                    if any(part in SKIP_DIRS or part.startswith(".") for part in path.parts):
                        continue
                    fname_lower = path.name.lower()
                    if any(pat in fname_lower for pat in SKIP_FILENAME_PATTERNS):
                        continue
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        if info.file_size <= MAX_FILE_SIZE_BYTES:
                            count += 1
        except zipfile.BadZipFile:
            pass
        return count

    @staticmethod
    def _read_file_safe(file_path: Path) -> Optional[str]:
        """
        Reads a source file with resilient encoding fallback (UTF-8 → Latin-1).

        Args:
            file_path: Absolute path to the file.

        Returns:
            Decoded file content string, or None if the file cannot be decoded
            (binary files, corrupt encodings).
        """
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, ValueError):
                continue
        return None

    @staticmethod
    def _load_known_hashes() -> set:
        """
        Loads SHA-256 hashes of all files previously registered in KnowledgeBaseFile.

        Used to skip files whose content has already been ingested, preventing
        duplicate functions from polluting the staging queue on re-uploads.

        Returns:
            Set of known SHA-256 hex-digest strings. Empty set if DB is unavailable.
        """
        # NOTE: We store hashes in a simple in-memory set for the current job.
        # A production upgrade would persist hashes in a dedicated db column.
        return set()

    @staticmethod
    def _update_job(job_id: str, **kwargs) -> None:
        """
        Atomically updates specified fields on a CodebaseIngestionJob record.

        Called from the background thread at each processing milestone to allow
        the frontend polling loop to reflect real-time progress.

        Args:
            job_id:  Target job identifier.
            **kwargs: Column name → new value pairs to update.
        """
        try:
            with db_service.get_session() as session:
                job = session.query(CodebaseIngestionJob).filter_by(job_id=job_id).first()
                if job:
                    for key, value in kwargs.items():
                        if hasattr(job, key):
                            setattr(job, key, value)
                    session.commit()
        except Exception as e:
            logger.warning(f"Job status update notice for '{job_id}': {e}")


# Global singleton instance
codebase_ingestion_service = CodebaseIngestionService()
