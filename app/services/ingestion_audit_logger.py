"""
IngestionAuditLogger: Comprehensive dual-store audit logging engine for DevKnow.

Produces dedicated, structured log files under `data/logs/ingestion/` for each
codebase ingestion job (both .zip uploads and local folder scans).

Key Principles:
1. Dual-Store Isolation: Generates dedicated log files for MySQL relational persistence
   and ChromaDB vector persistence independently, plus an overall summary log and JSON export.
2. Zero Terminal Pollution: All verbose metadata, function signatures, risk explanations,
   and embedding schemas are written directly to disk log files without spamming stdout/terminal.
3. Traceability: Every MySQL record ID is cross-referenced with its corresponding ChromaDB
   vector ID (`fn_{mysql_id}`), enabling immediate debugging and output auditing.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.config import Config

logger = logging.getLogger("DevKnow.IngestionAuditLogger")


class IngestionAuditLogger:
    """
    Builds dedicated audit log reports for codebase ingestion jobs.
    Generates:
        1. ingestion_<job_id>_mysql.log     - Comprehensive MySQL relational records
        2. ingestion_<job_id>_vectordb.log  - Comprehensive ChromaDB vector store records
        3. ingestion_<job_id>_summary.log   - Executive run summary & cross-store audit
        4. ingestion_<job_id>_audit.json    - Full structured JSON audit manifest
    """

    @classmethod
    def get_logs_directory(cls) -> Path:
        """Returns the ingestion logs directory, creating it if necessary."""
        logs_dir = getattr(Config, "INGESTION_LOGS_PATH", Config.LOGS_PATH / "ingestion")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    @classmethod
    def generate_audit_logs(
        cls,
        job_id: str,
        source_name: str,
        job_type: str,
        start_time: datetime.datetime,
        completed_at: datetime.datetime,
        file_results: List[Dict[str, Any]],
        staged_functions: List[Dict[str, Any]],
        chroma_stats: Optional[Dict[str, Any]] = None,
        job_status: str = "completed",
        error_message: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generates dedicated audit logs for MySQL and VectorDB and returns the file paths.

        Args:
            job_id: Unique identifier for the ingestion job.
            source_name: Archive name or folder path scanned.
            job_type: 'zip_upload' or 'local_scan'.
            start_time: Timestamp when job execution started.
            completed_at: Timestamp when job execution finished.
            file_results: Per-file results list from the crawler.
            staged_functions: All extracted function records across all files.
            chroma_stats: Vector database metrics before/after the run.
            job_status: 'completed' or 'failed'.
            error_message: Fatal error if job failed.

        Returns:
            Dict containing relative and absolute paths of the generated log files.
        """
        logs_dir = cls.get_logs_directory()
        duration_sec = round((completed_at - start_time).total_seconds(), 2)

        # Categorize functions
        verified_funcs = [f for f in staged_functions if f.get("verification_status") == "verified"]
        quarantined_funcs = [f for f in staged_functions if f.get("verification_status") == "pending_review"]
        conflict_funcs = [f for f in staged_functions if f.get("conflict_status") == "conflict_detected"]

        # ── 1. Generate MySQL Persistence Log ──────────────────────────────
        mysql_log_filename = f"ingestion_{job_id}_mysql.log"
        mysql_log_path = logs_dir / mysql_log_filename
        cls._write_mysql_log(
            path=mysql_log_path,
            job_id=job_id,
            source_name=source_name,
            job_type=job_type,
            start_time=start_time,
            completed_at=completed_at,
            duration_sec=duration_sec,
            job_status=job_status,
            error_message=error_message,
            file_results=file_results,
            staged_functions=staged_functions,
            verified_funcs=verified_funcs,
            quarantined_funcs=quarantined_funcs,
            conflict_funcs=conflict_funcs
        )

        # ── 2. Generate ChromaDB VectorDB Persistence Log ───────────────────
        vectordb_log_filename = f"ingestion_{job_id}_vectordb.log"
        vectordb_log_path = logs_dir / vectordb_log_filename
        cls._write_vectordb_log(
            path=vectordb_log_path,
            job_id=job_id,
            source_name=source_name,
            job_type=job_type,
            start_time=start_time,
            completed_at=completed_at,
            duration_sec=duration_sec,
            verified_funcs=verified_funcs,
            quarantined_funcs=quarantined_funcs,
            chroma_stats=chroma_stats or {}
        )

        # ── 3. Generate Consolidated Summary Log ────────────────────────────
        summary_log_filename = f"ingestion_{job_id}_summary.log"
        summary_log_path = logs_dir / summary_log_filename
        cls._write_summary_log(
            path=summary_log_path,
            job_id=job_id,
            source_name=source_name,
            job_type=job_type,
            start_time=start_time,
            completed_at=completed_at,
            duration_sec=duration_sec,
            job_status=job_status,
            error_message=error_message,
            file_results=file_results,
            staged_functions=staged_functions,
            verified_funcs=verified_funcs,
            quarantined_funcs=quarantined_funcs,
            conflict_funcs=conflict_funcs,
            mysql_log_name=mysql_log_filename,
            vectordb_log_name=vectordb_log_filename
        )

        # ── 4. Generate Machine-Readable JSON Manifest ──────────────────────
        json_filename = f"ingestion_{job_id}_audit.json"
        json_path = logs_dir / json_filename
        cls._write_json_manifest(
            path=json_path,
            job_id=job_id,
            source_name=source_name,
            job_type=job_type,
            start_time=start_time,
            completed_at=completed_at,
            duration_sec=duration_sec,
            job_status=job_status,
            error_message=error_message,
            file_results=file_results,
            staged_functions=staged_functions,
            verified_funcs=verified_funcs,
            quarantined_funcs=quarantined_funcs,
            conflict_funcs=conflict_funcs,
            chroma_stats=chroma_stats or {}
        )

        # Log concise confirmation to server log WITHOUT printing huge dumps to terminal
        logger.info(
            f"Codebase Ingestion Audit Logs created for '{job_id}':\n"
            f"  - MySQL Log    : {mysql_log_filename}\n"
            f"  - VectorDB Log : {vectordb_log_filename}\n"
            f"  - Summary Log  : {summary_log_filename}\n"
            f"  - JSON Manifest: {json_filename}"
        )

        return {
            "mysql_log": str(mysql_log_path),
            "vectordb_log": str(vectordb_log_path),
            "summary_log": str(summary_log_path),
            "json_manifest": str(json_path),
            "mysql_log_filename": mysql_log_filename,
            "vectordb_log_filename": vectordb_log_filename,
            "summary_log_filename": summary_log_filename,
            "json_filename": json_filename
        }

    # ── Internal Log Writers ──────────────────────────────────────────────────

    @classmethod
    def _write_mysql_log(
        cls,
        path: Path,
        job_id: str,
        source_name: str,
        job_type: str,
        start_time: datetime.datetime,
        completed_at: datetime.datetime,
        duration_sec: float,
        job_status: str,
        error_message: Optional[str],
        file_results: List[Dict[str, Any]],
        staged_functions: List[Dict[str, Any]],
        verified_funcs: List[Dict[str, Any]],
        quarantined_funcs: List[Dict[str, Any]],
        conflict_funcs: List[Dict[str, Any]]
    ) -> None:
        """Writes comprehensive MySQL relational storage audit log."""
        sep = "=" * 90
        sub_sep = "-" * 90

        lines = [
            sep,
            "                 DEVKNOW KNOWLEDGE BASE - MYSQL DATABASE AUDIT LOG",
            sep,
            f"JOB ID                 : {job_id}",
            f"SOURCE TYPE            : {job_type.upper()}",
            f"SOURCE CODEBASE        : {source_name}",
            f"EXECUTION STATUS       : {job_status.upper()}",
            f"START TIME (UTC)       : {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"COMPLETION TIME (UTC)  : {completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"DURATION               : {duration_sec}s",
            f"DATABASE ENGINE        : MySQL (Database: devknow_db)",
            sep,
            "",
            "1. RELATIONAL DATABASE PERSISTENCE SUMMARY",
            sub_sep,
            f"  * Total Source Files Processed     : {len([f for f in file_results if f.get('status') == 'processed'])}",
            f"  * Total Source Files Skipped       : {len([f for f in file_results if f.get('status') == 'skipped'])}",
            f"  * Total Function Records Persisted : {len(staged_functions)}",
            f"  * Auto-Verified Clean Functions    : {len(verified_funcs)} (Directly active in Knowledge Base)",
            f"  * Quarantined / Pending Review     : {len(quarantined_funcs)} (Held in Staging Queue)",
            f"  * Semantic Conflicts Detected      : {len(conflict_funcs)} (Version increments proposed)",
            ""
        ]

        if error_message:
            lines.extend([
                "FATAL ERROR OCCURRED DURING INGESTION:",
                f"  {error_message}",
                ""
            ])

        # Section: Table codebase_ingestion_jobs Record
        lines.extend([
            "2. TABLE: `codebase_ingestion_jobs` RECORD SNAPSHOT",
            sub_sep,
            f"  job_id                 : {job_id}",
            f"  archive_name           : {source_name}",
            f"  status                 : {job_status}",
            f"  total_files            : {len(file_results)}",
            f"  processed_files        : {len([f for f in file_results if f.get('status') == 'processed'])}",
            f"  skipped_files          : {len([f for f in file_results if f.get('status') == 'skipped'])}",
            f"  total_functions_staged : {len(staged_functions)}",
            f"  total_conflicts        : {len(conflict_funcs)}",
            f"  created_at             : {start_time.isoformat()}Z",
            f"  completed_at           : {completed_at.isoformat()}Z",
            ""
        ])

        # Section: Table functions Detailed Rows
        lines.extend([
            "3. TABLE: `functions` DETAILED PERSISTED RECORDS",
            sub_sep,
            f"Total rows stored: {len(staged_functions)}",
            ""
        ])

        if not staged_functions:
            lines.append("  (No functions extracted or persisted)")
        else:
            for idx, fn in enumerate(staged_functions, 1):
                status_tag = "[AUTO-VERIFIED]" if fn.get("verification_status") == "verified" else "[PENDING REVIEW - QUARANTINED]"
                risk_tag = fn.get("risk_level", "safe").upper()
                lines.extend([
                    f"[{idx:03d}] {status_tag} {fn.get('function_name')} (MySQL ID: #{fn.get('id', 'N/A')})",
                    f"      Rule Version Name   : {fn.get('rule_version_name', 'N/A')}",
                    f"      Language            : {fn.get('language', 'unknown')} | Version: {fn.get('version', '1.0.0')}",
                    f"      Verification Status : {fn.get('verification_status')} | Conflict: {fn.get('conflict_status')}",
                    f"      Parent Function ID  : {fn.get('parent_function_id') or 'None (Original definition)'}",
                    f"      Risk Assessment     : Level={risk_tag}",
                ])

                if fn.get("risk_explanation"):
                    lines.append(f"      Risk Explanation    : {fn.get('risk_explanation')}")

                header = fn.get("function_header") or "N/A"
                lines.append(f"      Signature / Header  : {header.strip()}")

                params = fn.get("parameters") or []
                lines.append(f"      AST Parameters      : {', '.join(params) if params else 'None'}")
                lines.append(f"      Complexity (AST)    : {fn.get('complexity', 1)} | Return Type: {fn.get('return_type') or 'void/inferred'}")
                lines.append(f"      Lines in Source     : {fn.get('start_line', 1)} - {fn.get('end_line', 1)}")

                desc = fn.get("description") or ""
                lines.append(f"      Semantic Context    : {desc.strip()[:200] + ('...' if len(desc) > 200 else '')}")
                lines.append("")

        lines.extend([
            sep,
            "                        END OF MYSQL DATABASE AUDIT LOG",
            sep
        ])

        path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def _write_vectordb_log(
        cls,
        path: Path,
        job_id: str,
        source_name: str,
        job_type: str,
        start_time: datetime.datetime,
        completed_at: datetime.datetime,
        duration_sec: float,
        verified_funcs: List[Dict[str, Any]],
        quarantined_funcs: List[Dict[str, Any]],
        chroma_stats: Dict[str, Any]
    ) -> None:
        """Writes comprehensive ChromaDB vector storage audit log."""
        sep = "=" * 90
        sub_sep = "-" * 90

        collection_name = chroma_stats.get("collection_name", "devknow_functions")
        metric_space = chroma_stats.get("metric_space", "cosine (hnsw:space)")
        pre_count = chroma_stats.get("pre_count", "N/A")
        post_count = chroma_stats.get("post_count", "N/A")
        embedding_model = chroma_stats.get("embedding_model", Config.EMBEDDING_MODEL)

        lines = [
            sep,
            "               DEVKNOW KNOWLEDGE BASE - CHROMADB VECTOR DATABASE AUDIT LOG",
            sep,
            f"JOB ID                 : {job_id}",
            f"SOURCE TYPE            : {job_type.upper()}",
            f"SOURCE CODEBASE        : {source_name}",
            f"START TIME (UTC)       : {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"COMPLETION TIME (UTC)  : {completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"DURATION               : {duration_sec}s",
            f"VECTOR STORE ENGINE    : ChromaDB PersistentClient",
            f"COLLECTION NAME        : {collection_name}",
            f"DISTANCE METRIC        : {metric_space}",
            f"EMBEDDING MODEL        : {embedding_model}",
            f"PERSISTENCE PATH       : {Config.CHROMA_PATH}",
            sep,
            "",
            "1. CHROMADB COLLECTION METRICS SNAPSHOT",
            sub_sep,
            f"  * Pre-Ingestion Vector Count  : {pre_count}",
            f"  * Vectors Upserted This Job   : {len(verified_funcs)} (Auto-verified clean functions)",
            f"  * Vectors Withheld / Quarantined: {len(quarantined_funcs)} (Excluded from vector search until human review)",
            f"  * Post-Ingestion Vector Count : {post_count}",
            ""
        ]

        # Section: Vectors Upserted
        lines.extend([
            "2. VECTORS UPSERTED INTO CHROMADB (ACTIVE KNOWLEDGE)",
            sub_sep,
            f"Total vectors indexed: {len(verified_funcs)}",
            ""
        ])

        if not verified_funcs:
            lines.append("  (No vectors upserted in this job run)")
        else:
            for idx, fn in enumerate(verified_funcs, 1):
                vec_id = fn.get("vector_id") or f"fn_{fn.get('id', 'N/A')}"
                lines.extend([
                    f"[{idx:03d}] VECTOR ID: {vec_id} (Target MySQL ID: #{fn.get('id', 'N/A')})",
                    f"      Function Name       : {fn.get('function_name')}",
                    f"      Rule Version Name   : {fn.get('rule_version_name', 'N/A')}",
                    f"      Language            : {fn.get('language')}",
                    f"      Version             : {fn.get('version', '1.0.0')}",
                    f"      Knowledge Source    : code_file",
                    f"      Vector Status       : ACTIVE IN RETRIEVAL INDEX",
                    f"      Chroma Metadata Payload:",
                    f"        mysql_id             : {fn.get('id')}",
                    f"        function_name        : {fn.get('function_name')}",
                    f"        rule_version_name    : {fn.get('rule_version_name')}",
                    f"        programming_language : {fn.get('language')}",
                    f"        version              : {fn.get('version', '1.0.0')}",
                    f"        knowledge_source     : code_file",
                ])

                # Token-dense search document preview
                header = fn.get("function_header") or ""
                desc = fn.get("description") or ""
                search_doc_preview = (
                    f"Rule/Module: {fn.get('rule_version_name')} | Function: {fn.get('function_name')} | "
                    f"Language: {fn.get('language')} | Header: {header.strip()} | Context: {desc.strip()[:100]}"
                )
                lines.extend([
                    f"      Search Document Text Indexed:",
                    f"        \"{search_doc_preview}...\"",
                    ""
                ])

        # Section: Quarantined functions not in ChromaDB
        lines.extend([
            "3. QUARANTINED FUNCTIONS (EXCLUDED FROM CHROMADB)",
            sub_sep,
            f"Total withheld: {len(quarantined_funcs)}",
            "These functions were quarantined to protect RAG retrieval quality from hallucinations,",
            "syntax corruption, or security risks. They will ONLY be indexed after human approval.",
            ""
        ])

        if not quarantined_funcs:
            lines.append("  (Zero functions quarantined. All code was clean and safely vectorized!)")
        else:
            for idx, fn in enumerate(quarantined_funcs, 1):
                lines.extend([
                    f"[{idx:03d}] FUNCTION: {fn.get('function_name')} (MySQL ID: #{fn.get('id', 'N/A')})",
                    f"      Risk Level          : {fn.get('risk_level', 'warning').upper()}",
                    f"      Risk Explanation    : {fn.get('risk_explanation', 'Pending review')}",
                    f"      Vector Store Status : WITHHELD FROM VECTOR INDEX (0 vectors created)",
                    f"      Resolution Action   : Requires review in Knowledge Base -> Staging Queue UI",
                    ""
                ])

        lines.extend([
            sep,
            "                      END OF CHROMADB VECTOR DATABASE AUDIT LOG",
            sep
        ])

        path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def _write_summary_log(
        cls,
        path: Path,
        job_id: str,
        source_name: str,
        job_type: str,
        start_time: datetime.datetime,
        completed_at: datetime.datetime,
        duration_sec: float,
        job_status: str,
        error_message: Optional[str],
        file_results: List[Dict[str, Any]],
        staged_functions: List[Dict[str, Any]],
        verified_funcs: List[Dict[str, Any]],
        quarantined_funcs: List[Dict[str, Any]],
        conflict_funcs: List[Dict[str, Any]],
        mysql_log_name: str,
        vectordb_log_name: str
    ) -> None:
        """Writes high-level executive summary audit log."""
        sep = "=" * 90
        sub_sep = "-" * 90

        processed_files = [f for f in file_results if f.get("status") == "processed"]
        skipped_files = [f for f in file_results if f.get("status") == "skipped"]
        error_files = [f for f in file_results if f.get("status") == "error"]

        lines = [
            sep,
            "             DEVKNOW KNOWLEDGE BASE - CODEBASE INGESTION EXECUTIVE SUMMARY",
            sep,
            f"JOB IDENTIFIER         : {job_id}",
            f"SOURCE TYPE            : {job_type.upper()}",
            f"SOURCE TARGET          : {source_name}",
            f"OVERALL STATUS         : {job_status.upper()}",
            f"STARTED AT (UTC)       : {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"COMPLETED AT (UTC)     : {completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"TOTAL DURATION         : {duration_sec} seconds",
            sep,
            "",
            "1. CRAWLER & FILE SYSTEM BREAKDOWN",
            sub_sep,
            f"  * Total Files Discovered   : {len(file_results)}",
            f"  * Source Files Ingested    : {len(processed_files)}",
            f"  * Files Skipped (Filtered) : {len(skipped_files)}",
            f"  * Files with Errors        : {len(error_files)}",
            "",
            "2. DUAL-DATABASE STORAGE SUMMARY",
            sub_sep,
            f"  [MySQL Relational Store]",
            f"    - Total Functions Stored     : {len(staged_functions)}",
            f"    - Clean Functions (Verified) : {len(verified_funcs)}",
            f"    - Quarantined for Review     : {len(quarantined_funcs)}",
            f"    - Semantic Conflicts         : {len(conflict_funcs)}",
            "",
            f"  [ChromaDB Vector Store]",
            f"    - Target Collection          : devknow_functions",
            f"    - New Vectors Upserted       : {len(verified_funcs)}",
            f"    - Quarantined (Withheld)     : {len(quarantined_funcs)}",
            "",
            "3. DETAILED LOG MANIFEST FOR DEBUGGING",
            sub_sep,
            f"  * MySQL Relational Log   : data/logs/ingestion/{mysql_log_name}",
            f"  * ChromaDB VectorDB Log  : data/logs/ingestion/{vectordb_log_name}",
            f"  * JSON Audit Export      : data/logs/ingestion/ingestion_{job_id}_audit.json",
            "",
            "4. FILE PROCESSING MANIFEST",
            sub_sep
        ]

        for idx, f in enumerate(file_results, 1):
            status = f.get("status", "unknown").upper()
            funcs = f.get("functions_staged", 0)
            reason = f" - Reason: {f.get('skip_reason')}" if f.get("skip_reason") else ""
            lines.append(f"  [{idx:03d}] [{status:9s}] {f.get('file')} (Functions: {funcs}){reason}")

        lines.extend([
            "",
            sep,
            "                        END OF EXECUTIVE SUMMARY LOG",
            sep
        ])

        path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def _write_json_manifest(
        cls,
        path: Path,
        job_id: str,
        source_name: str,
        job_type: str,
        start_time: datetime.datetime,
        completed_at: datetime.datetime,
        duration_sec: float,
        job_status: str,
        error_message: Optional[str],
        file_results: List[Dict[str, Any]],
        staged_functions: List[Dict[str, Any]],
        verified_funcs: List[Dict[str, Any]],
        quarantined_funcs: List[Dict[str, Any]],
        conflict_funcs: List[Dict[str, Any]],
        chroma_stats: Dict[str, Any]
    ) -> None:
        """Writes machine-readable complete JSON audit manifest."""
        manifest = {
            "job_id": job_id,
            "source_name": source_name,
            "job_type": job_type,
            "job_status": job_status,
            "error_message": error_message,
            "start_time": start_time.isoformat() + "Z",
            "completed_at": completed_at.isoformat() + "Z",
            "duration_seconds": duration_sec,
            "crawler_metrics": {
                "total_files": len(file_results),
                "processed_files": len([f for f in file_results if f.get("status") == "processed"]),
                "skipped_files": len([f for f in file_results if f.get("status") == "skipped"]),
                "error_files": len([f for f in file_results if f.get("status") == "error"]),
            },
            "mysql_metrics": {
                "total_functions": len(staged_functions),
                "verified_count": len(verified_funcs),
                "quarantined_count": len(quarantined_funcs),
                "conflict_count": len(conflict_funcs),
            },
            "chromadb_metrics": {
                "collection_name": chroma_stats.get("collection_name", "devknow_functions"),
                "metric_space": chroma_stats.get("metric_space", "cosine"),
                "pre_count": chroma_stats.get("pre_count"),
                "post_count": chroma_stats.get("post_count"),
                "vectors_upserted": len(verified_funcs),
                "vectors_withheld": len(quarantined_funcs),
            },
            "file_results": file_results,
            "functions": staged_functions
        }
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
