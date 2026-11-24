"""
Excel Ingestion Service Module for DevKnow.

This module provides enterprise-grade parsing, validation, and ingestion of Excel (.xlsx/.xls)
knowledge base files containing code rules, script headers, and function definitions.

Workflow:
1. Validates spreadsheet column structure using fuzzy header resolution (ExcelValidator).
2. Initiates tracking records in the database (KnowledgeBaseFile).
3. Processes rows iteratively:
   - Validates script headers and function bodies.
   - Infers or validates programming languages.
   - Extracts accurate function signatures via unified Tree-sitter AST with regex fallback.
   - Performs deduplication checks (upsert semantics).
   - Generates audit log entries (IngestionLog).
4. Commits relational records and triggers batch embedding vectorization in ChromaDB.
"""

import os
import re
import logging
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from app.config import Config
from app.utils.excel_validator import ExcelValidator
from app.utils.language_detector import detect_language_from_code
from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.code_parser_service import code_parser
from app.models.database_models import KnowledgeBaseFile, FunctionRecord, IngestionLog

logger = logging.getLogger("DevKnow.Ingestion")


class ExcelIngestionService:
    """
    Service responsible for orchestrating the end-to-end ingestion lifecycle of
    code rulebooks and function repositories uploaded as Excel spreadsheets.
    """

    @staticmethod
    def extract_function_name(header_text: str, language: Optional[str] = None) -> str:
        """
        Extracts a clean, canonical function name from raw script header text.

        Prioritizes deterministic Tree-sitter AST parsing for accurate syntax identification.
        Falls back to resilient regex and whitespace tokenization if parsing fails.

        Args:
            header_text: Raw function declaration string (e.g., 'public void ProcessPayment(int id)').
            language: Target programming language identifier (e.g., 'c#', 'python', 'typescript').

        Returns:
            Extracted function name string, or 'anonymous_function' if completely unresolvable.
        """
        if not header_text or not header_text.strip():
            return "anonymous_function"

        # 1. Primary path: Use unified Tree-sitter parser for structural accuracy
        try:
            meta = code_parser.parse_function_metadata(header_text, language)
            if meta and meta.get("function_name") and meta["function_name"] != "unnamed_function":
                return meta["function_name"]
        except Exception as ex:
            logger.debug(f"Tree-sitter extraction fallback triggered for header '{header_text[:30]}...': {ex}")

        # 2. Secondary fallback: Regex match looking for identifier before opening parenthesis
        header = header_text.strip()
        match = re.search(r"(\w+)\s*\(", header)
        if match:
            return match.group(1)

        # 3. Tertiary fallback: Extract trailing token before parenthesis on the first non-empty line
        first_line = header.split("\n")[0].split("(")[0].strip()
        tokens = first_line.split()
        if tokens:
            return tokens[-1]

        # 4. Final safety truncate
        return header[:50]

    @classmethod
    def ingest_excel_file(cls, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Executes the complete ingestion pipeline for a validated Excel workbook.

        Lifecycle:
        1. Excel Structure Validation: Validates columns using flexible aliases.
        2. Database Tracking: Creates a KnowledgeBaseFile entry with 'processing' status.
        3. Row Processing & Deduplication: Reads each row, extracts metadata, checks
           for identical existing records (by name, language, version, rule name),
           and performs either update or insert (Upsert).
        4. Vector Indexing: Batch upserts all valid records into ChromaDB vector store.
        5. Audit Logging: Records per-row success/skip/failure entries in IngestionLog.

        Args:
            file_path: Absolute filesystem path to the saved Excel spreadsheet.
            original_filename: User-facing name of the uploaded file.

        Returns:
            Dict containing ingestion statistics, detected languages, and execution status.
        """
        # Step 1: Validate column schema and extract field mappings
        is_valid, err_msg, mapping, detected_langs = ExcelValidator.validate_excel_structure(file_path)
        if not is_valid:
            logger.warning(f"Excel validation rejected '{original_filename}': {err_msg}")
            return {
                "success": False,
                "error": err_msg,
                "total_records": 0,
                "successful_records": 0,
                "failed_records": 0,
                "skipped_records": 0,
                "languages_detected": []
            }

        with db_service.get_session() as session:
            # Step 2: Initialize file tracking in the knowledge_base_files table
            kb_file = KnowledgeBaseFile(
                file_name=original_filename,
                file_path=file_path,
                uploaded_at=datetime.datetime.utcnow(),
                ingestion_status="processing"
            )
            session.add(kb_file)
            session.flush()  # Generate primary key ID for foreign key references
            file_id = kb_file.id

            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                logger.error(f"Failed reading Excel rows from '{file_path}': {e}")
                kb_file.ingestion_status = "failed"
                session.commit()
                return {"success": False, "error": f"Failed reading Excel rows: {str(e)}"}

            total_records = len(df)
            successful_records = 0
            failed_records = 0
            skipped_records = 0
            all_languages = set()
            new_records_for_indexing: List[Dict[str, Any]] = []

            # Step 3: Iterate through spreadsheet rows
            for idx, row in df.iterrows():
                row_num = idx + 2  # Excel row numbers start at 2 (accounting for header row)
                try:
                    # Extract field values via resolved column mapping
                    header_col = mapping.get("script_header")
                    body_col = mapping.get("script_body")
                    rule_name_col = mapping.get("rule_version_name")
                    rule_desc_col = mapping.get("rule_version_description")
                    lang_col = mapping.get("language")
                    framework_col = mapping.get("framework")
                    ver_col = mapping.get("version")
                    class_col = mapping.get("class_name")
                    ns_col = mapping.get("namespace")
                    desc_col = mapping.get("description")
                    meta_col = mapping.get("metadata")

                    header_val = str(row[header_col]).strip() if header_col and pd.notna(row.get(header_col)) else ""
                    body_val = str(row[body_col]).strip() if body_col and pd.notna(row.get(body_col)) else ""

                    # Skip empty rows that lack vital code content
                    if not header_val or not body_val or header_val.lower() == "nan" or body_val.lower() == "nan":
                        skipped_records += 1
                        session.add(IngestionLog(
                            file_id=file_id,
                            record_identifier=f"Row {row_num}",
                            status="skipped",
                            error_message="Missing required script header or body"
                        ))
                        continue

                    # Language resolution: use explicit column if present, else infer from code
                    lang_val = ""
                    if lang_col and pd.notna(row.get(lang_col)):
                        lang_val = str(row[lang_col]).strip().lower()
                    if not lang_val or lang_val == "nan":
                        detected = detect_language_from_code(body_val + " " + header_val)
                        lang_val = detected if detected else "c#"  # Default assumption for enterprise rulebooks
                    else:
                        lang_val = lang_val.lower()

                    all_languages.add(lang_val)

                    # Extract metadata values with safe string conversion
                    rule_name = str(row[rule_name_col]).strip() if rule_name_col and pd.notna(row.get(rule_name_col)) else ""
                    rule_desc = str(row[rule_desc_col]).strip() if rule_desc_col and pd.notna(row.get(rule_desc_col)) else ""
                    func_name = cls.extract_function_name(header_val, lang_val)

                    framework_val = str(row[framework_col]).strip() if framework_col and pd.notna(row.get(framework_col)) else ""
                    ver_val = str(row[ver_col]).strip() if ver_col and pd.notna(row.get(ver_col)) else "1.0.0"
                    class_val = str(row[class_col]).strip() if class_col and pd.notna(row.get(class_col)) else ""
                    ns_val = str(row[ns_col]).strip() if ns_col and pd.notna(row.get(ns_col)) else ""
                    desc_val = str(row[desc_col]).strip() if desc_col and pd.notna(row.get(desc_col)) else ""
                    meta_val = str(row[meta_col]).strip() if meta_col and pd.notna(row.get(meta_col)) else ""

                    # Deduplication check: Match identical function name, language, version, and rule name
                    existing = session.query(FunctionRecord).filter(
                        FunctionRecord.function_name == func_name,
                        FunctionRecord.programming_language == lang_val,
                        FunctionRecord.version == ver_val,
                        FunctionRecord.rule_version_name == rule_name
                    ).first()

                    if existing:
                        # Update existing entity
                        existing.function_header = header_val
                        existing.function_body = body_val
                        existing.rule_version_description = rule_desc
                        existing.framework = framework_val
                        existing.class_name = class_val
                        existing.namespace = ns_val
                        existing.description = desc_val
                        existing.additional_metadata = meta_val
                        existing.source_file_id = file_id
                        existing.updated_at = datetime.datetime.utcnow()
                        record_id = existing.id
                        action_status = "updated"
                    else:
                        # Insert new entity
                        new_func = FunctionRecord(
                            rule_version_name=rule_name,
                            rule_version_description=rule_desc,
                            function_name=func_name,
                            function_header=header_val,
                            function_body=body_val,
                            programming_language=lang_val,
                            framework=framework_val,
                            version=ver_val,
                            class_name=class_val,
                            namespace=ns_val,
                            description=desc_val,
                            additional_metadata=meta_val,
                            knowledge_source="excel",
                            source_file_id=file_id,
                            created_at=datetime.datetime.utcnow(),
                            updated_at=datetime.datetime.utcnow()
                        )
                        session.add(new_func)
                        session.flush()
                        record_id = new_func.id
                        action_status = "inserted"

                    # Collect record payload for downstream vector embedding indexing
                    new_records_for_indexing.append({
                        "mysql_id": record_id,
                        "function_name": func_name,
                        "rule_version_name": rule_name,
                        "rule_version_description": rule_desc,
                        "function_header": header_val,
                        "programming_language": lang_val,
                        "framework": framework_val,
                        "version": ver_val,
                        "class_name": class_val,
                        "namespace": ns_val,
                        "description": desc_val,
                        "additional_metadata": meta_val,
                        "knowledge_source": "excel",
                        "source_file_id": file_id
                    })

                    successful_records += 1
                    session.add(IngestionLog(
                        file_id=file_id,
                        record_identifier=f"{func_name} ({lang_val} v{ver_val})",
                        status="success",
                        error_message=f"Record {action_status} successfully"
                    ))

                except Exception as ex:
                    failed_records += 1
                    logger.error(f"Error processing row {row_num}: {ex}")
                    session.add(IngestionLog(
                        file_id=file_id,
                        record_identifier=f"Row {row_num}",
                        status="failed",
                        error_message=str(ex)
                    ))

            # Step 4: Finalize file tracking statistics
            kb_file.total_records = total_records
            kb_file.successful_records = successful_records
            kb_file.failed_records = failed_records
            kb_file.skipped_records = skipped_records
            kb_file.languages_detected = ",".join(sorted(list(all_languages)))
            kb_file.ingestion_status = "completed" if failed_records == 0 else "partial"

        # Step 5: Synchronize vectors in ChromaDB vector store
        if new_records_for_indexing:
            try:
                chroma_service.upsert_batch(new_records_for_indexing)
                logger.info(f"Indexed {len(new_records_for_indexing)} records into ChromaDB vector store.")
            except Exception as e:
                logger.error(f"ChromaDB batch upsert failed: {e}")

        return {
            "success": True,
            "file_id": file_id,
            "file_name": original_filename,
            "uploaded_at": datetime.datetime.utcnow().isoformat(),
            "total_records": total_records,
            "successful_records": successful_records,
            "failed_records": failed_records,
            "skipped_records": skipped_records,
            "languages_detected": sorted(list(all_languages)),
            "ingestion_status": "completed" if failed_records == 0 else "partial"
        }
