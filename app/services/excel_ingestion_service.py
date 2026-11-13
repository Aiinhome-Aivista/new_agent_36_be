import os
import re
import logging
import datetime
import pandas as pd
from typing import Dict, Any, List

from app.config import Config
from app.utils.excel_validator import ExcelValidator
from app.utils.language_detector import detect_language_from_code
from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.models.database_models import KnowledgeBaseFile, FunctionRecord, IngestionLog

logger = logging.getLogger("DevKnow.Ingestion")

class ExcelIngestionService:
    @staticmethod
    def extract_function_name(header_text: str) -> str:
        """Extracts candidate function name from a header/signature string"""
        if not header_text:
            return "anonymous_function"
        header = header_text.strip()
        # Look for pattern like: return_type FunctionName(args) or def func_name(
        # e.g., bool IsRealBreakerPart(AtomicPart part)
        match = re.search(r"(\w+)\s*\(", header)
        if match:
            return match.group(1)
        # Or words separated by spaces, take last token before paren or first token
        first_line = header.split("\n")[0].split("(")[0].strip()
        tokens = first_line.split()
        if tokens:
            return tokens[-1]
        return header[:50]

    @classmethod
    def ingest_excel_file(cls, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Executes complete ingestion pipeline:
        Excel Read -> Validation -> MySQL Persistence -> Vector Indexing -> Audit Logging
        """
        is_valid, err_msg, mapping, detected_langs = ExcelValidator.validate_excel_structure(file_path)
        if not is_valid:
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
            # Create file tracking record
            kb_file = KnowledgeBaseFile(
                file_name=original_filename,
                file_path=file_path,
                uploaded_at=datetime.datetime.utcnow(),
                ingestion_status="processing"
            )
            session.add(kb_file)
            session.flush()
            file_id = kb_file.id

            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                kb_file.ingestion_status = "failed"
                session.commit()
                return {"success": False, "error": f"Failed reading Excel rows: {str(e)}"}

            total_records = len(df)
            successful_records = 0
            failed_records = 0
            skipped_records = 0
            all_languages = set()
            new_records_for_indexing = []

            for idx, row in df.iterrows():
                row_num = idx + 2  # Excel 1-based indexing including header
                try:
                    # Extract fields via mapped column names
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

                    if not header_val or not body_val or header_val.lower() == "nan" or body_val.lower() == "nan":
                        skipped_records += 1
                        log = IngestionLog(
                            file_id=file_id,
                            record_identifier=f"Row {row_num}",
                            status="skipped",
                            error_message="Missing required script header or body"
                        )
                        session.add(log)
                        continue

                    rule_name = str(row[rule_name_col]).strip() if rule_name_col and pd.notna(row.get(rule_name_col)) else ""
                    rule_desc = str(row[rule_desc_col]).strip() if rule_desc_col and pd.notna(row.get(rule_desc_col)) else ""
                    func_name = cls.extract_function_name(header_val)

                    # Programming language resolution
                    lang_val = ""
                    if lang_col and pd.notna(row.get(lang_col)):
                        lang_val = str(row[lang_col]).strip().lower()
                    if not lang_val or lang_val == "nan":
                        # Attempt detection from code snippet
                        detected = detect_language_from_code(body_val + " " + header_val)
                        lang_val = detected if detected else "c#"  # Default for C# rulebook scripts
                    else:
                        lang_val = lang_val.lower()

                    all_languages.add(lang_val)

                    framework_val = str(row[framework_col]).strip() if framework_col and pd.notna(row.get(framework_col)) else ""
                    ver_val = str(row[ver_col]).strip() if ver_col and pd.notna(row.get(ver_col)) else "1.0.0"
                    class_val = str(row[class_col]).strip() if class_col and pd.notna(row.get(class_col)) else ""
                    ns_val = str(row[ns_col]).strip() if ns_col and pd.notna(row.get(ns_col)) else ""
                    desc_val = str(row[desc_col]).strip() if desc_col and pd.notna(row.get(desc_col)) else ""
                    meta_val = str(row[meta_col]).strip() if meta_col and pd.notna(row.get(meta_col)) else ""

                    # Duplicate check: same function_name + programming_language + version + rule_name
                    existing = session.query(FunctionRecord).filter(
                        FunctionRecord.function_name == func_name,
                        FunctionRecord.programming_language == lang_val,
                        FunctionRecord.version == ver_val,
                        FunctionRecord.rule_version_name == rule_name
                    ).first()

                    if existing:
                        # Update existing record
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

                    # Prepare for ChromaDB indexing
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
                    log = IngestionLog(
                        file_id=file_id,
                        record_identifier=f"{func_name} ({lang_val} v{ver_val})",
                        status="success",
                        error_message=f"Record {action_status} successfully"
                    )
                    session.add(log)

                except Exception as ex:
                    failed_records += 1
                    logger.error(f"Error processing row {row_num}: {ex}")
                    log = IngestionLog(
                        file_id=file_id,
                        record_identifier=f"Row {row_num}",
                        status="failed",
                        error_message=str(ex)
                    )
                    session.add(log)

            # Update file summary
            kb_file.total_records = total_records
            kb_file.successful_records = successful_records
            kb_file.failed_records = failed_records
            kb_file.skipped_records = skipped_records
            kb_file.languages_detected = ",".join(sorted(list(all_languages)))
            kb_file.ingestion_status = "completed" if failed_records == 0 else "partial"

        # Index vectors in ChromaDB in batch
        if new_records_for_indexing:
            try:
                chroma_service.upsert_batch(new_records_for_indexing)
                logger.info(f"Indexed {len(new_records_for_indexing)} records in ChromaDB vector store.")
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

