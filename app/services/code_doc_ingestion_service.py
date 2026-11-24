"""
CodeDocIngestionService: Ingestion engine for source code files (.py, .cs, .ts, .js)
and unstructured developer documentation (.md, .txt).

Key Principles:
1. AST + Tree-sitter Extracts Deterministic Facts (signatures, params, complexity, syntax).
2. LLM Extracts Semantic Context (purpose, organizational rules, edge cases).
3. Zero-Direct-Ingestion: All extracted items enter a Staging Queue (verification_status='pending_review').
4. Conflict & Versioning: Detects duplicate/conflicting function names and suggests next version (e.g. v1.1.0).
5. Only Human-Approved functions are pushed to the ChromaDB vector index and become active in chat retrieval.
"""
import os
import re
import datetime
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.code_parser_service import code_parser
from app.services.gemini_service import gemini_service
from app.models.database_models import FunctionRecord, KnowledgeBaseFile

logger = logging.getLogger("DevKnow.CodeDocIngestion")


class CodeDocIngestionService:
    SUPPORTED_CODE_EXTENSIONS = {
        ".py": "python",
        ".cs": "c_sharp",
        ".ts": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "c++"
    }

    SUPPORTED_DOC_EXTENSIONS = {
        ".md": "markdown",
        ".txt": "text"
    }

    @classmethod
    def is_supported_file(cls, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_CODE_EXTENSIONS or ext in cls.SUPPORTED_DOC_EXTENSIONS

    @classmethod
    def get_file_type(cls, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in cls.SUPPORTED_CODE_EXTENSIONS:
            return "code"
        if ext in cls.SUPPORTED_DOC_EXTENSIONS:
            return "doc"
        return "unknown"

    @classmethod
    def ingest_code_or_doc_file(
        cls,
        file_path: str,
        original_filename: str,
        author_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parses a code file or documentation file, runs AST analysis on all code units,
        uses LLM for contextual understanding, and stages results for human verification.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path_obj.read_text(encoding="latin-1")
            except Exception as e:
                return {"success": False, "error": f"Cannot read file encoding: {e}"}

        ext = file_path_obj.suffix.lower()
        file_type = cls.get_file_type(original_filename)

        if file_type == "unknown":
            return {
                "success": False,
                "error": f"Unsupported file type '{ext}'. Supported: .py, .cs, .ts, .js, .md, .txt"
            }

        # Record file in knowledge_base_files tracking table
        with db_service.get_session() as session:
            kb_file = KnowledgeBaseFile(
                file_name=original_filename,
                file_path=str(file_path),
                uploaded_at=datetime.datetime.utcnow(),
                ingestion_status="pending_review",
                languages_detected=cls.SUPPORTED_CODE_EXTENSIONS.get(ext, ext.lstrip("."))
            )
            session.add(kb_file)
            session.flush()
            file_id = kb_file.id

        if file_type == "code":
            lang = cls.SUPPORTED_CODE_EXTENSIONS.get(ext, "general")
            return cls._process_code_file(content, original_filename, lang, file_id, author_notes)
        else:
            return cls._process_doc_file(content, original_filename, file_id, author_notes)

    @classmethod
    def _process_code_file(
        cls,
        code_content: str,
        filename: str,
        language: str,
        file_id: int,
        author_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extracts all functions via AST, checks conflicts, and saves to Staging Queue"""
        extracted = code_parser.extract_all_functions(code_content, language, source_filename=filename)

        if not extracted:
            # Fall back to single function parsing if whole file extraction didn't yield blocks
            meta = code_parser.parse_function_metadata(code_content, language)
            if meta.get("function_name") and meta["function_name"] != "unnamed_function":
                extracted = [{
                    "function_name": meta["function_name"],
                    "function_header": code_content.strip().split("\n")[0][:150],
                    "function_body": code_content.strip(),
                    "full_code": code_content.strip(),
                    "parameters": meta.get("parameters", []),
                    "return_type": meta.get("return_type", ""),
                    "start_line": 1,
                    "end_line": len(code_content.splitlines()),
                    "complexity": meta.get("complexity", 1),
                    "language": language
                }]

        if not extracted:
            return {
                "success": False,
                "error": f"No functions or methods could be identified in '{filename}' via AST / Tree-sitter.",
                "file_id": file_id
            }

        staged_items = []
        conflicts_count = 0

        with db_service.get_session() as session:
            for item in extracted:
                fn_name = item["function_name"]
                fn_header = item["function_header"]
                fn_body = item["function_body"]
                complexity = item.get("complexity", 1)
                params = item.get("parameters", [])
                ret_type = item.get("return_type", "")

                # 1. Conflict & Versioning Check: Does function already exist in KB?
                existing = (
                    session.query(FunctionRecord)
                    .filter_by(function_name=fn_name)
                    .order_by(FunctionRecord.created_at.desc())
                    .first()
                )

                if existing:
                    conflicts_count += 1
                    conflict_status = "conflict_detected"
                    parent_id = existing.id
                    # Auto-increment patch version (e.g., 1.0.0 -> 1.1.0)
                    next_version = cls._increment_version(existing.version or "1.0.0")
                    rule_desc = f"Proposed update to existing function '{fn_name}' (Current: {existing.version}, Proposed: {next_version})"
                else:
                    conflict_status = "new_function"
                    parent_id = None
                    next_version = "1.0.0"
                    rule_desc = f"New function extracted from source file '{filename}'"

                # 2. LLM Semantic Context Generation (with graceful fallback)
                description = cls._generate_semantic_context(
                    fn_name=fn_name,
                    code_snippet=item.get("full_code", fn_body),
                    language=language,
                    complexity=complexity,
                    params=params,
                    author_notes=author_notes
                )

                # 3. Store into Staging Queue (verification_status='pending_review')
                staged_record = FunctionRecord(
                    rule_version_name=f"SourceFile_{Path(filename).stem}",
                    rule_version_description=rule_desc,
                    function_name=fn_name,
                    function_header=fn_header,
                    function_body=fn_body,
                    programming_language=language,
                    framework=Path(filename).suffix.lstrip("."),
                    version=next_version,
                    description=description,
                    additional_metadata=f"AST Complexity: {complexity} | Params: {', '.join(params)} | Lines: {item.get('start_line', 1)}-{item.get('end_line', 1)}",
                    knowledge_source="code_file",
                    verification_status="pending_review",  # CRITICAL: Requires human approval
                    conflict_status=conflict_status,
                    parent_function_id=parent_id,
                    source_file_id=file_id,
                    created_at=datetime.datetime.utcnow(),
                    updated_at=datetime.datetime.utcnow()
                )
                session.add(staged_record)
                session.flush()

                staged_items.append({
                    "id": staged_record.id,
                    "function_name": fn_name,
                    "language": language,
                    "version": next_version,
                    "conflict_status": conflict_status,
                    "parent_function_id": parent_id,
                    "complexity": complexity,
                    "parameters": params,
                    "description": description
                })

            # Update file summary
            kb_f = session.query(KnowledgeBaseFile).filter_by(id=file_id).first()
            if kb_f:
                kb_f.total_records = len(staged_items)
                kb_f.successful_records = len(staged_items)
                kb_f.ingestion_status = "staged_for_review"

        logger.info(
            f"Staged {len(staged_items)} function(s) from '{filename}' for human verification. "
            f"Conflicts detected: {conflicts_count}."
        )

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "file_type": "code",
            "total_extracted": len(staged_items),
            "conflicts_count": conflicts_count,
            "staged_functions": staged_items,
            "message": f"Successfully extracted {len(staged_items)} function(s) via AST. Staged for human verification before indexing into Knowledge Base."
        }

    @classmethod
    def _process_doc_file(
        cls,
        doc_content: str,
        filename: str,
        file_id: int,
        author_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parses unstructured documentation (.md, .txt), finds code blocks,
        runs AST on each snippet, attaches surrounding prose as context,
        and stages for human verification.
        """
        # Match fenced code blocks: ```lang\ncode\n```
        pattern = r"```([a-zA-Z0-9_#+]*)\s*\n(.*?)```"
        matches = list(re.finditer(pattern, doc_content, re.DOTALL))

        staged_items = []
        conflicts_count = 0

        with db_service.get_session() as session:
            for idx, m in enumerate(matches):
                raw_lang = m.group(1).strip().lower() or "general"
                code_snippet = m.group(2).strip()
                if not code_snippet:
                    continue

                lang = code_parser.normalize_language(raw_lang)

                # AST inspection on this snippet
                ast_meta = code_parser.parse_function_metadata(code_snippet, lang)
                fn_name = ast_meta.get("function_name")
                if not fn_name or fn_name == "unnamed_function":
                    # Derive a clean name from document context
                    fn_name = f"doc_rule_{Path(filename).stem}_{idx+1}"

                # Extract preceding text (up to 400 chars) as senior developer doc context
                start_pos = m.start()
                preceding_text = doc_content[max(0, start_pos - 400):start_pos].strip()
                doc_context_summary = " ".join([line.strip() for line in preceding_text.splitlines() if line.strip() and not line.startswith("#")][-3:])

                # Conflict check
                existing = (
                    session.query(FunctionRecord)
                    .filter_by(function_name=fn_name)
                    .order_by(FunctionRecord.created_at.desc())
                    .first()
                )
                if existing:
                    conflicts_count += 1
                    conflict_status = "conflict_detected"
                    parent_id = existing.id
                    next_version = cls._increment_version(existing.version or "1.0.0")
                else:
                    conflict_status = "new_function"
                    parent_id = None
                    next_version = "1.0.0"

                description = doc_context_summary or f"Architectural rule and code snippet from documentation '{filename}'."

                first_line = code_snippet.splitlines()[0][:150]

                staged_record = FunctionRecord(
                    rule_version_name=f"Doc_{Path(filename).stem}",
                    rule_version_description=f"Extracted from senior developer documentation '{filename}'",
                    function_name=fn_name,
                    function_header=first_line,
                    function_body=code_snippet,
                    programming_language=lang,
                    framework="documentation",
                    version=next_version,
                    description=description,
                    additional_metadata=f"Extracted via AST + Document Parser | Snippet #{idx+1} | Source: {filename}",
                    knowledge_source="doc_file",
                    verification_status="pending_review",  # Human review required
                    conflict_status=conflict_status,
                    parent_function_id=parent_id,
                    source_file_id=file_id,
                    created_at=datetime.datetime.utcnow(),
                    updated_at=datetime.datetime.utcnow()
                )
                session.add(staged_record)
                session.flush()

                staged_items.append({
                    "id": staged_record.id,
                    "function_name": fn_name,
                    "language": lang,
                    "version": next_version,
                    "conflict_status": conflict_status,
                    "description": description
                })

            kb_f = session.query(KnowledgeBaseFile).filter_by(id=file_id).first()
            if kb_f:
                kb_f.total_records = len(staged_items)
                kb_f.successful_records = len(staged_items)
                kb_f.ingestion_status = "staged_for_review"

        logger.info(f"Staged {len(staged_items)} code snippet(s) from doc '{filename}' for human verification.")

        # Generate dedicated AST JSON and Log for doc file
        try:
            import json
            base_dir = Path(__file__).resolve().parent.parent.parent
            ast_dir = base_dir / "data" / "logs" / "ast"
            ast_dir.mkdir(parents=True, exist_ok=True)

            clean_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(filename).stem)
            clean_ext = Path(filename).suffix.lstrip(".")
            file_tag = f"{clean_stem}_{clean_ext}" if clean_ext else clean_stem
            ts_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            doc_report = {
                "source_file": filename,
                "file_type": "documentation",
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "total_extracted": len(staged_items),
                "staged_items": staged_items
            }

            per_doc_json = ast_dir / f"ast_{file_tag}_{ts_str}.json"
            per_doc_json.write_text(json.dumps(doc_report, indent=2), encoding="utf-8")

            per_doc_log = ast_dir / f"ast_{file_tag}_{ts_str}.log"
            sep = "=" * 80
            log_body = (
                f"{sep}\n"
                f"DOCUMENTATION AST EXTRACTION REPORT: {filename}\n"
                f"TIMESTAMP                          : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"TOTAL CODE SNIPPETS / RULES FOUND  : {len(staged_items)}\n"
                f"{sep}\n\n"
            )
            for idx, item in enumerate(staged_items):
                log_body += (
                    f"[{idx+1}] Function/Rule: {item['function_name']}\n"
                    f"    Language     : {item['language']}\n"
                    f"    Version      : {item['version']}\n"
                    f"    Status       : {item['conflict_status']}\n"
                    f"    Description  : {item['description']}\n\n"
                )
            per_doc_log.write_text(log_body, encoding="utf-8")

            print(f"\n{sep}", flush=True)
            print(f"📖 [AST Engine] Documentation Ingestion Complete:", flush=True)
            print(f"• Source File          : {filename}", flush=True)
            print(f"• Total Snippets/Rules : {len(staged_items)}", flush=True)
            print(f"📁 Dedicated AST Log   -> {per_doc_log.name}", flush=True)
            print(f"📁 Dedicated AST JSON  -> {per_doc_json.name}", flush=True)
            print(f"{sep}\n", flush=True)
        except Exception as e:
            logger.debug(f"Doc AST logging notice: {e}")

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "file_type": "doc",
            "total_extracted": len(staged_items),
            "conflicts_count": conflicts_count,
            "staged_functions": staged_items,
            "message": f"Extracted {len(staged_items)} architectural rule/code snippet(s) from document. Staged for human verification."
        }

    @classmethod
    def list_staging_queue(cls, status: str = "pending_review") -> List[Dict[str, Any]]:
        """Returns all functions currently waiting in the Staging Queue for human review"""
        with db_service.get_session() as session:
            records = (
                session.query(FunctionRecord)
                .filter(FunctionRecord.verification_status == status)
                .order_by(FunctionRecord.created_at.desc())
                .all()
            )

            result = []
            for r in records:
                item = r.to_dict()
                # Attach parent/conflict comparison if conflict exists
                if r.parent_function_id:
                    parent = session.query(FunctionRecord).filter_by(id=r.parent_function_id).first()
                    if parent:
                        item["conflicting_parent"] = {
                            "id": parent.id,
                            "function_name": parent.function_name,
                            "version": parent.version,
                            "header": parent.function_header,
                            "body": parent.function_body,
                            "created_at": parent.created_at.isoformat() if parent.created_at else None
                        }
                result.append(item)
            return result

    @classmethod
    def approve_staged_function(
        cls,
        function_id: int,
        approved_version: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Human Verification Action: Approves a staged function.
        1. Updates verification_status to 'verified'.
        2. Upserts into ChromaDB so it becomes immediately searchable in chat.
        3. If it updates an existing parent function, preserves audit trail.
        """
        with db_service.get_session() as session:
            func = session.query(FunctionRecord).filter_by(id=function_id).first()
            if not func:
                return {"success": False, "error": f"Function {function_id} not found."}

            func.verification_status = "verified"
            if approved_version:
                func.version = approved_version
            if notes:
                func.additional_metadata = f"{func.additional_metadata or ''} | Verified by Admin: {notes}".strip(" |")
            func.updated_at = datetime.datetime.utcnow()

            # Prepare dict for ChromaDB upsert
            f_dict = func.to_dict()
            session.commit()

        # Upsert into ChromaDB vector store
        try:
            chroma_service.upsert_function(
                mysql_id=f_dict["id"],
                function_name=f_dict["function_name"],
                rule_version_name=f_dict["rule_version_name"] or "Verified_Code",
                rule_version_description=f_dict["description"] or "",
                function_header=f_dict["function_header"] or "",
                programming_language=f_dict["programming_language"] or "general",
                framework=f_dict["framework"] or "",
                version=f_dict["version"] or "1.0.0",
                description=f_dict["description"] or "",
                knowledge_source=f_dict["knowledge_source"] or "verified_code"
            )
            logger.info(f"Function {function_id} ('{f_dict['function_name']}') approved & indexed in ChromaDB.")
            return {
                "success": True,
                "message": f"Function '{f_dict['function_name']}' approved, verified, and indexed into Knowledge Base.",
                "function": f_dict
            }
        except Exception as e:
            logger.error(f"Error vectorizing approved function {function_id}: {e}")
            return {
                "success": True,
                "warning": f"Function approved in database, but vector indexing encountered notice: {e}",
                "function": f_dict
            }

    @classmethod
    def reject_staged_function(cls, function_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """Human Verification Action: Rejects a staged function (never enters ChromaDB)"""
        with db_service.get_session() as session:
            func = session.query(FunctionRecord).filter_by(id=function_id).first()
            if not func:
                return {"success": False, "error": f"Function {function_id} not found."}

            func.verification_status = "rejected"
            if reason:
                func.additional_metadata = f"{func.additional_metadata or ''} | Rejection Reason: {reason}".strip(" |")
            func.updated_at = datetime.datetime.utcnow()
            session.commit()

        logger.info(f"Function {function_id} rejected by human reviewer. Excluded from Knowledge Base.")
        return {
            "success": True,
            "message": f"Function '{func.function_name}' has been rejected and will not be indexed."
        }

    @classmethod
    def _generate_semantic_context(
        cls,
        fn_name: str,
        code_snippet: str,
        language: str,
        complexity: int,
        params: List[str],
        author_notes: Optional[str] = None
    ) -> str:
        """Uses Gemini to generate concise business context for an extracted code unit"""
        prompt = (
            f"You are DevKnow Code Intelligence. In 1 or 2 concise sentences, summarize the purpose "
            f"and business logic of this {language} function for an enterprise developer knowledge base:\n\n"
            f"Function Name: {fn_name}\n"
            f"Parameters: {', '.join(params) if params else 'None'}\n"
            f"Complexity: {complexity}\n"
            f"Author Context: {author_notes or 'Organizational codebase'}\n"
            f"Code:\n```{language}\n{code_snippet[:600]}\n```\n\n"
            f"Return ONLY the 1-2 sentence description. No markdown headers."
        )
        try:
            desc = gemini_service.generate_text(prompt)
            if desc and len(desc.strip()) > 10:
                return desc.strip()
        except Exception as e:
            logger.debug(f"LLM semantic generation fallback: {e}")

        # Deterministic fallback
        return f"{language.capitalize()} function '{fn_name}' with {len(params)} parameter(s) and cyclomatic complexity {complexity}."

    @staticmethod
    def _increment_version(ver_str: str) -> str:
        """Increments version string (e.g. 1.0.0 -> 1.1.0 or 1.0 -> 1.1)"""
        parts = ver_str.replace("v", "").split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            parts[1] = str(int(parts[1]) + 1)
            return ".".join(parts)
        elif len(parts) >= 1 and parts[0].isdigit():
            return f"{parts[0]}.1.0"
        return "1.1.0"
