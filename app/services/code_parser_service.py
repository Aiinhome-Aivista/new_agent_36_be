"""
CodeParserService: AST and Tree-sitter Code Intelligence Engine for DevKnow.
Provides deterministic syntax validation, function signature extraction,
cyclomatic complexity calculation, and code quality smell detection across
C#, TypeScript, JavaScript, and Python.
"""
import ast
import re
import json
import logging
import warnings
import textwrap
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional

# Suppress deprecation warnings from C-extensions / tree_sitter legacy bindings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

logger = logging.getLogger("DevKnow.CodeParser")

# ── Dedicated AST Detail Logger (writes directly to data/logs/) ───────────────
_ast_detail_logger: Optional[logging.Logger] = None

def _get_ast_logger() -> logging.Logger:
    """
    Returns the dedicated AST detail file logger, creating it on first call.
    Writes to:
      1. new_agent_36_be/data/logs/ast_extraction.log (main logs directory)
      2. new_agent_36_be/data/logs/ast/ast_analysis.log (backward compatibility)
    Uses RotatingFileHandler: 10MB per file, keeps last 5 files.
    """
    global _ast_detail_logger
    if _ast_detail_logger is not None:
        return _ast_detail_logger

    base_dir = Path(__file__).resolve().parent.parent.parent  # → new_agent_36_be/
    main_logs_dir = base_dir / "data" / "logs"
    main_logs_dir.mkdir(parents=True, exist_ok=True)
    ast_subdir = main_logs_dir / "ast"
    ast_subdir.mkdir(parents=True, exist_ok=True)

    _ast_detail_logger = logging.getLogger("DevKnow.AST.Detail")
    _ast_detail_logger.setLevel(logging.DEBUG)
    _ast_detail_logger.propagate = False   # Don't bubble up to root logger

    if not _ast_detail_logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Primary log file in data/logs/
        main_log_file = main_logs_dir / "ast_extraction.log"
        h1 = RotatingFileHandler(
            str(main_log_file),
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,
            encoding="utf-8"
        )
        h1.setFormatter(formatter)
        _ast_detail_logger.addHandler(h1)

        # Mirror file in data/logs/ast/ for backwards compatibility
        mirror_file = ast_subdir / "ast_analysis.log"
        h2 = RotatingFileHandler(
            str(mirror_file),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        h2.setFormatter(formatter)
        _ast_detail_logger.addHandler(h2)

        logger.info(f"AST detail logger initialized → {main_log_file}")

    return _ast_detail_logger


def _write_extraction_logs(
    results: List[Dict[str, Any]],
    language: str,
    filename: Optional[str] = None,
    parser_engine: str = "Tree-sitter"
) -> Dict[str, Path]:
    """
    Writes rich structured extraction logs directly into data/logs/ folder:
    1. Appends full audit entry to data/logs/ast_extraction.log
    2. Overwrites latest JSON report at data/logs/ast_extracted_functions.json
    3. If filename provided, creates dedicated per-upload log:
       data/logs/ast_upload_<filename>.log and timestamped copy
    4. Mirrors to data/logs/ast/ directory for backward compatibility.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    main_logs_dir = base_dir / "data" / "logs"
    main_logs_dir.mkdir(parents=True, exist_ok=True)
    ast_subdir = main_logs_dir / "ast"
    ast_subdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_compact = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_display = filename or "unnamed_source_code"

    sep = "=" * 80
    summary_lines = []
    for i, r in enumerate(results):
        summary_lines.append(
            f"  [{i+1}] {r.get('function_name', 'unnamed')} | "
            f"Lines {r.get('start_line', '?')}-{r.get('end_line', '?')} | "
            f"Complexity: {r.get('complexity', 1)} | "
            f"Params: {r.get('parameters', [])} | "
            f"Return: '{r.get('return_type', '')}' | "
            f"Async: {r.get('is_async', False)}"
        )

    log_entry = (
        f"\n{sep}\n"
        f"TIMESTAMP          : {timestamp}\n"
        f"EVENT              : PROGRAM FILE FUNCTION EXTRACTION (TREE-SITTER)\n"
        f"SOURCE FILE        : {file_display}\n"
        f"LANGUAGE           : {language}\n"
        f"PARSER ENGINE      : {parser_engine}\n"
        f"TOTAL FUNCTIONS    : {len(results)}\n"
        f"FUNCTIONS EXTRACTED:\n" +
        ("\n".join(summary_lines) if summary_lines else "  No functions identified") +
        f"\n{sep}\n"
    )

    # 1. Append to primary ast_extraction.log
    al = _get_ast_logger()
    al.info(log_entry)

    # 2. Save JSON reports
    report_data = {
        "timestamp": timestamp,
        "event": "program_file_function_extraction",
        "file_name": file_display,
        "language": language,
        "parser_engine": parser_engine,
        "total_functions": len(results),
        "functions": results
    }
    json_str = json.dumps(report_data, indent=2)

    # Write in main data/logs/
    main_json = main_logs_dir / "ast_extracted_functions.json"
    main_json.write_text(json_str, encoding="utf-8")

    # Mirror in data/logs/ast/
    sub_json = ast_subdir / "ast_extracted_functions.json"
    sub_json.write_text(json_str, encoding="utf-8")
    (ast_subdir / f"ast_report_{ts_compact}.json").write_text(json_str, encoding="utf-8")

    written_paths = {"main_log": main_logs_dir / "ast_extraction.log", "main_json": main_json}

    # 3. Dedicated per-file upload log if filename is provided
    if filename:
        safe_fname = re.sub(r'[^a-zA-Z0-9_\.-]', '_', Path(filename).name)
        per_file_log = main_logs_dir / f"ast_upload_{safe_fname}.log"
        ts_per_file_log = main_logs_dir / f"ast_upload_{ts_compact}_{safe_fname}.log"

        file_specific_content = (
            f"{sep}\n"
            f"DEV-KNOW AST INTELLIGENCE - PROGRAM FILE UPLOAD EXTRACTION LOG\n"
            f"{sep}\n"
            f"Uploaded File     : {filename}\n"
            f"Timestamp         : {timestamp}\n"
            f"Detected Language : {language}\n"
            f"Parser Engine     : {parser_engine}\n"
            f"Extracted Count   : {len(results)} function(s)\n"
            f"{sep}\n\n"
            f"EXTRACTED FUNCTION DETAILS:\n"
        )
        for i, r in enumerate(results):
            file_specific_content += (
                f"\n--- Function #{i+1}: {r.get('function_name')} ---\n"
                f"  Header     : {r.get('function_header', '').strip()}\n"
                f"  Line Range : Lines {r.get('start_line')}-{r.get('end_line')}\n"
                f"  Parameters : {r.get('parameters', [])}\n"
                f"  Return Type: {r.get('return_type', 'None specified')}\n"
                f"  Complexity : {r.get('complexity', 1)}\n"
                f"  Async      : {r.get('is_async', False)}\n"
                f"  Code:\n{textwrap.indent(r.get('full_code', ''), '    ')}\n"
            )
        file_specific_content += f"\n{sep}\n[END OF EXTRACTION REPORT]\n{sep}\n"

        per_file_log.write_text(file_specific_content, encoding="utf-8")
        ts_per_file_log.write_text(file_specific_content, encoding="utf-8")
        written_paths["upload_log"] = per_file_log

    return written_paths


def _log_ast_result(
    phase: str,
    code_snippet: str,
    language: str,
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Writes a full structured log entry to the AST log file.
    Called after every parse_function_metadata / analyze_code_quality run.
    """
    al = _get_ast_logger()
    # Truncate code for readability (first 20 lines max)
    code_lines = code_snippet.strip().splitlines()
    code_preview = "\n".join(code_lines[:20])
    if len(code_lines) > 20:
        code_preview += f"\n... [{len(code_lines) - 20} more lines truncated]"

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "phase": phase,
        "language_input": language,
        "code_preview": code_preview,
        "result": result,
    }
    if extra:
        entry["extra"] = extra

    separator = "=" * 80
    al.info(
        f"\n{separator}\n"
        f"PHASE        : {phase}\n"
        f"LANGUAGE     : {language}\n"
        f"PARSER USED  : {result.get('parser_engine', result.get('parser_used', 'N/A'))}\n"
        f"FUNCTION     : {result.get('function_name', 'N/A')}\n"
        f"PARAMETERS   : {result.get('parameters', [])}\n"
        f"RETURN TYPE  : {result.get('return_type', 'N/A')}\n"
        f"COMPLEXITY   : {result.get('complexity', 'N/A')}\n"
        f"LINES OF CODE: {result.get('lines_of_code', 'N/A')}\n"
        f"SYNTAX VALID : {result.get('is_valid_syntax', 'N/A')}\n"
        f"SYNTAX ERROR : {result.get('syntax_error', 'None')}\n"
        f"CODE SMELLS  : {result.get('code_smells', [])}\n"
        f"IS ASYNC     : {result.get('is_async', 'N/A')}\n"
        f"EXTRA        : {json.dumps(extra, default=str) if extra else 'None'}\n"
        f"CODE PREVIEW :\n{textwrap.indent(code_preview, '  ')}\n"
        f"FULL RESULT  : {json.dumps(result, default=str, indent=2)}\n"
        f"{separator}\n"
    )

    # Write latest analysis JSON file in both data/logs/ and data/logs/ast/
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        main_logs_dir = base_dir / "data" / "logs"
        ast_dir = main_logs_dir / "ast"
        ast_dir.mkdir(parents=True, exist_ok=True)
        (main_logs_dir / "ast_latest_analysis.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        (ast_dir / "ast_latest_analysis.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

        print(
            f"🌳 [AST Inspection] {result.get('parser_engine', 'AST').upper()}: "
            f"Parsed '{result.get('function_name', 'unnamed')}' | "
            f"Complexity: {result.get('complexity', 1)} | "
            f"Syntax: {'Valid' if result.get('is_valid_syntax') else 'Irregular'} | "
            f"Params: {result.get('parameters', [])}",
            flush=True
        )
    except Exception:
        pass

# Attempt lazy import of tree_sitter_languages
_PARSERS: Dict[str, Any] = {}
_TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter_languages import get_parser  # type: ignore[import-untyped, import-not-found]
    # Pre-cache parsers for all supported languages
    for lang_key in ["python", "c_sharp", "javascript", "typescript", "java", "go", "cpp", "c"]:
        try:
            _PARSERS[lang_key] = get_parser(lang_key)
        except Exception as e:
            logger.warning(f"Could not load Tree-sitter parser for {lang_key}: {e}")
    _TREE_SITTER_AVAILABLE = len(_PARSERS) > 0
except (ImportError, Exception) as ts_err:
    logger.info(f"Tree-sitter not available ({ts_err}). Operating with native Python AST and regex.")


class CodeParserService:
    """
    Centralized code parsing and static AST analysis engine.
    Combines Tree-sitter and native Python AST for multi-language intelligence.
    """

    @classmethod
    def normalize_language(cls, lang: Optional[str]) -> str:
        """Normalizes language string to standardized keys"""
        if not lang:
            return "c_sharp"
        l = lang.strip().lower()
        if l in ("c#", "csharp", "cs"):
            return "c_sharp"
        if l in ("js", "javascript", "jsx", "mjs", "cjs"):
            return "javascript"
        if l in ("ts", "typescript", "tsx"):
            return "typescript"
        if l in ("py", "python"):
            return "python"
        if l in ("java",):
            return "java"
        if l in ("go", "golang"):
            return "go"
        if l in ("cpp", "c++", "cplusplus", "cc", "hpp"):
            return "cpp"
        if l in ("c", "h"):
            return "c"
        return l

    @classmethod
    def parse_function_metadata(cls, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts function name, parameters, return type, and async status
        using Tree-sitter or Python AST, with graceful regex fallback.
        Prioritizes Tree-sitter for multi-language deterministic extraction.
        """
        if not code or not code.strip():
            return {
                "function_name": "",
                "parameters": [],
                "return_type": "",
                "is_async": False,
                "complexity": 1,
                "parser_used": "none"
            }

        norm_lang = cls.normalize_language(language)

        # 1. Tree-sitter for all supported languages
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            ts_meta = cls._parse_tree_sitter(code, norm_lang)
            if ts_meta and ts_meta.get("function_name") and ts_meta.get("function_name") != "unnamed_function":
                return ts_meta

        # 2. Native Python AST for Python code
        if norm_lang == "python":
            py_meta = cls._parse_python_ast(code)
            if py_meta and py_meta.get("function_name"):
                return py_meta

        # 3. Fallback to Regex
        return cls._fallback_regex_parse(code, norm_lang)

    @classmethod
    def _parse_python_ast(cls, code: str) -> Optional[Dict[str, Any]]:
        """Parses Python code using the standard library ast module"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [a.arg for a in node.args.args if a.arg != "self"]
                    ret_type = ast.unparse(node.returns) if node.returns else ""
                    complexity = cls._calculate_python_complexity(node)
                    return {
                        "function_name": node.name,
                        "parameters": params,
                        "return_type": ret_type,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "complexity": complexity,
                        "parser_used": "python-ast"
                    }
        except SyntaxError:
            pass
        return None

    @classmethod
    def _parse_tree_sitter(cls, code: str, norm_lang: str) -> Optional[Dict[str, Any]]:
        """Parses single function/method code using Tree-sitter across C#, JS, TS, Python, Java, Go, C++"""
        parser = _PARSERS.get(norm_lang)
        if not parser:
            return None

        try:
            code_bytes = code.encode("utf-8")
            tree = parser.parse(code_bytes)
            root = tree.root_node

            target_node = None
            fn_name = ""
            ret_type = ""
            params = []
            is_async = False

            # Recognized function node types across Tree-sitter grammars
            func_types = (
                "method_declaration",
                "local_function_statement",
                "constructor_declaration",
                "function_declaration",
                "function_expression",
                "arrow_function",
                "function_definition",
                "method_definition",
                "generator_function_declaration"
            )

            # Walk AST to find function/method declaration
            def find_func_node(node):
                nonlocal target_node
                if target_node:
                    return
                if node.type in func_types:
                    target_node = node
                    return
                for child in node.children:
                    find_func_node(child)

            find_func_node(root)

            if not target_node:
                # If wrapped in export_statement
                for child in root.children:
                    if child.type == "export_statement":
                        decl = child.child_by_field_name("declaration")
                        if decl:
                            target_node = decl
                            break

            if target_node:
                # 1. Resolve Function Name
                name_node = target_node.child_by_field_name("name")
                if name_node:
                    fn_name = name_node.text.decode("utf-8")
                elif target_node.parent and target_node.parent.type in ("variable_declarator", "lexical_binding"):
                    vn = target_node.parent.child_by_field_name("name")
                    if vn:
                        fn_name = vn.text.decode("utf-8")
                elif target_node.parent and target_node.parent.type == "assignment_expression":
                    left = target_node.parent.child_by_field_name("left")
                    if left:
                        fn_name = left.text.decode("utf-8")
                elif target_node.parent and target_node.parent.type == "pair":
                    key = target_node.parent.child_by_field_name("key")
                    if key:
                        fn_name = key.text.decode("utf-8")

                # C/C++ declarator resolution
                if not fn_name:
                    decl_node = target_node.child_by_field_name("declarator")
                    while decl_node and decl_node.type in ("function_declarator", "pointer_declarator"):
                        decl_node = decl_node.child_by_field_name("declarator")
                    if decl_node and decl_node.type == "identifier":
                        fn_name = decl_node.text.decode("utf-8")

                # 2. Return Type
                for rf in ("type", "return_type", "result"):
                    type_node = target_node.child_by_field_name(rf)
                    if type_node:
                        ret_type = type_node.text.decode("utf-8").lstrip(": ").strip()
                        break

                # 3. Parameters
                params_node = target_node.child_by_field_name("parameters")
                if params_node:
                    for p in params_node.children:
                        p_text = p.text.decode("utf-8").strip()
                        if p_text and p_text not in (",", "(", ")"):
                            params.append(p_text)

                # 4. Async check
                first_slice = target_node.text.decode("utf-8")[:40].lower()
                if "async" in first_slice:
                    is_async = True

                complexity = cls._calculate_tree_sitter_complexity(target_node)

                if fn_name:
                    return {
                        "function_name": fn_name,
                        "parameters": params,
                        "return_type": ret_type,
                        "is_async": is_async,
                        "complexity": complexity,
                        "parser_used": "tree-sitter"
                    }
        except Exception as e:
            logger.debug(f"Tree-sitter parse notice: {e}")

        return None

    @classmethod
    def _fallback_regex_parse(cls, code: str, norm_lang: str) -> Dict[str, Any]:
        """Graceful regex fallback when AST parsing cannot find a complete function"""
        fn_name = ""
        first_line = code.strip().split("\n")[0]

        patterns = [
            r'(?:function|def|sub)\s+([a-zA-Z0-9_]+)\s*\(',
            r'(?:public|private|protected|internal|static|async|\s)+\s+(?:[\w<>\[\],\s]+)\s+([a-zA-Z0-9_]+)\s*\(',
            r'(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            r'([a-zA-Z0-9_]+)\s*\([^)]*\)'
        ]
        for pat in patterns:
            match = re.search(pat, first_line)
            if match:
                fn_name = match.group(1)
                break

        # Rough complexity estimate
        complexity = 1 + len(re.findall(r'\b(if|else if|for|while|catch|case|\?\?|&&|\|\|)\b', code))

        return {
            "function_name": fn_name or "unnamed_function",
            "parameters": [],
            "return_type": "",
            "is_async": "async" in first_line.lower(),
            "complexity": complexity,
            "parser_used": "regex_fallback"
        }

    @classmethod
    def validate_syntax(cls, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Validates whether code snippet has syntax errors.
        Returns: { "is_valid": bool, "error_message": str or None }
        """
        if not code or not code.strip():
            return {"is_valid": True, "error_message": None}

        norm_lang = cls.normalize_language(language)

        # 1. Python AST validation
        if norm_lang == "python":
            try:
                ast.parse(code)
                return {"is_valid": True, "error_message": None}
            except SyntaxError as se:
                return {
                    "is_valid": False,
                    "error_message": f"SyntaxError at line {se.lineno}: {se.msg}"
                }

        # 2. Tree-sitter error node validation
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                tree = parser.parse(code.encode("utf-8"))
                has_error = tree.root_node.has_error
                if not has_error:
                    return {"is_valid": True, "error_message": None}

                # Find first error node
                error_node = None
                def find_error(n):
                    nonlocal error_node
                    if error_node:
                        return
                    if n.type == "ERROR" or n.is_missing:
                        error_node = n
                        return
                    for c in n.children:
                        find_error(c)

                find_error(tree.root_node)
                err_text = error_node.text.decode("utf-8") if error_node else "Syntax irregularity"
                return {
                    "is_valid": False,
                    "error_message": f"Tree-sitter parse error near: '{err_text[:40]}'"
                }
            except Exception as e:
                return {"is_valid": True, "error_message": None}

        return {"is_valid": True, "error_message": None}

    @classmethod
    def _calculate_python_complexity(cls, func_node) -> int:
        """Calculates McCabe Cyclomatic Complexity for Python AST"""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.IfExp):
                complexity += 1
        return complexity

    @classmethod
    def _calculate_tree_sitter_complexity(cls, func_node) -> int:
        """Calculates Cyclomatic Complexity by traversing Tree-sitter branching nodes"""
        branch_types = {
            "if_statement", "while_statement", "for_statement", "for_in_statement",
            "catch_clause", "switch_case", "conditional_expression", "binary_expression"
        }
        complexity = 1

        def count_branches(n):
            nonlocal complexity
            if n.type in branch_types:
                if n.type == "binary_expression":
                    # Only logical && and ||
                    op = n.child_by_field_name("operator")
                    if op and op.text.decode("utf-8") in ("&&", "||", "and", "or"):
                        complexity += 1
                else:
                    complexity += 1
            for child in n.children:
                count_branches(child)

        count_branches(func_node)
        return complexity

    @classmethod
    def analyze_code_quality(cls, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs comprehensive deterministic code quality, complexity,
        and defensive programming inspection.
        Writes a full structured report to data/logs/ast/ast_analysis.log.
        """
        norm_lang = cls.normalize_language(language)
        meta = cls.parse_function_metadata(code, norm_lang)
        syntax = cls.validate_syntax(code, norm_lang)

        smells = []
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        loc = len(lines)

        # Complexity risk
        complexity = meta.get("complexity", 1)
        if complexity > 15:
            smells.append(f"High cyclomatic complexity ({complexity}). Consider breaking into smaller helper functions.")
        elif complexity > 8:
            smells.append(f"Moderate cyclomatic complexity ({complexity}). Ensure all branches have unit test coverage.")

        # Parameter null safety check
        params = meta.get("parameters", [])
        if params:
            param_names = [p.split()[-1].replace(":", "") for p in params]
            for p_name in param_names[:3]:
                # Look if p_name is accessed with '.' without checking if it's null
                if f"{p_name}." in code and not any(check in code for check in [f"!{p_name}", f"{p_name} == null", f"{p_name} is null", f"{p_name} is None", f"if {p_name}:", f"?."]):
                    smells.append(f"Parameter '{p_name}' accessed via dot-notation without apparent null-check guard.")
                    break

        result = {
            "function_name": meta.get("function_name", "unnamed"),
            "language": norm_lang,
            "parameters": meta.get("parameters", []),
            "return_type": meta.get("return_type", ""),
            "complexity": complexity,
            "lines_of_code": loc,
            "is_valid_syntax": syntax["is_valid"],
            "syntax_error": syntax["error_message"],
            "code_smells": smells,
            "parser_engine": meta.get("parser_used", "none"),
            "is_async": meta.get("is_async", False),
        }

        # ── Write full AST analysis to dedicated log file ──────────────────
        try:
            _log_ast_result(
                phase="analyze_code_quality",
                code_snippet=code,
                language=norm_lang,
                result=result,
                extra={
                    "raw_parser_meta": meta,
                    "syntax_check": syntax,
                    "lines_in_code": loc,
                    "tree_sitter_available": _TREE_SITTER_AVAILABLE,
                    "loaded_parsers": list(_PARSERS.keys()),
                }
            )
        except Exception as log_err:
            logger.debug(f"AST logger write error (non-critical): {log_err}")

        return result

    @classmethod
    def extract_all_functions(
        cls,
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses an entire source code file (.py, .cs, .ts, .js, .jsx, .tsx, .java, .go, .cpp, .c)
        using Tree-sitter to deterministically extract every function/method.
        Directly saves structured extraction logs into data/logs/ folder.
        """
        if not code or not code.strip():
            return []

        norm_lang = cls.normalize_language(language)
        results = []
        parser_used = "none"

        # 1. Tree-sitter AST extraction (Primary Engine across all languages)
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                code_bytes = code.encode("utf-8")
                tree = parser.parse(code_bytes)
                lines = code.split("\n")

                func_types = (
                    "method_declaration",
                    "local_function_statement",
                    "constructor_declaration",
                    "function_declaration",
                    "function_expression",
                    "arrow_function",
                    "function_definition",
                    "method_definition",
                    "generator_function_declaration"
                )

                found_nodes = []
                def collect_nodes(n):
                    if n.type in func_types:
                        found_nodes.append(n)
                    for child in n.children:
                        collect_nodes(child)

                collect_nodes(tree.root_node)

                for node in found_nodes:
                    # 1. Function Name Extraction
                    fn_name = ""
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        fn_name = name_node.text.decode("utf-8")
                    elif node.parent and node.parent.type in ("variable_declarator", "lexical_binding"):
                        vn = node.parent.child_by_field_name("name")
                        if vn:
                            fn_name = vn.text.decode("utf-8")
                    elif node.parent and node.parent.type == "assignment_expression":
                        left = node.parent.child_by_field_name("left")
                        if left:
                            fn_name = left.text.decode("utf-8")
                    elif node.parent and node.parent.type == "pair":
                        key = node.parent.child_by_field_name("key")
                        if key:
                            fn_name = key.text.decode("utf-8")

                    # C / C++ Declarator identifier
                    if not fn_name:
                        decl = node.child_by_field_name("declarator")
                        while decl and decl.type in ("function_declarator", "pointer_declarator"):
                            decl = decl.child_by_field_name("declarator")
                        if decl and decl.type == "identifier":
                            fn_name = decl.text.decode("utf-8")

                    # If still unnamed, provide clean identifiable placeholder
                    if not fn_name:
                        fn_name = f"anonymous_fn_L{node.start_point[0] + 1}"

                    # 2. Return Type
                    ret_type = ""
                    for rf in ("type", "return_type", "result"):
                        type_node = node.child_by_field_name(rf)
                        if type_node:
                            ret_type = type_node.text.decode("utf-8").lstrip(": ").strip()
                            break

                    # 3. Parameters
                    params = []
                    params_node = node.child_by_field_name("parameters")
                    if params_node:
                        for p in params_node.children:
                            if p.type in ("parameter", "identifier", "required_parameter", "optional_parameter",
                                          "default_parameter", "typed_parameter", "typed_default_parameter",
                                          "formal_parameters"):
                                p_text = p.text.decode("utf-8").strip()
                                if p_text and p_text not in (",", "(", ")"):
                                    params.append(p_text)

                    # 4. Line Range & Code
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    fn_lines = lines[start_line - 1 : end_line]
                    fn_full = "\n".join(fn_lines)
                    header = lines[start_line - 1].strip()
                    body = "\n".join(lines[start_line : end_line]) if len(lines) > start_line else fn_full
                    complexity = cls._calculate_tree_sitter_complexity(node)

                    # 5. Async check
                    is_async = "async" in node.text.decode("utf-8")[:40].lower()

                    results.append({
                        "function_name": fn_name,
                        "function_header": header,
                        "function_body": body,
                        "full_code": fn_full,
                        "parameters": params,
                        "return_type": ret_type,
                        "start_line": start_line,
                        "end_line": end_line,
                        "complexity": complexity,
                        "is_async": is_async,
                        "language": norm_lang,
                        "parser_engine": "tree-sitter"
                    })

                if results:
                    parser_used = "tree-sitter"

            except Exception as ts_err:
                logger.debug(f"Tree-sitter full file extraction notice: {ts_err}")

        # 2. Fallback to Native Python AST for Python files
        if not results and norm_lang == "python":
            try:
                tree = ast.parse(code)
                lines = code.split("\n")
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        start_line = node.lineno
                        end_line = getattr(node, "end_lineno", node.lineno)
                        fn_lines = lines[start_line - 1 : end_line]
                        fn_full = "\n".join(fn_lines)
                        header = lines[start_line - 1].strip()
                        body = "\n".join(lines[start_line : end_line])
                        params = [a.arg for a in node.args.args if a.arg != "self"]
                        ret_type = ast.unparse(node.returns) if node.returns else ""
                        complexity = cls._calculate_python_complexity(node)
                        results.append({
                            "function_name": node.name,
                            "function_header": header,
                            "function_body": body,
                            "full_code": fn_full,
                            "parameters": params,
                            "return_type": ret_type,
                            "start_line": start_line,
                            "end_line": end_line,
                            "complexity": complexity,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "language": "python",
                            "parser_engine": "python-ast"
                        })
                if results:
                    parser_used = "python-ast"
            except Exception as pe:
                logger.debug(f"Python full-file AST fallback notice: {pe}")

        # 3. Fallback to Regex if AST failed completely
        if not results:
            meta = cls._fallback_regex_parse(code, norm_lang)
            if meta.get("function_name") and meta["function_name"] != "unnamed_function":
                lines = code.split("\n")
                results.append({
                    "function_name": meta["function_name"],
                    "function_header": lines[0].strip()[:150],
                    "function_body": code.strip(),
                    "full_code": code.strip(),
                    "parameters": meta.get("parameters", []),
                    "return_type": meta.get("return_type", ""),
                    "start_line": 1,
                    "end_line": len(lines),
                    "complexity": meta.get("complexity", 1),
                    "is_async": meta.get("is_async", False),
                    "language": norm_lang,
                    "parser_engine": "regex_fallback"
                })
                parser_used = "regex_fallback"

        # ── Write rich structured logs directly into data/logs/ ───────────────
        try:
            written_paths = _write_extraction_logs(
                results=results,
                language=norm_lang,
                filename=filename,
                parser_engine=parser_used
            )

            sep = "=" * 80
            print(f"\n{sep}", flush=True)
            print(f"🌳 [Tree-sitter AST Engine] File Extraction Complete ({norm_lang.upper()}):", flush=True)
            print(f"• Source File: {filename or 'In-Memory Code'}", flush=True)
            print(f"• Parser Used: {parser_used}", flush=True)
            print(f"• Total Functions Extracted: {len(results)}", flush=True)
            for i, r in enumerate(results[:5]):
                print(f"  [{i+1}] {r['function_name']} | lines {r.get('start_line')}-{r.get('end_line')} "
                      f"| complexity {r.get('complexity')} | params: {r.get('parameters')} | ret: '{r.get('return_type', '')}'", flush=True)
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more functions", flush=True)
            print(f"📁 AST Extraction Log: {written_paths.get('main_log')}", flush=True)
            if "upload_log" in written_paths:
                print(f"📁 Dedicated Upload Log: {written_paths['upload_log']}", flush=True)
            print(f"{sep}\n", flush=True)

        except Exception as log_err:
            logger.debug(f"AST extraction log write error (non-critical): {log_err}")

        return results


# Singleton export
code_parser = CodeParserService()
