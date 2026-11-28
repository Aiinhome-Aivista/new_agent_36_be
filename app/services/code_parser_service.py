"""
CodeParserService: Unified Tree-sitter Code Intelligence Engine for DevKnow.

This module provides deterministic AST analysis, syntax validation, function signature
extraction, cyclomatic complexity calculation, and code smell detection across:
- C# (.cs)
- TypeScript (.ts, .tsx)
- JavaScript (.js, .jsx)
- Python (.py)

Key Principles:
1. Unified Tree-sitter Grammar: Uses concrete syntax trees to parse all languages consistently.
2. Resilience: Employs fallback regex parsers when syntax trees are partial or irregular.
3. Observability: Emits detailed per-file JSON reports and human-readable audit logs
   to `data/logs/treesitter/`.
"""

import re
import json
import logging
import warnings
import textwrap
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional, Set

# Suppress deprecation warnings from C-extensions / tree_sitter legacy bindings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

logger = logging.getLogger("DevKnow.CodeParser")

# ── Dedicated Tree-sitter Detail Logger ──────────────────────────────────────
_treesitter_detail_logger: Optional[logging.Logger] = None


def _get_treesitter_logger() -> logging.Logger:
    """
    Returns the dedicated Tree-sitter detail file logger, creating it on first call.

    Writes to: `<backend_root>/data/logs/treesitter/treesitter_analysis.log`.
    Configured with RotatingFileHandler (5 MB per file, max 5 backup generations).
    Logs are kept isolated from the root Flask logger to avoid log pollution.

    Returns:
        logging.Logger: The initialized Tree-sitter detail logger instance.
    """
    global _treesitter_detail_logger
    if _treesitter_detail_logger is not None:
        return _treesitter_detail_logger

    base_dir = Path(__file__).resolve().parent.parent.parent
    ts_log_dir = base_dir / "data" / "logs" / "treesitter"
    ts_log_dir.mkdir(parents=True, exist_ok=True)

    _treesitter_detail_logger = logging.getLogger("DevKnow.TreeSitter.Detail")
    _treesitter_detail_logger.setLevel(logging.DEBUG)
    _treesitter_detail_logger.propagate = False

    if not _treesitter_detail_logger.handlers:
        log_file = ts_log_dir / "treesitter_analysis.log"
        handler = RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _treesitter_detail_logger.addHandler(handler)
        logger.info(f"Tree-sitter detail logger initialized -> {log_file}")

    return _treesitter_detail_logger


def _log_treesitter_result(
    phase: str,
    code_snippet: str,
    language: str,
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Writes a formatted, structured audit entry to the Tree-sitter detail log and JSON pointer.

    Args:
        phase: Parsing phase description (e.g. 'analyze_code_quality', 'parse_function_metadata').
        code_snippet: The input source code snippet analyzed.
        language: Canonical language tag (e.g. 'c_sharp', 'python').
        result: The output dictionary produced by AST evaluation.
        extra: Optional additional diagnostic metadata.
    """
    tl = _get_treesitter_logger()
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
    tl.info(
        f"\n{separator}\n"
        f"PHASE        : {phase}\n"
        f"LANGUAGE     : {language}\n"
        f"PARSER USED  : {result.get('parser_engine', result.get('parser_used', 'tree-sitter'))}\n"
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

    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        ts_dir = base_dir / "data" / "logs" / "treesitter"
        ts_dir.mkdir(parents=True, exist_ok=True)
        latest_analysis_file = ts_dir / "treesitter_latest_analysis.json"
        latest_analysis_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")

        logger.debug(
            f"  [Tree-sitter Inspection] "
            f"Parsed '{result.get('function_name', 'unnamed')}' | "
            f"Complexity: {result.get('complexity', 1)} | "
            f"Syntax: {'Valid' if result.get('is_valid_syntax') else 'Irregular'} | "
            f"Params: {result.get('parameters', [])}"
        )
    except Exception:
        pass


# ── Lazy Parser Loading & Grammars Pre-caching ──────────────────────────────
_PARSERS: Dict[str, Any] = {}
_TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter_languages import get_parser  # type: ignore[import-untyped, import-not-found]
    for lang_key in ["python", "c_sharp", "javascript", "typescript", "java", "go", "cpp", "c"]:
        try:
            _PARSERS[lang_key] = get_parser(lang_key)
        except Exception as e:
            logger.warning(f"Could not load Tree-sitter parser for {lang_key}: {e}")
    _TREE_SITTER_AVAILABLE = len(_PARSERS) > 0
except (ImportError, Exception) as ts_err:
    logger.info(f"Tree-sitter not available ({ts_err}). Operating with native fallback regex.")


class CodeParserService:
    """
    Centralized code parsing and static AST analysis engine.

    Extracts metadata, function boundaries, arguments, types, and computes
    cyclomatic complexity using Tree-sitter concrete syntax trees.
    """

    @classmethod
    def normalize_language(cls, lang: Optional[str]) -> str:
        """
        Normalizes arbitrary language strings to standardized Tree-sitter grammar keys.

        Args:
            lang: Raw language input (e.g., 'C#', 'cs', 'python', 'js', 'TS').

        Returns:
            Normalized grammar key (e.g. 'c_sharp', 'javascript', 'typescript', 'python').
        """
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
        Extracts function name, parameters, return type, and async status.

        Uses Tree-sitter AST parsing first, then falls back to resilient regex if syntax
        is incomplete or outside grammar boundaries.

        Args:
            code: Source code or function signature string.
            language: Target programming language identifier.

        Returns:
            Dict containing function_name, parameters, return_type, is_async, complexity, parser_used.
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

        # 1. Primary Path: Tree-sitter across C#, TypeScript, JavaScript, Python
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            ts_meta = cls._parse_tree_sitter(code, norm_lang)
            if ts_meta and ts_meta.get("function_name"):
                return ts_meta

        # 2. Resilient Fallback: Regex tokenization
        return cls._fallback_regex_parse(code, norm_lang)

    @classmethod
    def _parse_tree_sitter(cls, code: str, norm_lang: str) -> Optional[Dict[str, Any]]:
        """
        Internal Tree-sitter parser traversal to identify function nodes and metadata.

        Args:
            code: Source code string to parse.
            norm_lang: Canonical language key.

        Returns:
            Parsed metadata dictionary or None if no function node was discovered.
        """
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
            params: List[str] = []
            is_async = False

            # Recognized function node types across Tree-sitter grammars
            func_types = (
                "method_declaration",
                "local_function_statement",
                "function_declaration",
                "function_expression",
                "arrow_function",
                "function_definition",
                "method_definition"
            )

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

            # Check if wrapped inside decorators or export statements
            if not target_node:
                for child in root.children:
                    if child.type in ("export_statement", "decorated_definition"):
                        for sub in child.children:
                            if sub.type in ("function_definition", "method_declaration", "function_declaration"):
                                target_node = sub
                                break
                    if target_node:
                        break

            if target_node:
                # Extract function identifier
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

                # Extract return type annotation
                type_node = target_node.child_by_field_name("type") or target_node.child_by_field_name("return_type")
                if type_node:
                    ret_type = type_node.text.decode("utf-8").lstrip(": ").strip()

                # Extract parameter list
                params_node = target_node.child_by_field_name("parameters")
                if params_node:
                    for p in params_node.children:
                        if p.type not in (",", "(", ")"):
                            p_text = p.text.decode("utf-8").strip()
                            if p_text and p_text not in (",", "(", ")") and p_text != "self":
                                params.append(p_text)

                # Async inspection
                if "async" in target_node.text.decode("utf-8")[:30].lower():
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
        """
        Graceful regex fallback when AST parsing cannot find a complete function node.

        Args:
            code: Source snippet.
            norm_lang: Canonical language key.

        Returns:
            Dict containing best-effort metadata estimates.
        """
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

        # Rough complexity estimate via control flow keyword occurrences
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
        Validates whether code snippet has syntax errors using Tree-sitter error nodes.

        Args:
            code: Code snippet to validate.
            language: Programming language key.

        Returns:
            Dict: { "is_valid": bool, "error_message": str or None }
        """
        if not code or not code.strip():
            return {"is_valid": True, "error_message": None}

        norm_lang = cls.normalize_language(language)

        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                tree = parser.parse(code.encode("utf-8"))
                has_error = tree.root_node.has_error
                if not has_error:
                    return {"is_valid": True, "error_message": None}

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
            except Exception:
                return {"is_valid": True, "error_message": None}

        return {"is_valid": True, "error_message": None}

    @classmethod
    def _calculate_tree_sitter_complexity(cls, func_node) -> int:
        """
        Calculates Cyclomatic Complexity by traversing AST branching nodes.

        Branches counted:
        - `if_statement`, `elif_clause`
        - `while_statement`, `for_statement`, `for_in_statement`
        - `catch_clause`, `except_clause`
        - `switch_case`, `conditional_expression`
        - Binary short-circuit boolean operators (`&&`, `||`, `and`, `or`)

        Args:
            func_node: Tree-sitter Node corresponding to the function definition.

        Returns:
            Integer cyclomatic complexity (starting from 1).
        """
        branch_types = {
            "if_statement", "elif_clause", "while_statement", "for_statement", "for_in_statement",
            "catch_clause", "except_clause", "switch_case", "conditional_expression",
            "binary_expression", "boolean_operator"
        }
        complexity = 1

        def count_branches(n):
            nonlocal complexity
            if n.type in branch_types:
                if n.type in ("binary_expression", "boolean_operator"):
                    op = n.child_by_field_name("operator")
                    op_text = op.text.decode("utf-8") if op else n.text.decode("utf-8")
                    if any(tok in op_text for tok in ("&&", "||", "and", "or")):
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

        Writes a full structured report to data/logs/treesitter/treesitter_analysis.log.

        Args:
            code: Source code string to evaluate.
            language: Target programming language.

        Returns:
            Dict containing quality metrics, complexity, syntax status, and detected smells.
        """
        norm_lang = cls.normalize_language(language)
        meta = cls.parse_function_metadata(code, norm_lang)
        syntax = cls.validate_syntax(code, norm_lang)

        smells: List[str] = []
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        loc = len(lines)

        # 1. Complexity thresholds
        complexity = meta.get("complexity", 1)
        if complexity > 15:
            smells.append(f"High cyclomatic complexity ({complexity}). Consider breaking into smaller helper functions.")
        elif complexity > 8:
            smells.append(f"Moderate cyclomatic complexity ({complexity}). Ensure all branches have unit test coverage.")

        # 2. Defensive parameter null safety guard check
        params = meta.get("parameters", [])
        if params:
            param_names = [p.split()[-1].replace(":", "") for p in params]
            for p_name in param_names[:3]:
                if f"{p_name}." in code and not any(check in code for check in [
                    f"!{p_name}", f"{p_name} == null", f"{p_name} is null",
                    f"{p_name} is None", f"if {p_name}:", f"?."
                ]):
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

        # 3. Write structured audit entry to dedicated Tree-sitter log
        try:
            _log_treesitter_result(
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
            logger.debug(f"Tree-sitter logger write error: {log_err}")

        return result

    @classmethod
    def extract_all_functions(
        cls,
        code: str,
        language: Optional[str] = None,
        source_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses an entire source code file (.cs, .ts, .js, .py) and extracts
        every function/method definition as a distinct, atomic chunk.

        Generates dedicated per-file Tree-sitter audit logs and JSON artifacts
        in `data/logs/treesitter/`.

        Args:
            code: Full file source contents.
            language: Language extension or name.
            source_filename: Optional filename for logging and reporting.

        Returns:
            List of dictionaries representing individual extracted functions.
        """
        if not code or not code.strip():
            return []

        norm_lang = cls.normalize_language(language)
        results: List[Dict[str, Any]] = []

        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                code_bytes = code.encode("utf-8")
                tree = parser.parse(code_bytes)
                lines = code.split("\n")

                # Extract top-level module docstring & file-level imports for parent context
                module_docstring = ""
                file_imports: List[str] = []

                if tree.root_node:
                    for child in tree.root_node.children:
                        if not module_docstring and child.type == "expression_statement":
                            for sub in child.children:
                                if sub.type == "string":
                                    raw_text = sub.text.decode("utf-8", errors="replace").strip("\"' \n\r\t")
                                    module_docstring = " ".join([line.strip() for line in raw_text.splitlines() if line.strip()][:5])
                                    break
                        elif child.type in ("import_statement", "import_from_statement", "using_directive"):
                            imp_str = child.text.decode("utf-8", errors="replace").strip()
                            if imp_str:
                                file_imports.append(imp_str)

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

                found_nodes: List[Any] = []

                def collect_nodes(n):
                    if n.type in func_types:
                        found_nodes.append(n)
                    for child in n.children:
                        collect_nodes(child)

                collect_nodes(tree.root_node)

                for node in found_nodes:
                    name_node = node.child_by_field_name("name")
                    fn_name = name_node.text.decode("utf-8") if name_node else "unnamed_function"

                    type_node = node.child_by_field_name("type") or node.child_by_field_name("return_type")
                    ret_type = type_node.text.decode("utf-8").lstrip(": ").strip() if type_node else ""

                    params: List[str] = []
                    params_node = node.child_by_field_name("parameters")
                    if params_node:
                        for p in params_node.children:
                            if p.type not in (",", "(", ")"):
                                p_text = p.text.decode("utf-8").strip()
                                if p_text and p_text not in (",", "(", ")") and p_text != "self":
                                    params.append(p_text)

                    # Account for function decorators (e.g. @classmethod, @app.route)
                    start_node = node
                    if node.parent and node.parent.type == "decorated_definition":
                        start_node = node.parent

                    # Parent Class Hierarchy Resolution
                    class_name = ""
                    curr = node.parent
                    while curr:
                        if curr.type in (
                            "class_definition", "class_declaration",
                            "interface_declaration", "class_specifier",
                            "struct_specifier"
                        ):
                            c_name_node = curr.child_by_field_name("name")
                            if c_name_node:
                                class_name = c_name_node.text.decode("utf-8", errors="replace").strip()
                                break
                        curr = curr.parent

                    start_line = start_node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    fn_lines = lines[start_line - 1: end_line]
                    fn_full = "\n".join(fn_lines)
                    header = lines[start_line - 1].strip()
                    body = "\n".join(lines[start_line: end_line]) if len(lines) > start_line else fn_full
                    complexity = cls._calculate_tree_sitter_complexity(node)

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
                        "language": norm_lang,
                        "class_name": class_name,
                        "module_docstring": module_docstring,
                        "imports": file_imports
                    })

            except Exception as ts_err:
                logger.debug(f"Tree-sitter file extraction notice: {ts_err}")

        # ── Write comprehensive per-file audit artifacts ─────────────────────
        try:
            tl = _get_treesitter_logger()
            sep = "=" * 80
            summary_lines = [
                f"  [{i+1}] {r['function_name']} | lines {r.get('start_line')}-{r.get('end_line')} "
                f"| complexity {r.get('complexity')} | params: {r.get('parameters')} | return: '{r.get('return_type', '')}'"
                for i, r in enumerate(results)
            ]
            log_msg = (
                f"\n{sep}\n"
                f"FILE                   : {source_filename or 'Direct Snippet'}\n"
                f"PHASE                  : extract_all_functions\n"
                f"LANGUAGE               : {norm_lang}\n"
                f"TOTAL FUNCTIONS FOUND  : {len(results)}\n"
                f"FUNCTIONS EXTRACTED:\n" +
                ("\n".join(summary_lines) if summary_lines else "  None identified") +
                f"\n{sep}\n"
            )
            tl.info(log_msg)

            base_dir = Path(__file__).resolve().parent.parent.parent
            ts_dir = base_dir / "data" / "logs" / "treesitter"
            ts_dir.mkdir(parents=True, exist_ok=True)

            if source_filename:
                clean_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(source_filename).stem)
                clean_ext = Path(source_filename).suffix.lstrip(".")
                file_tag = f"{clean_stem}_{clean_ext}" if clean_ext else clean_stem
            else:
                file_tag = norm_lang

            ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            report = {
                "source_file": source_filename or "direct_input",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "phase": "extract_all_functions",
                "engine": "tree-sitter",
                "language": norm_lang,
                "total_functions": len(results),
                "functions": results
            }

            per_file_json = ts_dir / f"treesitter_{file_tag}_{ts_str}.json"
            per_file_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

            per_file_log = ts_dir / f"treesitter_{file_tag}_{ts_str}.log"
            log_body = (
                f"{sep}\n"
                f"TREE-SITTER EXTRACTION REPORT FOR: {source_filename or 'Direct Code Input'}\n"
                f"TIMESTAMP                        : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"PARSER ENGINE                    : Tree-sitter\n"
                f"LANGUAGE                         : {norm_lang.upper()}\n"
                f"TOTAL FUNCTIONS DISCOVERED       : {len(results)}\n"
                f"{sep}\n\n"
                f"FUNCTION SIGNATURES SUMMARY:\n"
                + ("\n".join(summary_lines) if summary_lines else "  No functions discovered.")
                + f"\n\n{sep}\n"
                f"DETAILED FUNCTION BODIES:\n"
                f"{sep}\n"
            )
            for i, r in enumerate(results):
                log_body += (
                    f"\n--- [Function {i+1}]: {r['function_name']} (lines {r.get('start_line')}-{r.get('end_line')}) ---\n"
                    f"Complexity : {r.get('complexity', 1)}\n"
                    f"Parameters : {r.get('parameters', [])}\n"
                    f"Return Type: {r.get('return_type', 'unspecified')}\n"
                    f"Header     : {r.get('function_header', '')}\n"
                    f"Full Code  :\n{r.get('full_code', '')}\n"
                )
            per_file_log.write_text(log_body, encoding="utf-8")

            latest_file = ts_dir / "treesitter_extracted_functions.json"
            latest_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

            logger.debug(
                f"[Tree-sitter Engine] Full File Extraction Complete: {source_filename or 'Direct Input'} | "
                f"Language: {norm_lang.upper()} | Functions: {len(results)} | Log: {per_file_log.name}"
            )

        except Exception as log_err:
            logger.debug(f"Tree-sitter logger write error: {log_err}")

        return results

    # ── Multi-Language Built-ins & Logging Calls to Ignore in Call-Graph ──────
    IGNORED_CALLS: Set[str] = {
        # Python built-ins & standard methods
        "print", "len", "range", "str", "int", "float", "bool", "dict", "list", "set", "tuple",
        "super", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
        "append", "extend", "insert", "remove", "pop", "clear", "index", "count", "sort", "reverse",
        "keys", "values", "items", "get", "update", "split", "join", "strip", "lstrip", "rstrip",
        "replace", "lower", "upper", "startswith", "endswith", "format", "encode", "decode",
        "sleep", "round", "abs", "sum", "min", "max", "any", "all", "enumerate", "zip", "open",
        "exit", "quit", "help", "input", "id", "hash", "repr", "next", "iter",
        "dumps", "loads", "dump", "load", "read", "write", "close", "flush", "seek", "tell",

        # JavaScript / TypeScript standard library & DOM
        "console", "log", "warn", "error", "info", "debug", "trace", "dir",
        "parseInt", "parseFloat", "toString", "valueOf", "slice", "splice", "push", "shift", "unshift",
        "map", "filter", "reduce", "forEach", "find", "findIndex", "some", "every", "includes",
        "indexOf", "concat", "match", "search", "trim", "toLowerCase", "toUpperCase",
        "substring", "substr", "charAt", "charCodeAt", "setTimeout", "setInterval", "clearTimeout",
        "clearInterval", "Promise", "resolve", "reject", "then", "catch", "finally",

        # C# / Java / Go / C++ common methods & loggers
        "WriteLine", "Write", "ToString", "Equals", "GetHashCode", "GetType", "CompareTo",
        "IsNullOrEmpty", "IsNullOrWhiteSpace", "Format", "Add", "Remove", "Contains", "Count",
        "Clear", "Select", "Where", "ToList", "ToArray", "FirstOrDefault", "Any", "All",
        "Println", "Printf", "Sprintf", "Fatal", "Panic", "make", "new", "close", "panic", "recover",

        # Observability & logging frameworks
        "debug", "info", "warning", "error", "critical", "exception", "audit", "metric", "counter"
    }

    @classmethod
    def extract_invoked_calls(
        cls,
        code: str,
        language: Optional[str] = None
    ) -> List[str]:
        """
        Extracts all function and method identifiers invoked within a code snippet using Tree-sitter.
        Filters out language built-ins, standard libraries, and logging/observability noise.

        Handles:
        - Python: `call` nodes (e.g. `_create_engine()`, `self._validate_ssl()`)
        - JS/TS: `call_expression` nodes (e.g. `formatDate()`, `this.logger.info()`)
        - C#: `invocation_expression` nodes (e.g. `BuildConnection()`, `client.Send()`)
        - Java/Go/C++: standard call expressions

        Returns:
            List of distinct domain function/method names invoked.
        """
        if not code or not code.strip():
            return []

        norm_lang = cls.normalize_language(language)
        calls: List[str] = []
        ignored_lower = {c.lower() for c in cls.IGNORED_CALLS}

        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                tree = parser.parse(code.encode("utf-8"))
                call_types = ("call", "call_expression", "invocation_expression", "method_invocation")

                def collect_calls(node):
                    if node.type in call_types:
                        # Extract called identifier/attribute
                        fn_node = node.child_by_field_name("function") or node.child_by_field_name("name") or (node.children[0] if node.children else None)
                        if fn_node:
                            raw_name = fn_node.text.decode("utf-8", errors="replace").strip()
                            # Strip object prefixes (e.g. self._create_engine -> _create_engine, this.format -> format)
                            leaf_name = raw_name.split(".")[-1].split("->")[-1].strip()
                            # Filter out keywords, builtins, short tokens, and logging prefixes
                            if leaf_name and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', leaf_name):
                                if leaf_name.lower() not in ignored_lower and leaf_name not in cls.IGNORED_CALLS:
                                    if not any(raw_name.startswith(p) for p in (
                                        "logger.", "logging.", "console.", "metrics.", "tracer.", "audit.",
                                        "json.", "os.", "sys.", "re.", "math.", "time.", "datetime.", "random.", "pathlib.", "io."
                                    )):
                                        calls.append(leaf_name)

                    for child in node.children:
                        collect_calls(child)

                if tree.root_node:
                    collect_calls(tree.root_node)

            except Exception as e:
                logger.debug(f"Tree-sitter call extraction notice: {e}")

        # Resilient fallback if Tree-sitter didn't extract or unavailable
        if not calls:
            found = re.findall(r'(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            common_ignores = {"if", "for", "while", "switch", "catch", "return"} | ignored_lower
            calls = [fn for fn in found if fn.lower() not in common_ignores]

        return list(dict.fromkeys(calls))

    @classmethod
    def generate_skeleton_stub(
        cls,
        code: str,
        language: Optional[str] = None,
        max_lines: int = 50,
        source_file: Optional[str] = None
    ) -> str:
        """
        Generates a compact 'Skeleton Stub' (Signature + Parameters + Docstring + Contract)
        for a helper function when the full body exceeds token budgets or threshold lines.

        Example:
        def create_connection_pool(db_url: str, pool_size: int = 10) -> ConnectionPool:
            \"\"\"Initializes and returns a pooled database connection manager.\"\"\"
            ...  # [Implementation omitted: 280 lines from db/pool.py]

        Args:
            code: Full function source code.
            language: Programming language key.
            max_lines: Maximum lines allowed before stubbing.
            source_file: Optional source filename to annotate stub origin.

        Returns:
            str: Full code if <= max_lines, otherwise compact skeleton stub.
        """
        if not code or not code.strip():
            return ""

        raw_lines = code.strip().split("\n")
        total_lines = len(raw_lines)
        if total_lines <= max_lines:
            return code

        norm_lang = cls.normalize_language(language)
        file_hint = f" from {source_file}" if source_file else ""

        if norm_lang == "python":
            header_lines = []
            docstring_lines = []
            in_header = True
            in_docstring = False
            docstring_delim = None

            for line in raw_lines:
                if in_header:
                    header_lines.append(line)
                    if line.rstrip().endswith(":"):
                        in_header = False
                    continue

                stripped = line.strip()
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_delim = stripped[:3]
                        docstring_lines.append(line)
                        if stripped.endswith(docstring_delim) and len(stripped) > 3:
                            break
                    elif stripped:
                        break  # Non-docstring code reached
                else:
                    docstring_lines.append(line)
                    if stripped.endswith(docstring_delim):
                        break

            stub_header = "\n".join(header_lines)
            stub_doc = ("\n" + "\n".join(docstring_lines)) if docstring_lines else ""
            indent = "    "
            stub_body = f"\n{indent}...  # [Implementation omitted: {total_lines} lines{file_hint}]"
            return f"{stub_header}{stub_doc}{stub_body}"

        else:
            # JS, TS, C#, Java, Go, C++
            header_lines = []
            for line in raw_lines:
                header_lines.append(line)
                if "{" in line:
                    break
                if len(header_lines) > 10:
                    break

            stub_header = "\n".join(header_lines)
            indent = "    "
            stub_body = f"\n{indent}// ... [Implementation omitted: {total_lines} lines{file_hint}]\n{indent}return null;\n}}"
            return f"{stub_header}{stub_body}"


# Singleton export for system-wide consumption
code_parser = CodeParserService()
