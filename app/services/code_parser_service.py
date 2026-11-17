"""
CodeParserService: AST and Tree-sitter Code Intelligence Engine for DevKnow.
Provides deterministic syntax validation, function signature extraction,
cyclomatic complexity calculation, and code quality smell detection across
C#, TypeScript, JavaScript, and Python.
"""
import ast
import re
import logging
import warnings
from typing import Dict, Any, List, Optional

# Suppress deprecation warnings from C-extensions / tree_sitter legacy bindings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

logger = logging.getLogger("DevKnow.CodeParser")

# Attempt lazy import of tree_sitter_languages
_PARSERS: Dict[str, Any] = {}
_TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter_languages import get_parser  # type: ignore[import-untyped, import-not-found]
    # Pre-cache parsers
    for lang_key in ["c_sharp", "javascript", "typescript", "python"]:
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
    Combines native Python AST with Tree-sitter for multi-language intelligence.
    """

    @classmethod
    def normalize_language(cls, lang: Optional[str]) -> str:
        """Normalizes language string to standardized keys"""
        if not lang:
            return "c_sharp"
        l = lang.strip().lower()
        if l in ("c#", "csharp", "cs"):
            return "c_sharp"
        if l in ("js", "javascript"):
            return "javascript"
        if l in ("ts", "typescript"):
            return "typescript"
        if l in ("py", "python"):
            return "python"
        return l

    @classmethod
    def parse_function_metadata(cls, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts function name, parameters, return type, and async status
        using Tree-sitter or Python AST, with graceful regex fallback.
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

        # 1. Native Python AST for Python code
        if norm_lang == "python":
            py_meta = cls._parse_python_ast(code)
            if py_meta and py_meta.get("function_name"):
                return py_meta

        # 2. Tree-sitter for C#, TypeScript, JavaScript, Python
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            ts_meta = cls._parse_tree_sitter(code, norm_lang)
            if ts_meta and ts_meta.get("function_name"):
                return ts_meta

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
        """Parses code using Tree-sitter across C#, JS, TS, and Python"""
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

            # Walk AST to find function/method declaration
            def find_func_node(node):
                nonlocal target_node
                if target_node:
                    return
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
                # Name
                name_node = target_node.child_by_field_name("name")
                if name_node:
                    fn_name = name_node.text.decode("utf-8")

                # Return Type
                type_node = target_node.child_by_field_name("type") or target_node.child_by_field_name("return_type")
                if type_node:
                    ret_type = type_node.text.decode("utf-8").lstrip(": ").strip()

                # Parameters
                params_node = target_node.child_by_field_name("parameters")
                if params_node:
                    for p in params_node.children:
                        if p.type in ("parameter", "identifier", "required_parameter", "optional_parameter"):
                            p_text = p.text.decode("utf-8").strip()
                            if p_text and p_text not in (",", "(", ")"):
                                params.append(p_text)

                # Async check
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

        return {
            "function_name": meta.get("function_name", "unnamed"),
            "language": norm_lang,
            "parameters": meta.get("parameters", []),
            "return_type": meta.get("return_type", ""),
            "complexity": complexity,
            "lines_of_code": loc,
            "is_valid_syntax": syntax["is_valid"],
            "syntax_error": syntax["error_message"],
            "code_smells": smells,
            "parser_engine": meta.get("parser_used", "none")
        }

    @classmethod
    def extract_all_functions(cls, code: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parses an entire source code file (e.g. .cs, .ts, .js, .py)
        and extracts every function/method as a distinct, atomic chunk.
        """
        if not code or not code.strip():
            return []

        norm_lang = cls.normalize_language(language)
        results = []

        # 1. Native Python AST for Python files
        if norm_lang == "python":
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
                            "language": "python"
                        })
                if results:
                    return results
            except Exception as pe:
                logger.debug(f"Python full-file AST notice: {pe}")

        # 2. Tree-sitter for C#, TypeScript, JavaScript, Python
        if _TREE_SITTER_AVAILABLE and norm_lang in _PARSERS:
            try:
                parser = _PARSERS[norm_lang]
                code_bytes = code.encode("utf-8")
                tree = parser.parse(code_bytes)
                lines = code.split("\n")

                func_types = (
                    "method_declaration",
                    "local_function_statement",
                    "function_declaration",
                    "function_expression",
                    "arrow_function",
                    "function_definition",
                    "method_definition"
                )

                found_nodes = []
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

                    params = []
                    params_node = node.child_by_field_name("parameters")
                    if params_node:
                        for p in params_node.children:
                            if p.type in ("parameter", "identifier", "required_parameter", "optional_parameter"):
                                p_text = p.text.decode("utf-8").strip()
                                if p_text and p_text not in (",", "(", ")"):
                                    params.append(p_text)

                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    fn_lines = lines[start_line - 1 : end_line]
                    fn_full = "\n".join(fn_lines)
                    header = lines[start_line - 1].strip()
                    body = "\n".join(lines[start_line : end_line]) if len(lines) > start_line else fn_full
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
                        "language": norm_lang
                    })

                if results:
                    return results
            except Exception as ts_err:
                logger.debug(f"Tree-sitter file extraction notice: {ts_err}")

        return results


# Singleton export
code_parser = CodeParserService()
