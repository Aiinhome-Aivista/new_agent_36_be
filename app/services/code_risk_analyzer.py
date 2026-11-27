"""
Code Risk Analyzer Module for DevKnow.

Evaluates extracted functions and code units for security, structural, and behavioral risks
before they are committed to the Knowledge Base.

Policy:
1. Clean, valid code is marked risk_level='low' and auto-ingested directly into ChromaDB.
2. Suspicious, corrupt, or dangerous code is quarantined with an elaborated Risk & Impact
   Explanation describing exactly WHY it was flagged and what negative impact it could cause.
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DevKnow.RiskAnalyzer")

# Secret & Credential Patterns
SECRET_PATTERNS = [
    # AWS Access Key
    (r"(?i)\b(A3T[A-Z0-9]|AKIA[0-9A-Z]{16})\b", "AWS Access Key ID"),
    # Generic API Keys / Secrets / Tokens assigned to strings with >= 16 chars
    (r"""(?i)(?:api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|client[_-]?secret|jwt[_-]?secret|private[_-]?key)\s*[:=]\s*["']([A-Za-z0-9_\-\.\=\+]{16,})["']""", "Hardcoded API or Secret Key"),
    # Hardcoded passwords (excluding trivial placeholder names like 'password', 'test', 'dummy')
    (r"""(?i)(?:db[_-]?password|database[_-]?password|admin[_-]?password|master[_-]?key)\s*[:=]\s*["']([^"'\s]{8,})["']""", "Hardcoded Database/Admin Password"),
    # Private Key Header
    (r"-----BEGIN (?:RSA|EC|OPENSSH|PGP|DSA) PRIVATE KEY-----", "Raw Cryptographic Private Key Block"),
    # Slack / Discord / GitHub token prefixes
    (r"(xox[baprs]-[0-9a-zA-Z]{10,48}|ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{22,})", "Platform Access Token"),
]

# Max acceptable cyclomatic complexity before quarantine for review
MAX_ALLOWED_COMPLEXITY = 30


class CodeRiskAnalyzer:
    """
    Evaluates individual code units for corruption, hardcoded secrets,
    severe complexity, and breaking signature drift.
    """

    @classmethod
    def analyze_function(
        cls,
        code_body: str,
        ast_meta: Dict[str, Any],
        existing_function: Optional[Any] = None,
        language: str = "general"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive security, grammar, and lineage checks on a code unit.

        Args:
            code_body: Verbatim function body string.
            ast_meta: Tree-sitter AST metadata dict (syntax validity, complexity, parameters, header).
            existing_function: Pre-existing FunctionRecord from MySQL if name collision occurs.
            language: Programming language.

        Returns:
            Dict containing:
                - is_quarantined (bool): True if human review is strictly required.
                - risk_level (str): 'low' | 'medium' | 'high' | 'critical'
                - flags (List[str]): Machine-readable tags for matched risk triggers.
                - risk_explanation (str): Human-readable paragraph explaining WHY and the NEGATIVE IMPACT.
        """
        flags: List[str] = []
        reasons: List[str] = []
        impacts: List[str] = []
        risk_level = "low"

        # ── 1. Check for Hardcoded Secrets & Credentials (CRITICAL RISK) ───
        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, code_body):
                flags.append("hardcoded_secret")
                risk_level = "critical"
                reasons.append(f"Security Alert: {label} detected in function body.")
                impacts.append(
                    "Negative Impact: If auto-indexed into the vector store, this sensitive credential "
                    "or private token could be surfaced in conversational RAG chat answers to unauthorized developers."
                )
                break  # One critical secret is sufficient to quarantine

        # ── 2. Check for Syntax / AST Grammar Errors (HIGH RISK) ───────────
        is_valid_syntax = ast_meta.get("is_valid_syntax", True)
        syntax_err = ast_meta.get("syntax_error")
        if not is_valid_syntax:
            flags.append("syntax_error")
            if risk_level not in ["critical"]:
                risk_level = "high"
            reasons.append(f"Syntax Alert: Tree-sitter AST identified invalid syntax ({syntax_err or 'unclosed token or grammar failure'}).")
            impacts.append(
                "Negative Impact: The knowledge base will be poisoned with un-compilable code snippets, "
                "leading the AI assistant to recommend broken code patterns to developers."
            )

        # ── 3. Check for Breaking Signature Drift (HIGH / MEDIUM RISK) ────
        if existing_function:
            old_header = existing_function.function_header or ""
            new_header = ast_meta.get("function_header", "")
            old_version = existing_function.version or "1.0.0"

            # Check if parameters were radically dropped or changed
            new_params = ast_meta.get("parameters", [])
            # Extract basic parameter count from old header
            old_params_match = re.findall(r"\((.*?)\)", old_header)
            old_param_count = len([p for p in old_params_match[0].split(",") if p.strip()]) if old_params_match else 0

            # If an existing function with >= 2 parameters suddenly drops to 0 or loses > 50%
            if old_param_count >= 2 and len(new_params) == 0:
                flags.append("breaking_signature_drift")
                if risk_level not in ["critical", "high"]:
                    risk_level = "high"
                reasons.append(
                    f"Signature Breaking Drift: Parameter list dropped from {old_param_count} parameters in v{old_version} "
                    f"to 0 parameters in proposed update."
                )
                impacts.append(
                    f"Negative Impact: Downstream developers relying on '{existing_function.function_name}' may be guided "
                    f"by RAG chat to call an incompatible signature, breaking existing production code."
                )

        # ── 4. Check for Extreme Cyclomatic Complexity (MEDIUM RISK) ───────
        complexity = ast_meta.get("complexity", 1)
        if complexity > MAX_ALLOWED_COMPLEXITY:
            flags.append("extreme_complexity")
            if risk_level == "low":
                risk_level = "medium"
            reasons.append(f"Complexity Warning: Cyclomatic complexity score is {complexity} (exceeds threshold of {MAX_ALLOWED_COMPLEXITY}).")
            impacts.append(
                "Negative Impact: Extremely high cyclomatic complexity indicates deeply nested spaghetti logic "
                "or un-refactored monolithic methods. Cementing it as standard organizational knowledge without review encourages technical debt."
            )

        # Assemble final structured response
        is_quarantined = len(flags) > 0

        if is_quarantined:
            formatted_explanation = (
                f"⚠️ QUARANTINED FOR HUMAN REVIEW ({risk_level.upper()} RISK)\n\n"
                f"• Why Flagged:\n  " + "\n  ".join(reasons) + "\n\n"
                f"• Potential Negative Impact:\n  " + "\n  ".join(impacts)
            )
        else:
            formatted_explanation = ""

        return {
            "is_quarantined": is_quarantined,
            "risk_level": risk_level,
            "flags": flags,
            "risk_explanation": formatted_explanation
        }


# Global Singleton
code_risk_analyzer = CodeRiskAnalyzer()
