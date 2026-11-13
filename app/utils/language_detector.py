import re
from typing import Optional

LANGUAGE_PATTERNS = {
    "c#": [
        r"\bbool\s+\w+\s*\(", r"\bvoid\s+\w+\s*\(", r"\bstring\s+\w+",
        r"List<\w+>", r"Dictionary<\w+,\s*\w+>", r"using\s+System",
        r"namespace\s+\w+", r"\.GetAttrValue\(", r"\.ATTREXISTS\("
    ],
    "python": [
        r"def\s+\w+\s*\(.*?\):", r"import\s+\w+", r"from\s+\w+\s+import",
        r"elif\s+", r":\s*$", r"self\.", r"print\("
    ],
    "java": [
        r"public\s+static\s+void\s+main", r"public\s+class\s+\w+",
        r"System\.out\.println", r"@Override", r"package\s+[\w\.]+;"
    ],
    "javascript": [
        r"const\s+\w+\s*=", r"let\s+\w+\s*=", r"function\s+\w+\s*\(",
        r"=>\s*{?", r"console\.log\(", r"document\.getElementById"
    ],
    "typescript": [
        r":\s*(string|number|boolean|any|void)\b", r"interface\s+\w+",
        r"type\s+\w+\s*=", r"export\s+default"
    ],
    "sql": [
        r"\bSELECT\b.*?\bFROM\b", r"\bINSERT\s+INTO\b", r"\bUPDATE\b.*?\bSET\b",
        r"\bCREATE\s+TABLE\b", r"\bALTER\s+TABLE\b"
    ],
    "go": [
        r"func\s+\w+\s*\(", r"package\s+main", r":=\s*", r"fmt\.Println"
    ],
    "c++": [
        r"#include\s*<iostream>", r"std::cout", r"std::vector"
    ],
    "c": [
        r"#include\s*<stdio\.h>", r"printf\("
    ],
    "php": [
        r"<\?php", r"\$\w+\s*="
    ]
}

def detect_language_from_code(code_str: str) -> Optional[str]:
    """Lightweight rule-based programming language detection"""
    if not code_str or not code_str.strip():
        return None

    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, code_str, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[lang] = score

    if scores:
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0]

    return None

