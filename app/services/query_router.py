import re
from typing import Dict, Any, Optional

INTENT_PATTERNS = {
    "code_conversion": [
        r"\bconvert\b.*?\bto\b", r"\bport\b.*?\bto\b", r"\brewrite\b.*?\bin\b",
        r"\btranslate\b.*?\bto\b", r"\bhow\s+to\s+write\s+this\s+in\b"
    ],
    "code_improvement": [
        r"\bimprove\b", r"\boptimize\b", r"\brefactor\b", r"\bclean\s+up\b",
        r"\bperformance\b", r"\bmake\s+faster\b", r"\bbetter\s+way\b",
        r"\bbest\s+practice\b", r"\bsecurity\s+issue\b"
    ],
    "code_analysis": [
        r"\banalyze\b", r"\bcode\s+review\b", r"\bbugs?\b", r"\bvulnerabilit(y|ies)\b",
        r"\bcomplexity\b", r"\bpotential\s+issues\b", r"\bcheck\s+this\s+code\b"
    ],
    "cross_language": [
        r"\bacross\s+languages\b", r"\bequivalent\s+in\b", r"\balternative\s+in\b",
        r"\bother\s+languages?\b", r"\bdifferent\s+languages?\b"
    ],
    "version_lookup": [
        r"\blatest\s+version\b", r"\bcompare\s+versions?\b", r"\bversion\s+difference\b",
        r"\bwhich\s+version\b", r"\bv\d+(\.\d+)*\b"
    ],
    "function_explanation": [
        r"\bexplain\b", r"\bhow\s+does\b.*?\bwork\b", r"\bwhat\s+does\b.*?\bdo\b",
        r"\bunderstand\b", r"\bwalkthrough\b", r"\bmeaning\s+of\b"
    ],
    "code_similarity": [
        r"\bsimilar\s+functions?\b", r"\bduplicates?\b", r"\balready\s+exists?\b",
        r"\breusable\b", r"\bsomething\s+like\s+this\b", r"\balready\s+have\b"
    ],
    "function_lookup": [
        r"\bfind\b", r"\bsearch\b", r"\bshow\s+me\b", r"\bget\b",
        r"\bis\s+there\s+a\s+function\b", r"\bwhere\s+is\b", r"\blist\b"
    ],
    "greeting": [
        r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening))\b",
        r"\b(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+devknow)\b",
        r"^(thanks|thank\s+you)\b"
    ]
}

KNOWN_LANGUAGES = [
    "python", "java", "javascript", "typescript", "c#", "csharp", "c++", "cpp",
    "c", "go", "golang", "php", "ruby", "kotlin", "swift", "sql", "rust", "scala"
]

class QueryRouter:
    @classmethod
    def route_query(cls, query: str, user_selected_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies user query intent, extracts referenced languages,
        and provides routing directives for the RAG pipeline.
        """
        clean_query = query.strip()
        lower_query = clean_query.lower()

        # 1. Detect Intent
        detected_intent = "general_question"
        for intent, patterns in INTENT_PATTERNS.items():
            if any(re.search(pat, lower_query) for pat in patterns):
                detected_intent = intent
                break

        # 2. Extract Source and Target Programming Languages
        extracted_languages = []
        for lang in KNOWN_LANGUAGES:
            # Word boundary matching
            escaped = re.escape(lang)
            if re.search(rf"\b{escaped}\b", lower_query):
                std_lang = "c#" if lang in ["csharp", "c#"] else ("go" if lang == "golang" else ("c++" if lang == "cpp" else lang))
                if std_lang not in extracted_languages:
                    extracted_languages.append(std_lang)

        # Detect conversion target (e.g. "convert to Java", "in Python")
        target_language = None
        source_language = None

        if detected_intent == "code_conversion":
            conv_match = re.search(r"\b(?:to|into)\s+([a-zA-Z#+]+)", lower_query)
            if conv_match:
                cand = conv_match.group(1).lower()
                for lang in KNOWN_LANGUAGES:
                    if cand == lang:
                        target_language = "c#" if lang in ["csharp", "c#"] else ("go" if lang == "golang" else ("c++" if lang == "cpp" else lang))
                        break

        # Determine effective filter language
        filter_language = None
        if user_selected_language and user_selected_language.lower() != "all":
            filter_language = user_selected_language.lower()
        elif extracted_languages:
            filter_language = extracted_languages[0]

        return {
            "intent": detected_intent,
            "filter_language": filter_language,
            "target_language": target_language,
            "extracted_languages": extracted_languages,
            "requires_mysql_body": detected_intent in [
                "function_explanation", "code_improvement", "code_conversion",
                "code_analysis", "function_lookup", "code_similarity"
            ]
        }

