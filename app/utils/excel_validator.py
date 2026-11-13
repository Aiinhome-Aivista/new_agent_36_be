import os
import pandas as pd
from typing import Dict, List, Tuple, Any
from app.config import Config

class ExcelValidator:
    @staticmethod
    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ["xlsx", "xls"]

    @staticmethod
    def resolve_column_mappings(df_columns: List[str]) -> Dict[str, str]:
        """
        Dynamically maps DataFrame columns to internal standard field names.
        First tests exact config mappings, then fuzzy matches common variations.
        """
        clean_cols = {str(col).strip().lower(): col for col in df_columns}
        cfg = Config.EXCEL_COLUMNS
        resolved = {}

        field_patterns = {
            "rule_version_name": [
                cfg["rule_version_name"].lower(), "ruleversionname", "rule name", "module", "category"
            ],
            "rule_version_description": [
                cfg["rule_version_description"].lower(), "ruleversiondescription", "rule description", "context"
            ],
            "script_header": [
                cfg["script_header"].lower(), "rule version script header/name", "script header", "function header",
                "function signature", "function name", "header/name", "header"
            ],
            "script_body": [
                cfg["script_body"].lower(), "rule version script body", "script body", "function body", "body", "code"
            ],
            "language": [
                cfg["language"].lower(), "programming language", "language", "lang", "tech", "technology"
            ],
            "framework": [
                cfg["framework"].lower(), "framework", "library", "stack"
            ],
            "version": [
                cfg["version"].lower(), "version", "ver", "revision"
            ],
            "class_name": [
                cfg["class_name"].lower(), "class name", "classname", "class"
            ],
            "namespace": [
                cfg["namespace"].lower(), "namespace", "package", "namespace / package"
            ],
            "description": [
                cfg["description"].lower(), "description", "details", "comments", "summary"
            ],
            "metadata": [
                cfg["metadata"].lower(), "additional metadata", "metadata", "tags", "notes"
            ]
        }

        for internal_key, patterns in field_patterns.items():
            matched_col = None
            for p in patterns:
                if p in clean_cols:
                    matched_col = clean_cols[p]
                    break
            if not matched_col:
                # Substring check
                for col_clean, original_col in clean_cols.items():
                    for p in patterns:
                        if p in col_clean or col_clean in p:
                            matched_col = original_col
                            break
                    if matched_col:
                        break
            resolved[internal_key] = matched_col

        return resolved

    @classmethod
    def validate_excel_structure(cls, file_path: str) -> Tuple[bool, str, Dict[str, str], List[str]]:
        """
        Validates file existence, format, and mandatory columns.
        Returns: (is_valid, error_message, column_mapping, detected_languages)
        """
        if not os.path.exists(file_path):
            return False, f"File does not exist at {file_path}", {}, []

        try:
            # Read first few rows to check columns
            df_preview = pd.read_excel(file_path, nrows=5)
        except Exception as e:
            return False, f"Unable to read Excel file: {str(e)}", {}, []

        columns = [str(c) for c in df_preview.columns]
        mapping = cls.resolve_column_mappings(columns)

        missing = []
        if not mapping.get("script_header"):
            missing.append("Function/Script Header (e.g. 'Rule Version Script Header/Name' or 'Function Header')")
        if not mapping.get("script_body"):
            missing.append("Function/Script Body (e.g. 'Rule Version Script Body' or 'Function Body')")

        if missing:
            return False, f"Excel file is missing required columns: {', '.join(missing)}", mapping, []

        # Check language column
        lang_col = mapping.get("language")
        detected_languages = []
        if lang_col and lang_col in df_preview.columns:
            try:
                full_df = pd.read_excel(file_path, usecols=[lang_col])
                unique_langs = full_df[lang_col].dropna().astype(str).str.strip().unique().tolist()
                detected_languages = [l for l in unique_langs if l and l.lower() != "nan"]
            except Exception:
                pass

        return True, "", mapping, detected_languages

