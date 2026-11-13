import os
from pathlib import Path
from dotenv import load_dotenv

# Base backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class Config:
    BASE_DIR = BASE_DIR
    APP_NAME = os.getenv("APP_NAME", "DevKnow - Multi-Language Developer Knowledge Assistant")
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = APP_ENV == "development"

    # Server Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 5000))
    API_PREFIX = os.getenv("API_PREFIX", "/api")
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    # Database Settings (Single-line Connection String URI)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:@localhost:3306/devknow_db?charset=utf8mb4"
    ).strip()

    # Legacy individual field fallbacks (if DATABASE_URL is not set)
    DB_PROVIDER = os.getenv("DB_PROVIDER", "MYSQL").upper()
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "devknow_db")

    # Storage Paths
    UPLOAD_PATH = BASE_DIR / os.getenv("UPLOAD_PATH", "data/uploads")
    CHROMA_PATH = BASE_DIR / os.getenv("CHROMA_PATH", "data/chroma_db")
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # LLM Provider Configuration
    # Active Provider Options: "gemini" | "openai" | "claude" | "azure_openai" | "custom"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODE = os.getenv("LLM_MODE", "STRICT").upper()
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1 if LLM_MODE == "STRICT" else 0.7))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.95))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 8192))

    # 1. Google Gemini Settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    # 2. OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # 3. Anthropic Claude Settings
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # 4. Azure OpenAI Settings
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # 5. Local / Custom LLM (Ollama, vLLM, Mistral)
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small:24b")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 300))

    # Embedding Settings & Modes
    # EMBEDDING_PROVIDER options: "LOCAL" (SentenceTransformers on CPU, 100% free & offline)
    #                             "GEMINI" (Google Gemini text-embedding-004, 768-dim cloud API)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "LOCAL").upper()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    # RAG Settings
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.50))
    CONVERSATION_MEMORY_TURNS = int(os.getenv("CONVERSATION_MEMORY_TURNS", 10))
    AUTO_SAVE_LLM_ANSWERS = os.getenv("AUTO_SAVE_LLM_ANSWERS", "false").lower() == "true"

    # Configurable Excel Column Mapping
    EXCEL_COLUMNS = {
        "rule_version_name": os.getenv("COL_RULE_NAME", "RuleVersionName"),
        "rule_version_description": os.getenv("COL_RULE_DESC", "RuleVersionDescription"),
        "script_header": os.getenv("COL_SCRIPT_HEADER", "Rule Version Script Header/Name"),
        "script_body": os.getenv("COL_SCRIPT_BODY", "Rule Version Script Body"),
        "language": os.getenv("COL_LANGUAGE", "Programming Language"),
        "framework": os.getenv("COL_FRAMEWORK", "Framework"),
        "version": os.getenv("COL_VERSION", "Version"),
        "class_name": os.getenv("COL_CLASS", "Class Name"),
        "namespace": os.getenv("COL_NAMESPACE", "Namespace"),
        "description": os.getenv("COL_DESCRIPTION", "Description"),
        "metadata": os.getenv("COL_METADATA", "Additional Metadata")
    }

    @classmethod
    def get_database_uri(cls):
        """Constructs or returns SQLAlchemy Database URI"""
        if cls.DATABASE_URL:
            # Normalize standard mysql:// to mysql+pymysql:// for PyMySQL driver
            if cls.DATABASE_URL.startswith("mysql://"):
                return cls.DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
            return cls.DATABASE_URL

        if cls.DB_PROVIDER == "MYSQL":
            password_part = f":{cls.DB_PASSWORD}" if cls.DB_PASSWORD else ""
            return f"mysql+pymysql://{cls.DB_USER}{password_part}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}?charset=utf8mb4"

        # Fallback to local SQLite database in data directory
        sqlite_dir = cls.BASE_DIR / "data"
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_dir / 'devknow.db'}"

    @classmethod
    def ensure_directories(cls):
        """Ensures required data and upload directories exist"""
        cls.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PATH.mkdir(parents=True, exist_ok=True)

