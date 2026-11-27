"""
DatabaseService: Relational Database Access Layer for DevKnow.
Provides a thread-safe SQLAlchemy engine, automatic schema creation,
transparent MySQL-to-SQLite fallback for high offline resilience,
and a transactional context manager for unit-of-work sessions.
"""
import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker, scoped_session, Session
import pymysql

from app.config import Config
from app.models.database_models import Base

logger = logging.getLogger("DevKnow.Database")

class DatabaseService:
    """
    Singleton Database Manager:
    - Maintains the persistent SQLAlchemy connection pool.
    - Automatically checks and initializes the target database schema.
    - Seamlessly falls back to local SQLite if the remote MySQL server is unreachable.
    """
    _instance = None
    _engine = None
    _session_factory = None

    @classmethod
    def get_instance(cls) -> "DatabaseService":
        """Returns or creates the shared Singleton instance of DatabaseService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initializes database connection pool and creates all relational tables."""
        self._init_database()

    def _ensure_mysql_database_exists(self, db_uri: str) -> bool:
        """
        Connects to the MySQL host using PyMySQL to verify that the target database exists.
        If it does not exist, executes CREATE DATABASE with utf8mb4 encoding.
        
        Args:
            db_uri: The full SQLAlchemy connection string URI.
            
        Returns:
            bool: True if MySQL server is reached and database is ready; False on connection failure.
        """
        try:
            url = make_url(db_uri)
            if "mysql" in url.drivername and url.database:
                conn = pymysql.connect(
                    host=url.host or "localhost",
                    port=url.port or 3306,
                    user=url.username or "root",
                    password=url.password or "",
                    charset="utf8mb4",
                    connect_timeout=5
                )
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
                        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                    )
                conn.commit()
                conn.close()
                logger.info(f"MySQL database '{url.database}' verified/created successfully.")
                return True
        except Exception as e:
            logger.warning(f"Could not connect directly to MySQL server: {e}. Will attempt fallback.")
            return False
        return True

    def _init_database(self) -> None:
        """
        Initializes the SQLAlchemy engine:
        1. Checks MySQL connection via DATABASE_URL.
        2. Configures pool_pre_ping=True to automatically reconnect stale connections.
        3. If MySQL fails, gracefully activates local SQLite (data/devknow.db) to prevent crashes.
        4. Invokes Base.metadata.create_all() to ensure all tables exist.
        """
        Config.ensure_directories()
        db_uri = Config.get_database_uri()

        mysql_ready = False
        if "mysql" in db_uri:
            mysql_ready = self._ensure_mysql_database_exists(db_uri)

        if mysql_ready:
            try:
                self._engine = create_engine(
                    db_uri,
                    pool_pre_ping=True,  # Proactively validates socket before returning from pool
                    pool_recycle=3600,   # Recycles idle connections after 1 hour
                    echo=False
                )
                # Test connection with a lightweight ping query
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to MySQL database via DATABASE_URL successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to MySQL via DATABASE_URL: {e}. Falling back to SQLite for local reliability.")
                sqlite_path = Config.BASE_DIR / "data" / "devknow.db"
                self._engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        else:
            logger.info("Using SQLite database provider.")
            sqlite_path = Config.BASE_DIR / "data" / "devknow.db"
            self._engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)

        # Create all tables (functions, files, conversations, messages, logs) if not present
        Base.metadata.create_all(self._engine)
        self._ensure_columns_exist()
        self._session_factory = scoped_session(sessionmaker(bind=self._engine, autoflush=False, autocommit=False))
        logger.info("Database tables initialized successfully.")

    def _ensure_columns_exist(self) -> None:
        """Adds missing columns (e.g. risk_level, risk_explanation) to existing tables if needed."""
        try:
            with self._engine.connect() as conn:
                # Check functions table
                is_sqlite = "sqlite" in str(self._engine.url)
                if is_sqlite:
                    cols = [row[1] for row in conn.execute(text("PRAGMA table_info(functions)")).fetchall()]
                    if "risk_level" not in cols:
                        conn.execute(text("ALTER TABLE functions ADD COLUMN risk_level VARCHAR(50) DEFAULT 'low'"))
                    if "risk_explanation" not in cols:
                        conn.execute(text("ALTER TABLE functions ADD COLUMN risk_explanation TEXT DEFAULT ''"))
                    conn.commit()
                else:
                    # MySQL
                    cols = [row[0] for row in conn.execute(text("SHOW COLUMNS FROM functions")).fetchall()]
                    if "risk_level" not in cols:
                        conn.execute(text("ALTER TABLE functions ADD COLUMN risk_level VARCHAR(50) DEFAULT 'low'"))
                    if "risk_explanation" not in cols:
                        conn.execute(text("ALTER TABLE functions ADD COLUMN risk_explanation TEXT"))
                    conn.commit()
        except Exception as e:
            logger.debug(f"Schema column check notice: {e}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager providing transactional scope around database operations.
        
        Usage:
            with db_service.get_session() as session:
                session.add(record)
                # Auto-commits on clean exit
                
        Features:
            - Auto-commits transaction on block completion.
            - Auto-rollbacks transaction on unhandled exception.
            - Guarantees session.close() in finally block to prevent connection leaks.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# Export shared Singleton instance for import across controllers and services
db_service = DatabaseService.get_instance()

