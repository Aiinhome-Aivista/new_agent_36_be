import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker, scoped_session
import pymysql

from app.config import Config
from app.models.database_models import Base

logger = logging.getLogger("DevKnow.Database")

class DatabaseService:
    _instance = None
    _engine = None
    _session_factory = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._init_database()

    def _ensure_mysql_database_exists(self, db_uri: str):
        """Attempts to create MySQL database if it doesn't exist yet from DATABASE_URL"""
        try:
            url = make_url(db_uri)
            if "mysql" in url.drivername and url.database:
                conn = pymysql.connect(
                    host=url.host or "localhost",
                    port=url.port or 3306,
                    user=url.username or "root",
                    password=url.password or "",
                    charset="utf8mb4"
                )
                with conn.cursor() as cursor:
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{url.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
                conn.commit()
                conn.close()
                logger.info(f"MySQL database '{url.database}' verified/created successfully.")
                return True
        except Exception as e:
            logger.warning(f"Could not connect directly to MySQL server: {e}. Will attempt fallback.")
            return False
        return True

    def _init_database(self):
        Config.ensure_directories()
        db_uri = Config.get_database_uri()

        mysql_ready = False
        if "mysql" in db_uri:
            mysql_ready = self._ensure_mysql_database_exists(db_uri)

        if mysql_ready:
            try:
                self._engine = create_engine(
                    db_uri,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
                # Test connection
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

        # Create all tables if they don't exist
        Base.metadata.create_all(self._engine)
        self._session_factory = scoped_session(sessionmaker(bind=self._engine, autoflush=False, autocommit=False))
        logger.info("Database tables initialized successfully.")

    @contextmanager
    def get_session(self):
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

db_service = DatabaseService.get_instance()

