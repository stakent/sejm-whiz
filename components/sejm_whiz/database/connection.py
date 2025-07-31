"""Database connection management for sejm-whiz."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .config import get_database_config, create_database_engine, create_session_factory
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, config=None):
        """Initialize database manager."""
        self.config = config or get_database_config()
        self.engine = create_database_engine(self.config)
        self.SessionLocal = create_session_factory(self.engine)

    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def test_pgvector_extension(self) -> bool:
        """Test if pgvector extension is available."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                )
                return result.scalar() is True
        except SQLAlchemyError as e:
            logger.error(f"pgvector extension test failed: {e}")
            return False

    def get_vector_dimensions(self) -> int:
        """Get configured vector dimensions."""
        return self.config.vector_dimensions

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_session_direct(self) -> Session:
        """Get database session (manual management)."""
        return self.SessionLocal()

    def close(self):
        """Close database engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """Initialize database with tables and extensions."""
    db_manager = get_database_manager()

    # Test connection
    if not db_manager.test_connection():
        raise RuntimeError("Cannot connect to database")

    # Test pgvector extension
    if not db_manager.test_pgvector_extension():
        logger.warning("pgvector extension not found - vector operations may fail")

    # Create tables
    db_manager.create_tables()

    logger.info("Database initialization completed")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Convenience function for getting database session."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


def execute_sql(query: str, params: dict = None) -> any:
    """Execute raw SQL query."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        result = session.execute(text(query), params or {})
        if query.strip().upper().startswith("SELECT"):
            return result.fetchall()
        return result.rowcount


def check_database_health() -> dict:
    """Check database health and return status."""
    db_manager = get_database_manager()

    health_status = {
        "connection": False,
        "pgvector": False,
        "tables_exist": False,
        "vector_dimensions": db_manager.config.vector_dimensions,
        "config": {
            "host": db_manager.config.host,
            "database": db_manager.config.database,
            "ssl_mode": db_manager.config.ssl_mode,
        },
    }

    try:
        # Test connection
        health_status["connection"] = db_manager.test_connection()

        # Test pgvector
        health_status["pgvector"] = db_manager.test_pgvector_extension()

        # Check if main tables exist
        with db_manager.get_session() as session:
            result = session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'legal_documents')"
                )
            )
            health_status["tables_exist"] = result.scalar()

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["error"] = str(e)

    return health_status
