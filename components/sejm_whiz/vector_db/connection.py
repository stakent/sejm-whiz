"""Vector database connection management."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session

from sejm_whiz.database.connection import get_database_manager

logger = logging.getLogger(__name__)


class VectorDBConnection:
    """Vector database connection wrapper for pgvector operations."""

    def __init__(self):
        """Initialize vector database connection."""
        self.db_manager = get_database_manager()

    def test_vector_extension(self) -> bool:
        """Test if pgvector extension is available."""
        return self.db_manager.test_pgvector_extension()

    def get_vector_dimensions(self) -> int:
        """Get configured vector dimensions."""
        return self.db_manager.get_vector_dimensions()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session for vector operations."""
        with self.db_manager.get_session() as session:
            yield session

    def test_connection(self) -> bool:
        """Test database connection."""
        return self.db_manager.test_connection()


# Global vector database connection instance
_vector_db: VectorDBConnection = None


def get_vector_db() -> VectorDBConnection:
    """Get or create global vector database connection."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDBConnection()
    return _vector_db


@contextmanager
def get_vector_session() -> Generator[Session, None, None]:
    """Convenience function for getting vector database session."""
    vector_db = get_vector_db()
    with vector_db.get_session() as session:
        yield session
