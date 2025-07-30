"""Tests for vector_db connection module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sejm_whiz.vector_db.connection import (
    VectorDBConnection,
    get_vector_db,
    get_vector_session,
)


class TestVectorDBConnection:
    """Test VectorDBConnection class."""

    @patch("sejm_whiz.vector_db.connection.get_database_manager")
    def test_init(self, mock_get_db_manager):
        """Test VectorDBConnection initialization."""
        mock_db_manager = Mock()
        mock_get_db_manager.return_value = mock_db_manager

        connection = VectorDBConnection()

        assert connection.db_manager == mock_db_manager
        mock_get_db_manager.assert_called_once()

    @patch("sejm_whiz.vector_db.connection.get_database_manager")
    def test_test_vector_extension(self, mock_get_db_manager):
        """Test vector extension check."""
        mock_db_manager = Mock()
        mock_db_manager.test_pgvector_extension.return_value = True
        mock_get_db_manager.return_value = mock_db_manager

        connection = VectorDBConnection()
        result = connection.test_vector_extension()

        assert result is True
        mock_db_manager.test_pgvector_extension.assert_called_once()

    @patch("sejm_whiz.vector_db.connection.get_database_manager")
    def test_get_vector_dimensions(self, mock_get_db_manager):
        """Test getting vector dimensions."""
        mock_db_manager = Mock()
        mock_db_manager.get_vector_dimensions.return_value = 768
        mock_get_db_manager.return_value = mock_db_manager

        connection = VectorDBConnection()
        result = connection.get_vector_dimensions()

        assert result == 768
        mock_db_manager.get_vector_dimensions.assert_called_once()

    @patch("sejm_whiz.vector_db.connection.get_database_manager")
    def test_get_session(self, mock_get_db_manager):
        """Test getting database session."""
        mock_db_manager = Mock()
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__ = Mock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = Mock(return_value=None)
        mock_get_db_manager.return_value = mock_db_manager

        connection = VectorDBConnection()

        with connection.get_session() as session:
            assert session == mock_session

        mock_db_manager.get_session.assert_called_once()

    @patch("sejm_whiz.vector_db.connection.get_database_manager")
    def test_test_connection(self, mock_get_db_manager):
        """Test database connection test."""
        mock_db_manager = Mock()
        mock_db_manager.test_connection.return_value = True
        mock_get_db_manager.return_value = mock_db_manager

        connection = VectorDBConnection()
        result = connection.test_connection()

        assert result is True
        mock_db_manager.test_connection.assert_called_once()


class TestGlobalFunctions:
    """Test global helper functions."""

    @patch("sejm_whiz.vector_db.connection.VectorDBConnection")
    def test_get_vector_db_singleton(self, mock_vector_db_class):
        """Test get_vector_db returns singleton."""
        mock_instance = Mock()
        mock_vector_db_class.return_value = mock_instance

        # Clear the global instance
        import sejm_whiz.vector_db.connection

        sejm_whiz.vector_db.connection._vector_db = None

        # First call should create instance
        result1 = get_vector_db()
        assert result1 == mock_instance
        mock_vector_db_class.assert_called_once()

        # Second call should return same instance
        result2 = get_vector_db()
        assert result2 == mock_instance
        assert result1 is result2
        # Should not create another instance
        mock_vector_db_class.assert_called_once()

    @patch("sejm_whiz.vector_db.connection.get_vector_db")
    def test_get_vector_session(self, mock_get_vector_db):
        """Test get_vector_session context manager."""
        mock_vector_db = Mock()
        mock_session = Mock()
        mock_vector_db.get_session.return_value.__enter__ = Mock(
            return_value=mock_session
        )
        mock_vector_db.get_session.return_value.__exit__ = Mock(return_value=None)
        mock_get_vector_db.return_value = mock_vector_db

        with get_vector_session() as session:
            assert session == mock_session

        mock_get_vector_db.assert_called_once()
        mock_vector_db.get_session.assert_called_once()
