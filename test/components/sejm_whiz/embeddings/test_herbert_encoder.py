"""Tests for HerBERT encoder functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sejm_whiz.embeddings.herbert_encoder import HerBERTEncoder, get_herbert_encoder
from sejm_whiz.embeddings.config import EmbeddingConfig


class TestHerBERTEncoder:
    """Test HerBERT encoder functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock embedding configuration."""
        return EmbeddingConfig(
            model_name="allegro/herbert-klej-cased-v1",
            embedding_dim=768,
            batch_size=2,
            max_length=512,
            device="cpu",
        )

    @pytest.fixture
    def mock_encoder(self, mock_config):
        """Mock HerBERT encoder with patched embedder."""
        with patch(
            "sejm_whiz.embeddings.herbert_encoder.get_herbert_embedder"
        ) as mock_embedder:
            # Mock the embedder methods
            mock_embedder_instance = Mock()
            mock_embedder_instance._model_loaded = True
            mock_embedder_instance.embed_texts.return_value = [
                Mock(embedding=np.random.randn(768)) for _ in range(2)
            ]
            mock_embedder_instance.embed_legal_document.return_value = Mock(
                embedding=np.random.randn(768)
            )
            mock_embedder.return_value = mock_embedder_instance

            encoder = HerBERTEncoder(mock_config)
            return encoder

    def test_initialization(self, mock_config):
        """Test encoder initialization."""
        with patch("sejm_whiz.embeddings.herbert_encoder.get_herbert_embedder"):
            encoder = HerBERTEncoder(mock_config)
            assert encoder.config == mock_config
            assert encoder.embedder is not None

    def test_encode_single_text(self, mock_encoder):
        """Test encoding single text."""
        text = "Ustawa z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych."

        result = mock_encoder.encode(text)

        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        mock_encoder.embedder.embed_texts.assert_called_once_with([text])

    def test_encode_multiple_texts(self, mock_encoder):
        """Test encoding multiple texts."""
        texts = [
            "Ustawa z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych.",
            "Rozporządzenie Ministra Finansów z dnia 10 września 2019 r.",
        ]

        results = mock_encoder.encode(texts)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
        mock_encoder.embedder.embed_texts.assert_called_once_with(texts)

    def test_encode_with_metadata(self, mock_encoder):
        """Test encoding with full metadata."""
        text = "Art. 1. Ustawa reguluje zasady..."

        result = mock_encoder.encode_with_metadata(text)

        assert hasattr(result, "embedding")
        mock_encoder.embedder.embed_texts.assert_called_once_with([text])

    def test_encode_legal_document(self, mock_encoder):
        """Test legal document encoding."""
        title = "Ustawa o podatku dochodowym"
        content = "Art. 1. Ustawa reguluje zasady opodatkowania..."
        document_type = "act"

        result = mock_encoder.encode_legal_document(title, content, document_type)

        assert hasattr(result, "embedding")
        mock_encoder.embedder.embed_legal_document.assert_called_once_with(
            title, content, document_type
        )

    def test_get_embedding_dimension(self, mock_encoder):
        """Test getting embedding dimension."""
        dimension = mock_encoder.get_embedding_dimension()
        assert dimension == 768

    def test_get_model_name(self, mock_encoder):
        """Test getting model name."""
        model_name = mock_encoder.get_model_name()
        assert model_name == "allegro/herbert-klej-cased-v1"

    def test_is_ready(self, mock_encoder):
        """Test checking if encoder is ready."""
        assert mock_encoder.is_ready() == True

    def test_preload(self, mock_encoder):
        """Test model preloading."""
        mock_encoder.embedder._model_loaded = False
        mock_encoder.embedder._initialize_model = Mock()

        mock_encoder.preload()

        mock_encoder.embedder._initialize_model.assert_called_once()

    def test_cleanup(self, mock_encoder):
        """Test resource cleanup."""
        mock_encoder.embedder.cleanup = Mock()

        mock_encoder.cleanup()

        mock_encoder.embedder.cleanup.assert_called_once()


def test_get_herbert_encoder():
    """Test global encoder instance."""
    with (
        patch("sejm_whiz.embeddings.herbert_encoder.HerBERTEncoder") as mock_class,
        patch("sejm_whiz.embeddings.herbert_encoder._herbert_encoder", None),
    ):
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        encoder1 = get_herbert_encoder()
        encoder2 = get_herbert_encoder()

        # Should return same instance (singleton pattern)
        assert encoder1 == encoder2
        mock_class.assert_called_once()
