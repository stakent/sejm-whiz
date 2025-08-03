"""Integration tests for embeddings component core functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sejm_whiz.embeddings import (
    EmbeddingConfig,
    get_embedding_config,
    HerBERTEncoder,
    get_herbert_encoder,
    BagEmbeddingsGenerator,
    get_bag_embeddings_generator,
    SimilarityCalculator,
    get_similarity_calculator,
    BatchProcessor,
    get_batch_processor,
    cosine_similarity,
    euclidean_distance,
)


def test_all_imports_available():
    """Test that all expected components are importable."""
    # Configuration
    assert EmbeddingConfig is not None
    assert get_embedding_config is not None

    # HerBERT encoder
    assert HerBERTEncoder is not None
    assert get_herbert_encoder is not None

    # Bag embeddings
    assert BagEmbeddingsGenerator is not None
    assert get_bag_embeddings_generator is not None

    # Similarity
    assert SimilarityCalculator is not None
    assert get_similarity_calculator is not None

    # Batch processing
    assert BatchProcessor is not None
    assert get_batch_processor is not None

    # Convenience functions
    assert cosine_similarity is not None
    assert euclidean_distance is not None


def test_embedding_config_creation():
    """Test embedding configuration creation."""
    config = EmbeddingConfig()

    assert config.model_name == "allegro/herbert-base-cased"
    assert config.embedding_dim == 768
    assert config.pooling_strategy == "mean"
    assert config.batch_size > 0
    assert config.max_workers > 0


def test_embedding_config_for_gpu():
    """Test GPU-optimized configuration."""
    config = EmbeddingConfig.for_gpu()

    assert config.device == "cuda"
    assert config.use_half_precision
    assert config.compile_model
    assert config.batch_size >= 16


def test_embedding_config_for_cpu():
    """Test CPU-optimized configuration."""
    config = EmbeddingConfig.for_cpu()

    assert config.device == "cpu"
    assert not config.use_half_precision
    assert not config.compile_model


def test_embedding_config_for_production():
    """Test production configuration."""
    config = EmbeddingConfig.for_production()

    assert config.device == "auto"
    assert config.cache_embeddings
    assert config.compile_model
    assert config.similarity_threshold >= 0.8


@patch("sejm_whiz.embeddings.herbert_encoder.get_herbert_embedder")
def test_herbert_encoder_integration(mock_embedder):
    """Test HerBERT encoder integration."""
    # Mock the embedder
    mock_embedder_instance = Mock()
    mock_embedder_instance._model_loaded = True
    mock_embedder_instance.embed_texts.return_value = [
        Mock(embedding=np.random.randn(768))
    ]
    mock_embedder.return_value = mock_embedder_instance

    # Test encoder
    encoder = HerBERTEncoder()
    result = encoder.encode("Test text")

    assert isinstance(result, np.ndarray)
    assert len(result) == 768


@patch("sejm_whiz.embeddings.bag_embeddings.get_herbert_encoder")
def test_bag_embeddings_integration(mock_encoder):
    """Test bag embeddings integration."""
    # Mock encoder
    mock_encoder_instance = Mock()
    mock_encoder_instance.encode.return_value = [np.random.randn(768) for _ in range(3)]
    mock_encoder_instance.get_embedding_dimension.return_value = 768
    mock_encoder.return_value = mock_encoder_instance

    # Test bag generator
    generator = BagEmbeddingsGenerator()
    result = generator.generate_bag_embedding("Ustawa o podatku dochodowym")

    assert result.vocabulary_size > 0
    assert len(result.document_embedding) == 768
    assert len(result.token_embeddings) > 0


def test_similarity_calculator_integration():
    """Test similarity calculator integration."""
    calculator = SimilarityCalculator()

    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([1.0, 2.0, 3.0])

    result = calculator.cosine_similarity(embedding1, embedding2)

    assert abs(result.similarity_score - 1.0) < 1e-6
    assert result.method == "cosine"


def test_cosine_similarity_convenience():
    """Test cosine similarity convenience function."""
    embedding1 = np.array([1.0, 0.0, 0.0])
    embedding2 = np.array([0.0, 1.0, 0.0])

    similarity = cosine_similarity(embedding1, embedding2)

    assert isinstance(similarity, float)
    assert abs(similarity - 0.0) < 1e-6  # Orthogonal vectors


def test_euclidean_distance_convenience():
    """Test Euclidean distance convenience function."""
    embedding1 = np.array([0.0, 0.0, 0.0])
    embedding2 = np.array([3.0, 4.0, 0.0])

    distance = euclidean_distance(embedding1, embedding2)

    assert isinstance(distance, float)
    assert distance == 5.0  # 3-4-5 triangle


@patch("sejm_whiz.embeddings.batch_processor.get_herbert_encoder")
@patch("sejm_whiz.embeddings.batch_processor.get_bag_embeddings_generator")
@patch("sejm_whiz.embeddings.batch_processor.get_similarity_calculator")
def test_batch_processor_integration(mock_sim_calc, mock_bag_gen, mock_encoder):
    """Test batch processor integration."""
    # Mock dependencies
    mock_encoder_instance = Mock()
    mock_encoder.return_value = mock_encoder_instance

    mock_bag_gen_instance = Mock()
    mock_bag_gen.return_value = mock_bag_gen_instance

    mock_sim_calc_instance = Mock()
    mock_sim_calc.return_value = mock_sim_calc_instance

    # Test processor creation
    processor = BatchProcessor()

    assert processor.encoder is not None
    assert processor.bag_generator is not None
    assert processor.similarity_calculator is not None


def test_singleton_patterns():
    """Test that singleton patterns work correctly."""
    # Test that get functions return instances
    get_embedding_config()
    get_embedding_config()
    # Note: config doesn't use singleton pattern, so instances may differ

    with patch("sejm_whiz.embeddings.herbert_encoder.HerBERTEncoder"):
        encoder1 = get_herbert_encoder()
        encoder2 = get_herbert_encoder()
        assert encoder1 == encoder2

    with patch("sejm_whiz.embeddings.bag_embeddings.BagEmbeddingsGenerator"):
        generator1 = get_bag_embeddings_generator()
        generator2 = get_bag_embeddings_generator()
        assert generator1 == generator2

    calculator1 = get_similarity_calculator()
    calculator2 = get_similarity_calculator()
    assert calculator1 == calculator2

    with patch("sejm_whiz.embeddings.batch_processor.BatchProcessor"):
        processor1 = get_batch_processor()
        processor2 = get_batch_processor()
        assert processor1 == processor2


def test_component_configuration_compatibility():
    """Test that components work together with shared configuration."""
    config = EmbeddingConfig(
        embedding_dim=512,  # Non-default dimension
        batch_size=4,
        max_workers=2,
    )

    with patch("sejm_whiz.embeddings.herbert_encoder.get_herbert_embedder"):
        encoder = HerBERTEncoder(config)
        assert encoder.config.embedding_dim == 512
        assert encoder.config.batch_size == 4

    with patch("sejm_whiz.embeddings.bag_embeddings.get_herbert_encoder"):
        generator = BagEmbeddingsGenerator(config)
        assert generator.config.embedding_dim == 512
        assert generator.config.batch_size == 4

    with (
        patch("sejm_whiz.embeddings.batch_processor.get_herbert_encoder"),
        patch("sejm_whiz.embeddings.batch_processor.get_bag_embeddings_generator"),
        patch("sejm_whiz.embeddings.batch_processor.get_similarity_calculator"),
    ):
        processor = BatchProcessor(config)
        assert processor.config.embedding_dim == 512
        assert processor.config.batch_size == 4
        assert processor.max_workers == 2


@pytest.mark.integration
def test_end_to_end_workflow_mock():
    """Test end-to-end workflow with mocked dependencies."""
    with (
        patch(
            "sejm_whiz.embeddings.herbert_encoder.get_herbert_embedder"
        ) as mock_embedder,
        patch(
            "sejm_whiz.embeddings.bag_embeddings.get_herbert_encoder"
        ) as mock_bag_encoder,
    ):
        # Mock embedder for HerBERT encoder
        mock_embedder_instance = Mock()
        mock_embedder_instance._model_loaded = True
        mock_embedder_instance.embed_texts.return_value = [
            Mock(embedding=np.random.randn(768))
        ]
        mock_embedder.return_value = mock_embedder_instance

        # Mock encoder for bag embeddings
        mock_bag_encoder_instance = Mock()

        # Make the encode method return embeddings based on the number of input tokens
        def mock_encode(tokens):
            return [np.random.randn(768) for _ in range(len(tokens))]

        mock_bag_encoder_instance.encode.side_effect = mock_encode
        mock_bag_encoder_instance.get_embedding_dimension.return_value = 768
        mock_bag_encoder.return_value = mock_bag_encoder_instance

        # Step 1: Encode text with HerBERT
        encoder = HerBERTEncoder()
        text_embedding = encoder.encode("Ustawa o podatku dochodowym od osób prawnych")
        assert isinstance(text_embedding, np.ndarray)

        # Step 2: Generate bag of embeddings
        bag_generator = BagEmbeddingsGenerator()
        bag_result = bag_generator.generate_bag_embedding(
            "Ustawa o podatku dochodowym od osób prawnych",
            "Ustawa o podatku dochodowym",
        )
        assert bag_result.vocabulary_size > 0

        # Step 3: Calculate similarity
        calculator = SimilarityCalculator()
        similarity_result = calculator.cosine_similarity(
            text_embedding, bag_result.document_embedding
        )
        assert -1.0 <= similarity_result.similarity_score <= 1.0

        # Step 4: Use convenience function
        similarity_score = cosine_similarity(
            text_embedding, bag_result.document_embedding
        )
        assert isinstance(similarity_score, float)
