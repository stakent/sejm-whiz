"""Tests for bag of embeddings functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sejm_whiz.embeddings.bag_embeddings import (
    BagEmbeddingsGenerator, get_bag_embeddings_generator,
    BagEmbeddingResult
)
from sejm_whiz.embeddings.config import EmbeddingConfig


class TestBagEmbeddingsGenerator:
    """Test bag of embeddings generator."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock embedding configuration."""
        return EmbeddingConfig(
            model_name="allegro/herbert-klej-cased-v1",
            embedding_dim=768,
            batch_size=2,
            max_length=512
        )
    
    @pytest.fixture
    def mock_generator(self, mock_config):
        """Mock bag embeddings generator."""
        with patch('sejm_whiz.embeddings.bag_embeddings.get_herbert_encoder') as mock_encoder:
            # Mock encoder
            mock_encoder_instance = Mock()
            
            # Dynamic mock that returns embeddings based on input length
            def mock_encode(tokens):
                return [np.random.randn(768) for _ in range(len(tokens))]
            
            mock_encoder_instance.encode.side_effect = mock_encode
            mock_encoder_instance.get_embedding_dimension.return_value = 768
            mock_encoder.return_value = mock_encoder_instance
            
            generator = BagEmbeddingsGenerator(mock_config)
            return generator
    
    def test_initialization(self, mock_config):
        """Test generator initialization."""
        with patch('sejm_whiz.embeddings.bag_embeddings.get_herbert_encoder'):
            generator = BagEmbeddingsGenerator(mock_config)
            assert generator.config == mock_config
            assert generator.encoder is not None
            assert len(generator.legal_term_patterns) > 0
            assert len(generator.term_weights) > 0
            assert len(generator.polish_stopwords) > 0
    
    def test_tokenize_text(self, mock_generator):
        """Test text tokenization."""
        text = "Ustawa z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych."
        
        tokens = mock_generator._tokenize_text(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        # Should contain meaningful legal terms
        assert any('ustawa' in token.lower() for token in tokens)
    
    def test_tokenize_empty_text(self, mock_generator):
        """Test tokenization of empty text."""
        tokens = mock_generator._tokenize_text("")
        assert tokens == []
        
        tokens = mock_generator._tokenize_text(None)
        assert tokens == []
    
    def test_is_legal_term(self, mock_generator):
        """Test legal term identification."""
        assert mock_generator._is_legal_term("ustawa") == True
        assert mock_generator._is_legal_term("artykuł") == True
        assert mock_generator._is_legal_term("konstytucja") == True
        assert mock_generator._is_legal_term("random_word") == False
    
    def test_calculate_token_weights_simple(self, mock_generator):
        """Test token weight calculation without TF-IDF or legal weighting."""
        tokens = ["ustawa", "reguluje", "zasady", "opodatkowania"]
        text = "Ustawa reguluje zasady opodatkowania."
        
        weights = mock_generator._calculate_token_weights(
            tokens, text, use_tf_idf=False, use_legal_weighting=False
        )
        
        assert len(weights) == len(tokens)
        assert all(isinstance(w, float) for w in weights)
        assert all(w > 0 for w in weights)
        # Weights should sum to 1 (normalized)
        assert abs(sum(weights) - 1.0) < 0.01
    
    def test_calculate_token_weights_with_tf_idf(self, mock_generator):
        """Test token weight calculation with TF-IDF."""
        tokens = ["ustawa", "reguluje", "ustawa", "zasady"]  # "ustawa" appears twice
        text = "Ustawa reguluje... ustawa..."
        
        weights = mock_generator._calculate_token_weights(
            tokens, text, use_tf_idf=True, use_legal_weighting=False
        )
        
        assert len(weights) == len(tokens)
        # Weights for repeated "ustawa" should be adjusted by TF-IDF
        ustawa_weights = [weights[i] for i, token in enumerate(tokens) if token == "ustawa"]
        assert len(ustawa_weights) == 2
    
    def test_calculate_token_weights_with_legal_weighting(self, mock_generator):
        """Test token weight calculation with legal weighting."""
        tokens = ["ustawa", "random", "artykuł", "word"]
        text = "Ustawa art. 5 random word"
        
        weights = mock_generator._calculate_token_weights(
            tokens, text, use_tf_idf=False, use_legal_weighting=True
        )
        
        assert len(weights) == len(tokens)
        # Legal terms should have higher weights
        ustawa_idx = tokens.index("ustawa")
        random_idx = tokens.index("random")
        assert weights[ustawa_idx] > weights[random_idx]
    
    def test_create_weighted_embedding(self, mock_generator):
        """Test weighted embedding creation."""
        token_embeddings = [np.random.randn(768) for _ in range(3)]
        token_weights = [0.5, 0.3, 0.2]
        
        weighted_embedding = mock_generator._create_weighted_embedding(
            token_embeddings, token_weights
        )
        
        assert isinstance(weighted_embedding, np.ndarray)
        assert len(weighted_embedding) == 768
        # Should be normalized
        assert abs(np.linalg.norm(weighted_embedding) - 1.0) < 0.01
    
    def test_create_weighted_embedding_empty(self, mock_generator):
        """Test weighted embedding creation with empty input."""
        weighted_embedding = mock_generator._create_weighted_embedding([], [])
        
        assert isinstance(weighted_embedding, np.ndarray)
        assert len(weighted_embedding) == 768
        assert np.allclose(weighted_embedding, 0.0)
    
    def test_calculate_coverage_score(self, mock_generator):
        """Test coverage score calculation."""
        tokens = ["ustawa", "reguluje", "zasady"]
        text = "Ustawa reguluje zasady opodatkowania."
        
        coverage = mock_generator._calculate_coverage_score(tokens, text)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
        assert coverage > 0  # Should have some coverage
    
    def test_generate_bag_embedding(self, mock_generator):
        """Test bag embedding generation."""
        text = "Ustawa z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych."
        title = "Ustawa o podatku dochodowym"
        
        result = mock_generator.generate_bag_embedding(text, title)
        
        assert isinstance(result, BagEmbeddingResult)
        assert isinstance(result.document_embedding, np.ndarray)
        assert len(result.document_embedding) == 768
        assert len(result.token_embeddings) > 0
        assert len(result.token_weights) == len(result.tokens)
        assert result.vocabulary_size > 0
        assert 0.0 <= result.coverage_score <= 1.0
        assert result.processing_time >= 0.0
        assert isinstance(result.metadata, dict)
    
    def test_generate_bag_embedding_empty_text(self, mock_generator):
        """Test bag embedding generation with empty text."""
        result = mock_generator.generate_bag_embedding("")
        
        assert isinstance(result, BagEmbeddingResult)
        assert len(result.document_embedding) == 768
        assert result.token_embeddings == []
        assert result.tokens == []
        assert result.vocabulary_size == 0
    
    def test_generate_bag_embeddings_batch(self, mock_generator):
        """Test batch bag embedding generation."""
        documents = [
            {"text": "Ustawa o podatku dochodowym.", "title": "Podatek dochodowy"},
            {"text": "Rozporządzenie o VAT.", "title": "VAT"},
            {"text": "Kodeks cywilny art. 1."}  # No title
        ]
        
        results = mock_generator.generate_bag_embeddings_batch(documents)
        
        assert len(results) == 3
        assert all(isinstance(r, BagEmbeddingResult) for r in results)
        assert all(len(r.document_embedding) == 768 for r in results)
    
    def test_generate_bag_embeddings_batch_with_error(self, mock_generator):
        """Test batch generation with error handling."""
        # Mock encoder to raise exception for one document
        def side_effect(texts):
            if "error" in texts[0]:
                raise ValueError("Mock error")
            return [np.random.randn(768) for _ in texts]
        
        mock_generator.encoder.encode.side_effect = side_effect
        
        documents = [
            {"text": "Normal text"},
            {"text": "error text"},  # This should cause error
            {"text": "Another normal text"}
        ]
        
        results = mock_generator.generate_bag_embeddings_batch(documents)
        
        assert len(results) == 3
        # Error case should return empty result
        assert results[1].vocabulary_size == 0
        assert 'error' in results[1].metadata
    
    def test_get_token_importance_analysis(self, mock_generator):
        """Test token importance analysis."""
        # Create a mock result
        result = BagEmbeddingResult(
            document_embedding=np.random.randn(768),
            token_embeddings=[np.random.randn(768) for _ in range(5)],
            token_weights=[0.4, 0.3, 0.2, 0.08, 0.02],
            tokens=["ustawa", "podatek", "artykuł", "oraz", "i"],
            vocabulary_size=5,
            coverage_score=0.8,
            processing_time=1.0,
            metadata={}
        )
        
        analysis = mock_generator.get_token_importance_analysis(result)
        
        assert isinstance(analysis, dict)
        assert 'most_important_tokens' in analysis
        assert 'least_important_tokens' in analysis
        assert 'weight_statistics' in analysis
        assert 'vocabulary_diversity' in analysis
        assert 'coverage_score' in analysis
        
        # Most important token should be "ustawa" (highest weight)
        assert analysis['most_important_tokens'][0][0] == "ustawa"
        
        # Check statistics
        stats = analysis['weight_statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_get_token_importance_analysis_empty(self, mock_generator):
        """Test token importance analysis with empty result."""
        result = BagEmbeddingResult(
            document_embedding=np.zeros(768),
            token_embeddings=[],
            token_weights=[],
            tokens=[],
            vocabulary_size=0,
            coverage_score=0.0,
            processing_time=0.0,
            metadata={}
        )
        
        analysis = mock_generator.get_token_importance_analysis(result)
        
        assert 'error' in analysis


def test_get_bag_embeddings_generator():
    """Test global generator instance."""
    with patch('sejm_whiz.embeddings.bag_embeddings.BagEmbeddingsGenerator') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        generator1 = get_bag_embeddings_generator()
        generator2 = get_bag_embeddings_generator()
        
        # Should return same instance (singleton pattern)
        assert generator1 == generator2
        mock_class.assert_called_once()


class TestBagEmbeddingResult:
    """Test BagEmbeddingResult data class."""
    
    def test_creation(self):
        """Test result creation."""
        result = BagEmbeddingResult(
            document_embedding=np.random.randn(768),
            token_embeddings=[np.random.randn(768) for _ in range(3)],
            token_weights=[0.5, 0.3, 0.2],
            tokens=["token1", "token2", "token3"],
            vocabulary_size=3,
            coverage_score=0.8,
            processing_time=1.5,
            metadata={"test": "value"}
        )
        
        assert len(result.document_embedding) == 768
        assert len(result.token_embeddings) == 3
        assert len(result.token_weights) == 3
        assert len(result.tokens) == 3
        assert result.vocabulary_size == 3
        assert result.coverage_score == 0.8
        assert result.processing_time == 1.5
        assert result.metadata["test"] == "value"