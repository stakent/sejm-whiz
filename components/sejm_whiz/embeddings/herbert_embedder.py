"""HerBERT-based embedding generation for Polish legal documents."""

import torch
import logging
import re
import numpy as np
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from huggingface_hub import hf_hub_download, list_repo_files
import time

from .config import get_embedding_config, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: np.ndarray
    model_name: str
    processing_time: float
    token_count: int
    chunk_count: int = 1
    quality_score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HerBERTEmbedder:
    """HerBERT-based embedding generator for Polish legal text."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_embedding_config()
        self.tokenizer = None
        self.model = None
        self.device = None
        self._model_loaded = False

        # Text preprocessing patterns
        self.polish_legal_patterns = {
            "article_marker": re.compile(r"\bart\.\s*\d+[a-z]?\b", re.IGNORECASE),
            "paragraph_marker": re.compile(r"§\s*\d+", re.IGNORECASE),
            "section_marker": re.compile(r"\bust\.\s*\d+", re.IGNORECASE),
            "definition_marker": re.compile(
                r"oznacza|rozumie się|należy przez to rozumieć", re.IGNORECASE
            ),
            "amendment_marker": re.compile(
                r"zmienia się|uchyla się|dodaje się|skreśla się", re.IGNORECASE
            ),
        }

    def _initialize_model(self) -> None:
        """Initialize HerBERT model and tokenizer."""
        if self._model_loaded:
            return

        logger.info(f"Loading HerBERT model: {self.config.model_name}")
        logger.info(f"Config cache dir: {repr(self.config.model_cache_dir)}")
        start_time = time.time()

        try:
            # Determine device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            cache_dir = None
            if self.config.model_cache_dir and self.config.model_cache_dir.strip():
                import os

                cache_dir = os.path.abspath(self.config.model_cache_dir)
                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Resolved cache dir: {repr(cache_dir)}")

            try:
                logger.info(
                    f"Loading tokenizer with model_name: {repr(self.config.model_name)}"
                )

                # First, let's check what files are available in the repo
                try:
                    repo_files = list_repo_files(self.config.model_name)
                    logger.info(f"Available files in repo: {repo_files}")

                    # Check for vocabulary files
                    vocab_files = [f for f in repo_files if "vocab" in f.lower()]
                    logger.info(f"Vocabulary files found: {vocab_files}")

                except Exception as e:
                    logger.warning(f"Could not list repo files: {e}")

                # Try to manually download required files
                logger.info("Attempting to manually download tokenizer files...")
                try:
                    # Download config.json first
                    config_path = hf_hub_download(
                        repo_id=self.config.model_name,
                        filename="config.json",
                        cache_dir=cache_dir,
                    )
                    logger.info(f"Downloaded config.json to: {config_path}")

                    # Try to download tokenizer.json (fast tokenizer)
                    try:
                        tokenizer_path = hf_hub_download(
                            repo_id=self.config.model_name,
                            filename="tokenizer.json",
                            cache_dir=cache_dir,
                        )
                        logger.info(f"Downloaded tokenizer.json to: {tokenizer_path}")
                    except Exception as e:
                        logger.warning(f"Could not download tokenizer.json: {e}")

                    # Try to download vocab.json and merges.txt
                    for filename in ["vocab.json", "merges.txt"]:
                        try:
                            file_path = hf_hub_download(
                                repo_id=self.config.model_name,
                                filename=filename,
                                cache_dir=cache_dir,
                            )
                            logger.info(f"Downloaded {filename} to: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not download {filename}: {e}")

                except Exception as e:
                    logger.error(f"Manual download failed: {e}")

                # Now try loading the tokenizer
                logger.info("Attempting to load tokenizer after manual download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    trust_remote_code=True,
                )
                logger.info("Tokenizer loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # Load model
            try:
                logger.info(
                    f"Loading model with model_name: {repr(self.config.model_name)}"
                )
                # Try loading with minimal parameters first
                self.model = AutoModel.from_pretrained(self.config.model_name)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Compile model if requested (PyTorch 2.0+)
            if self.config.compile_model and hasattr(torch, "compile"):
                logger.info("Compiling model with PyTorch 2.0")
                self.model = torch.compile(self.model)

            self._model_loaded = True

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load HerBERT model: {e}")
            raise

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        self._initialize_model()

        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()

        results = []

        # Process texts in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)

        total_time = time.time() - start_time
        logger.info(f"Generated {len(results)} embeddings in {total_time:.2f}s")

        return results

    def _process_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process a batch of texts."""
        batch_results = []

        for text in texts:
            try:
                start_time = time.time()

                # Preprocess text
                processed_text = self._preprocess_text(text)

                # Check if text needs chunking
                if len(processed_text) > self.config.max_text_length:
                    chunks = self._chunk_text(processed_text)
                    embedding = self._embed_chunks(chunks)
                    chunk_count = len(chunks)
                else:
                    embedding = self._embed_single_text(processed_text)
                    chunk_count = 1

                # Calculate quality score
                quality_score = self._calculate_quality_score(processed_text)

                # Count tokens
                token_count = len(
                    self.tokenizer.encode(
                        processed_text,
                        truncation=True,
                        max_length=self.config.max_length,
                    )
                )

                processing_time = time.time() - start_time

                result = EmbeddingResult(
                    embedding=embedding,
                    model_name=self.config.model_name,
                    processing_time=processing_time,
                    token_count=token_count,
                    chunk_count=chunk_count,
                    quality_score=quality_score,
                    metadata={
                        "text_length": len(processed_text),
                        "device": str(self.device),
                        "pooling_strategy": self.config.pooling_strategy,
                    },
                )

                batch_results.append(result)

            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                # Create fallback result
                batch_results.append(
                    EmbeddingResult(
                        embedding=np.zeros(self.config.embedding_dim),
                        model_name=self.config.model_name,
                        processing_time=0.0,
                        token_count=0,
                        quality_score=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return batch_results

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        if not text:
            return ""

        processed = text

        # Normalize whitespace
        if self.config.normalize_text:
            processed = re.sub(r"\s+", " ", processed.strip())

        # Remove special characters if requested
        if self.config.remove_special_chars:
            processed = re.sub(r"[^\w\s\.,;:!?()-]", "", processed)

        # Lowercase if requested (not recommended for HerBERT)
        if self.config.lowercase:
            processed = processed.lower()

        # Ensure minimum length
        if len(processed) < self.config.min_text_length:
            processed = processed + " " * (self.config.min_text_length - len(processed))

        return processed

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks."""
        if self.config.chunk_strategy == "sentence":
            return self._chunk_by_sentences(text)
        elif self.config.chunk_strategy == "token":
            return self._chunk_by_tokens(text)
        else:  # fixed
            return self._chunk_fixed_size(text)

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with overlap."""
        # Simple sentence splitting for Polish
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk text by token count."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Chunk text by fixed character size."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)

        return chunks

    def _embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for chunks and combine them."""
        chunk_embeddings = []

        for chunk in chunks:
            embedding = self._embed_single_text(chunk)
            chunk_embeddings.append(embedding)

        # Combine chunk embeddings using mean pooling
        combined_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        return combined_embedding

    def _embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Apply pooling strategy
            if self.config.pooling_strategy == "cls":
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.config.pooling_strategy == "max":
                # Max pooling
                embedding = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
            else:  # mean
                # Mean pooling with attention mask
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                embedding = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.cpu().numpy()

        # Normalize embedding
        embedding = embedding.squeeze()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for the text."""
        score = 0.0

        # Length-based scoring
        text_length = len(text)
        if 100 <= text_length <= 10000:
            score += 0.3
        elif 50 <= text_length <= 50000:
            score += 0.2
        else:
            score += 0.1

        # Legal structure scoring
        if self.polish_legal_patterns["article_marker"].search(text):
            score += 0.2

        if self.polish_legal_patterns["paragraph_marker"].search(text):
            score += 0.1

        if self.polish_legal_patterns["definition_marker"].search(text):
            score += 0.2

        # Language quality (simple heuristic)
        word_count = len(text.split())
        if word_count > 10:
            avg_word_length = sum(len(word) for word in text.split()) / word_count
            if 4 <= avg_word_length <= 8:  # Reasonable for Polish
                score += 0.2

        return min(score, 1.0)

    def embed_legal_document(
        self, title: str, content: str, document_type: str = "unknown"
    ) -> EmbeddingResult:
        """Generate embeddings specifically for legal documents with weighted sections."""

        sections = self._extract_legal_sections(title, content, document_type)

        # Generate embeddings for each section
        section_embeddings = []
        section_weights = []

        for section_type, section_text in sections.items():
            if section_text:
                section_embedding = self._embed_single_text(section_text)
                section_weight = self.config.legal_section_weights.get(
                    section_type, 1.0
                )

                section_embeddings.append(section_embedding * section_weight)
                section_weights.append(section_weight)

        if not section_embeddings:
            # Fallback to simple content embedding
            return self.embed_text(content)

        # Weighted combination of section embeddings
        weighted_embedding = np.sum(section_embeddings, axis=0) / sum(section_weights)
        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)

        # Calculate total tokens
        total_text = " ".join(sections.values())
        token_count = len(
            self.tokenizer.encode(
                total_text, truncation=True, max_length=self.config.max_length
            )
        )

        return EmbeddingResult(
            embedding=weighted_embedding,
            model_name=self.config.model_name,
            processing_time=0.0,  # Would need to track this properly
            token_count=token_count,
            quality_score=self._calculate_quality_score(content),
            metadata={
                "sections_used": list(sections.keys()),
                "section_weights": section_weights,
                "document_type": document_type,
            },
        )

    def _extract_legal_sections(
        self, title: str, content: str, document_type: str
    ) -> Dict[str, str]:
        """Extract different sections from legal document for weighted embedding."""

        sections = {"title": title or "", "content": content or ""}

        # Extract articles
        articles = self.polish_legal_patterns["article_marker"].findall(content)
        if articles:
            sections["article"] = " ".join(articles[:5])  # First 5 articles

        # Extract definitions
        definition_matches = self.polish_legal_patterns["definition_marker"].finditer(
            content
        )
        definitions = []
        for match in definition_matches:
            start = match.start()
            definition_text = content[
                start : start + 200
            ]  # 200 chars after definition marker
            definitions.append(definition_text)

        if definitions:
            sections["definition"] = " ".join(definitions[:3])  # First 3 definitions

        # Extract amendments if applicable
        if self.polish_legal_patterns["amendment_marker"].search(content):
            amendment_matches = self.polish_legal_patterns["amendment_marker"].finditer(
                content
            )
            amendments = []
            for match in amendment_matches:
                start = match.start()
                amendment_text = content[start : start + 150]
                amendments.append(amendment_text)

            if amendments:
                sections["amendment"] = " ".join(amendments[:2])

        return sections

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        logger.info("HerBERT embedder cleaned up")


# Global embedder instance
_herbert_embedder: Optional[HerBERTEmbedder] = None


def get_herbert_embedder(config: Optional[EmbeddingConfig] = None) -> HerBERTEmbedder:
    """Get global HerBERT embedder instance."""
    global _herbert_embedder

    if _herbert_embedder is None:
        _herbert_embedder = HerBERTEmbedder(config)

    return _herbert_embedder
