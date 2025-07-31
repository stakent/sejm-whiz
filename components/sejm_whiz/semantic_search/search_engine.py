"""Main semantic search engine for legal documents."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


from sejm_whiz.embeddings import get_herbert_embedder
from sejm_whiz.vector_db import get_similarity_search, DistanceMetric
from sejm_whiz.text_processing import process_legal_document
from sejm_whiz.database.models import LegalDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from semantic search operation."""

    document: LegalDocument
    similarity_score: float
    embedding_distance: float
    matched_passages: List[str]
    search_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "document_id": str(self.document.id),
            "title": self.document.title,
            "document_type": self.document.document_type,
            "similarity_score": self.similarity_score,
            "embedding_distance": self.embedding_distance,
            "matched_passages": self.matched_passages,
            "legal_domain": self.document.legal_domain,
            "published_at": self.document.published_at.isoformat()
            if self.document.published_at
            else None,
            "search_metadata": self.search_metadata,
        }


class SemanticSearchEngine:
    """Main semantic search engine for legal documents."""

    def __init__(
        self,
        embedder=None,
        similarity_search=None,
    ):
        """Initialize semantic search engine.

        Args:
            embedder: HerBERT embedder instance
            similarity_search: Vector similarity search instance
        """
        self.embedder = embedder or get_herbert_embedder()
        self.similarity_search = similarity_search or get_similarity_search()
        self.logger = logging.getLogger(__name__)

    def search(
        self,
        query: str,
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        document_type: Optional[str] = None,
        legal_domain: Optional[str] = None,
        similarity_threshold: float = 0.0,
        include_passages: bool = True,
    ) -> List[SearchResult]:
        """Perform semantic search on legal documents.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            distance_metric: Vector distance metric to use
            document_type: Filter by document type (optional)
            legal_domain: Filter by legal domain (optional)
            similarity_threshold: Minimum similarity score threshold
            include_passages: Whether to extract relevant passages

        Returns:
            List of SearchResult objects
        """
        try:
            # Process query text
            processed_query = process_legal_document(query)
            clean_query_text = processed_query.get("clean_text", query)
            self.logger.info(
                f"Processed search query: {len(clean_query_text)} characters"
            )

            # Generate query embedding
            embedding_result = self.embedder.generate_embedding(clean_query_text)
            if not embedding_result.success:
                raise ValueError(
                    f"Failed to generate query embedding: {embedding_result.error}"
                )

            query_embedding = embedding_result.embedding.tolist()
            self.logger.info(
                f"Generated query embedding: {len(query_embedding)} dimensions"
            )

            # Perform vector similarity search
            similar_docs = self.similarity_search.find_similar_documents(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more candidates for filtering
                distance_metric=distance_metric,
                document_type=document_type,
                legal_domain=legal_domain,
            )

            # Convert to SearchResult objects
            search_results = []
            for doc, distance in similar_docs:
                # Convert distance to similarity score (0-1)
                if distance_metric == DistanceMetric.COSINE:
                    similarity_score = 1.0 - distance
                elif distance_metric == DistanceMetric.L2:
                    similarity_score = 1.0 / (1.0 + distance)
                else:  # Inner product
                    similarity_score = distance

                # Apply similarity threshold
                if similarity_score < similarity_threshold:
                    continue

                # Extract relevant passages if requested
                matched_passages = []
                if include_passages:
                    matched_passages = self._extract_relevant_passages(
                        query, doc.content, max_passages=3
                    )

                search_metadata = {
                    "distance_metric": distance_metric.value,
                    "raw_distance": distance,
                    "query_length": len(query),
                    "document_length": len(doc.content),
                    "search_timestamp": datetime.utcnow().isoformat(),
                }

                result = SearchResult(
                    document=doc,
                    similarity_score=similarity_score,
                    embedding_distance=distance,
                    matched_passages=matched_passages,
                    search_metadata=search_metadata,
                )

                search_results.append(result)

                if len(search_results) >= limit:
                    break

            self.logger.info(f"Found {len(search_results)} matching documents")
            return search_results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise

    def batch_search(
        self,
        queries: List[str],
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs,
    ) -> List[List[SearchResult]]:
        """Perform batch semantic search for multiple queries.

        Args:
            queries: List of search query texts
            limit: Maximum number of results per query
            distance_metric: Vector distance metric to use
            **kwargs: Additional search parameters

        Returns:
            List of search results for each query
        """
        try:
            results = []
            for query in queries:
                query_results = self.search(
                    query=query, limit=limit, distance_metric=distance_metric, **kwargs
                )
                results.append(query_results)

            self.logger.info(f"Completed batch search for {len(queries)} queries")
            return results

        except Exception as e:
            self.logger.error(f"Batch semantic search failed: {e}")
            raise

    def find_similar_to_document(
        self,
        document_id: UUID,
        limit: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        exclude_self: bool = True,
    ) -> List[SearchResult]:
        """Find documents similar to a given document.

        Args:
            document_id: ID of the reference document
            limit: Maximum number of results to return
            distance_metric: Vector distance metric to use
            exclude_self: Whether to exclude the reference document from results

        Returns:
            List of SearchResult objects
        """
        try:
            # Get the reference document with its embedding
            similar_docs = self.similarity_search.find_similar_to_document(
                document_id=document_id,
                limit=limit + (1 if exclude_self else 0),
                distance_metric=distance_metric,
            )

            search_results = []
            for doc, distance in similar_docs:
                # Skip self if requested
                if exclude_self and doc.id == document_id:
                    continue

                # Convert distance to similarity score
                if distance_metric == DistanceMetric.COSINE:
                    similarity_score = 1.0 - distance
                elif distance_metric == DistanceMetric.L2:
                    similarity_score = 1.0 / (1.0 + distance)
                else:  # Inner product
                    similarity_score = distance

                search_metadata = {
                    "distance_metric": distance_metric.value,
                    "raw_distance": distance,
                    "reference_document_id": str(document_id),
                    "search_timestamp": datetime.utcnow().isoformat(),
                }

                result = SearchResult(
                    document=doc,
                    similarity_score=similarity_score,
                    embedding_distance=distance,
                    matched_passages=[],  # Not applicable for document similarity
                    search_metadata=search_metadata,
                )

                search_results.append(result)

                if len(search_results) >= limit:
                    break

            self.logger.info(f"Found {len(search_results)} similar documents")
            return search_results

        except Exception as e:
            self.logger.error(f"Document similarity search failed: {e}")
            raise

    def _extract_relevant_passages(
        self,
        query: str,
        document_text: str,
        max_passages: int = 3,
        passage_length: int = 200,
    ) -> List[str]:
        """Extract relevant passages from document text.

        Args:
            query: Search query
            document_text: Full document text
            max_passages: Maximum number of passages to extract
            passage_length: Approximate length of each passage

        Returns:
            List of relevant text passages
        """
        try:
            # Simple passage extraction based on query term overlap
            # In production, this could use more sophisticated techniques

            passages = []
            query_terms = set(query.lower().split())

            # Split document into sentences
            sentences = document_text.split(".")

            # Score sentences by query term overlap
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                sentence_terms = set(sentence.lower().split())
                overlap_score = len(query_terms.intersection(sentence_terms)) / len(
                    query_terms
                )

                if overlap_score > 0:
                    scored_sentences.append((overlap_score, i, sentence))

            # Sort by score and select top sentences
            scored_sentences.sort(reverse=True)

            for score, idx, sentence in scored_sentences[:max_passages]:
                # Create passage with context
                start_idx = max(0, idx - 1)
                end_idx = min(len(sentences), idx + 2)

                passage = ". ".join(sentences[start_idx:end_idx]).strip()
                if len(passage) > passage_length:
                    passage = passage[:passage_length] + "..."

                passages.append(passage)

            return passages[:max_passages]

        except Exception as e:
            self.logger.warning(f"Failed to extract passages: {e}")
            return []


# Singleton instance
_search_engine_instance = None


def get_search_engine() -> SemanticSearchEngine:
    """Get singleton semantic search engine instance."""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = SemanticSearchEngine()
    return _search_engine_instance
