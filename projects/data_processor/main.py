#!/usr/bin/env python3
"""Data processor main entry point for batch processing workflows."""

import asyncio
import logging
from typing import Dict, Any

from sejm_whiz.data_pipeline.core import (
    DataPipeline,
    PipelineStep,
    BatchProcessor,
    DataPipelineError,
)
from sejm_whiz.sejm_api import SejmApiClient
from sejm_whiz.eli_api import EliApiClient
from sejm_whiz.text_processing import TextProcessor
from sejm_whiz.embeddings import BagEmbeddingsGenerator
from sejm_whiz.vector_db import VectorDBOperations
from sejm_whiz.database import DocumentOperations


class SejmDataIngestionStep(PipelineStep):
    """Pipeline step for ingesting Sejm proceedings data."""

    def __init__(self):
        super().__init__("sejm_ingestion")
        self.client = SejmApiClient()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and process Sejm proceedings data."""
        self.logger.info("Fetching Sejm proceedings data")

        # Extract parameters from input data
        session_id = data.get("session_id")
        # date_range = data.get("date_range")  # Reserved for future filtering

        # Fetch proceeding sittings for the session (limit to prevent infinite processing)
        if session_id:
            proceedings = await self.client.get_proceeding_sittings(
                term=int(session_id)
            )
        else:
            # Default to current term proceeding sittings
            proceedings = await self.client.get_proceeding_sittings()

        # Limit to first 5 proceedings to prevent infinite processing
        if len(proceedings) > 5:
            self.logger.info(
                f"Limiting proceedings from {len(proceedings)} to 5 for processing"
            )
            proceedings = proceedings[:5]

        return {
            **data,
            "sejm_proceedings": proceedings,
            "step_completed": "sejm_ingestion",
        }


class ELIDataIngestionStep(PipelineStep):
    """Pipeline step for ingesting ELI legal documents."""

    def __init__(self):
        super().__init__("eli_ingestion")
        self.client = EliApiClient()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and process ELI legal documents."""
        self.logger.info("Fetching ELI legal documents")

        # Extract parameters from input data
        document_ids = data.get("document_ids", [])
        category = data.get("category")

        # Fetch legal documents
        if document_ids:
            documents = await self.client.batch_get_documents(document_ids)
        else:
            # Search for recent documents in category
            documents = await self.client.search_documents(
                filters={"type": category} if category else None, limit=10
            )

        return {**data, "eli_documents": documents, "step_completed": "eli_ingestion"}


class TextProcessingStep(PipelineStep):
    """Pipeline step for text processing and cleaning."""

    def __init__(self):
        super().__init__("text_processing")
        self.processor = TextProcessor()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean text data."""
        self.logger.info("Processing text data")

        processed_data = {}

        # Process Sejm proceedings if available
        if "sejm_proceedings" in data:
            self.logger.info("Processing Sejm proceedings text")
            proceedings = data["sejm_proceedings"]
            processed_proceedings = []

            for proceeding in proceedings:
                # Extract text content from proceeding (agenda contains the main content)
                content = getattr(proceeding, "agenda", "") or ""
                processed_text = self.processor.clean_text(content)

                # Convert to dict and add processed content
                proceeding_dict = proceeding.model_dump()
                proceeding_dict["processed_content"] = processed_text
                processed_proceedings.append(proceeding_dict)

            processed_data["processed_sejm_proceedings"] = processed_proceedings

        # Process ELI documents if available
        if "eli_documents" in data:
            self.logger.info("Processing ELI documents text")
            documents = data["eli_documents"]
            processed_documents = []

            for document in documents:
                processed_text = self.processor.clean_text(document.get("content", ""))
                processed_documents.append(
                    {**document, "processed_content": processed_text}
                )

            processed_data["processed_eli_documents"] = processed_documents

        return {**data, **processed_data, "step_completed": "text_processing"}


class EmbeddingGenerationStep(PipelineStep):
    """Pipeline step for generating embeddings."""

    def __init__(self):
        super().__init__("embedding_generation")
        self.generator = BagEmbeddingsGenerator()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for processed text."""
        self.logger.info("Generating embeddings")

        embeddings_data = {}

        # Generate embeddings for Sejm proceedings
        if "processed_sejm_proceedings" in data:
            proceedings = data["processed_sejm_proceedings"]
            proceedings_with_embeddings = []
            total_proceedings = len(proceedings)

            self.logger.info(
                f"Generating embeddings for {total_proceedings} proceedings"
            )

            for i, proceeding in enumerate(proceedings):
                text = proceeding.get("processed_content", "")
                if text:
                    self.logger.info(f"Processing proceeding {i+1}/{total_proceedings}")
                    embedding = self.generator.generate_bag_embedding(text)
                    proceedings_with_embeddings.append(
                        {**proceeding, "embedding": embedding}
                    )

            embeddings_data["sejm_proceedings_embeddings"] = proceedings_with_embeddings
            self.logger.info(
                f"Completed embedding generation for {len(proceedings_with_embeddings)} proceedings"
            )

        # Generate embeddings for ELI documents
        if "processed_eli_documents" in data:
            documents = data["processed_eli_documents"]
            documents_with_embeddings = []

            for document in documents:
                text = document.get("processed_content", "")
                if text:
                    embedding = self.generator.generate_bag_embedding(text)
                    documents_with_embeddings.append(
                        {**document, "embedding": embedding}
                    )

            embeddings_data["eli_documents_embeddings"] = documents_with_embeddings

        return {**data, **embeddings_data, "step_completed": "embedding_generation"}


class DatabaseStorageStep(PipelineStep):
    """Pipeline step for storing data in database."""

    def __init__(self):
        super().__init__("database_storage")
        self.db = DocumentOperations()
        self.vector_db = VectorDBOperations()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed data in database."""
        self.logger.info("Storing data in database")

        storage_results = {}

        # Store Sejm proceedings with embeddings
        if "sejm_proceedings_embeddings" in data:
            proceedings = data["sejm_proceedings_embeddings"]
            stored_ids = []

            for proceeding in proceedings:
                # Create document record (returns UUID)
                proceeding_id = self.db.create_document(
                    title=proceeding.get("title", "Sejm Proceeding"),
                    content=proceeding.get("processed_content", ""),
                    document_type="sejm_proceeding",
                    eli_identifier=None,
                    source_url=proceeding.get("url"),
                    metadata=proceeding,
                )

                # Store embedding in vector database
                if "embedding" in proceeding:
                    self.vector_db.create_document_embedding(
                        document_id=proceeding_id,  # Now proceeding_id is already a UUID
                        embedding=proceeding["embedding"].document_embedding.tolist(),
                        model_name="allegro/herbert-base-cased",
                        model_version="1.0",
                        embedding_method="bag_of_embeddings",
                        token_count=len(proceeding["embedding"].tokens)
                        if hasattr(proceeding["embedding"], "tokens")
                        else None,
                    )

                stored_ids.append(proceeding_id)

            storage_results["stored_sejm_proceedings"] = stored_ids

        # Store ELI documents with embeddings
        if "eli_documents_embeddings" in data:
            documents = data["eli_documents_embeddings"]
            stored_ids = []

            for document in documents:
                # Create document record (returns UUID)
                document_id = self.db.create_document(
                    title=document.get("title", "ELI Document"),
                    content=document.get("processed_content", ""),
                    document_type="eli_document",
                    eli_identifier=document.get("eli_identifier"),
                    source_url=document.get("url"),
                    metadata=document,
                )

                # Store embedding in vector database
                if "embedding" in document:
                    self.vector_db.create_document_embedding(
                        document_id=document_id,  # Now document_id is already a UUID
                        embedding=document["embedding"].document_embedding.tolist(),
                        model_name="allegro/herbert-base-cased",
                        model_version="1.0",
                        embedding_method="bag_of_embeddings",
                        token_count=len(document["embedding"].tokens)
                        if hasattr(document["embedding"], "tokens")
                        else None,
                    )

                stored_ids.append(document_id)

            storage_results["stored_eli_documents"] = stored_ids

        return {**data, **storage_results, "step_completed": "database_storage"}


async def create_full_ingestion_pipeline() -> DataPipeline:
    """Create a complete data ingestion pipeline."""
    steps = [
        SejmDataIngestionStep(),
        ELIDataIngestionStep(),
        TextProcessingStep(),
        EmbeddingGenerationStep(),
        DatabaseStorageStep(),
    ]

    return DataPipeline("full_ingestion", steps)


async def create_sejm_only_pipeline() -> DataPipeline:
    """Create a Sejm-only data ingestion pipeline."""
    steps = [
        SejmDataIngestionStep(),
        TextProcessingStep(),
        EmbeddingGenerationStep(),
        DatabaseStorageStep(),
    ]

    return DataPipeline("sejm_ingestion", steps)


async def create_eli_only_pipeline() -> DataPipeline:
    """Create an ELI-only data ingestion pipeline."""
    steps = [
        ELIDataIngestionStep(),
        TextProcessingStep(),
        EmbeddingGenerationStep(),
        DatabaseStorageStep(),
    ]

    return DataPipeline("eli_ingestion", steps)


async def main():
    """Main entry point for data processor."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("data_processor")
    logger.info("Starting data processor")

    try:
        # Example: Process Sejm proceedings for a specific session
        pipeline = await create_sejm_only_pipeline()

        # Sample input data
        input_data = {
            "session_id": "10",
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
        }

        # Run pipeline
        result = await pipeline.run(input_data)
        logger.info(
            f"Pipeline completed successfully. Result keys: {list(result.keys())}"
        )

        # Print metrics
        metrics = pipeline.get_metrics()
        logger.info(f"Pipeline metrics: {metrics}")

        # Example: Batch processing multiple sessions
        logger.info("Starting batch processing example")

        batch_processor = BatchProcessor(pipeline, batch_size=5)

        batch_data = [
            {
                "session_id": "10",
                "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            },
            {
                "session_id": "11",
                "date_range": {"start": "2024-02-01", "end": "2024-02-28"},
            },
            {
                "session_id": "12",
                "date_range": {"start": "2024-03-01", "end": "2024-03-31"},
            },
        ]

        batch_results = await batch_processor.process_batch(batch_data)
        logger.info(f"Batch processing completed. Processed {len(batch_results)} items")

    except DataPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    logger.info("Data processor completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
