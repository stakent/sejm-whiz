#!/usr/bin/env python3
"""Data processor main entry point for batch processing workflows."""

import asyncio
import logging

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
from .models import (
    PipelineInput,
    SejmIngestionData,
    EliIngestionData,
    TextProcessingData,
    EmbeddingGenerationData,
    DatabaseStorageData,
    ProcessedDocument,
    DocumentWithEmbedding,
)


class SejmDataIngestionStep(PipelineStep):
    """Pipeline step for ingesting Sejm proceedings data."""

    def __init__(self):
        super().__init__("sejm_ingestion")
        self.client = SejmApiClient()

    async def process(self, data: PipelineInput) -> SejmIngestionData:
        """Fetch and process Sejm proceedings data."""
        self.logger.info("Fetching Sejm proceedings data")

        # Extract parameters from input data
        session_id = data.session_id
        # date_range = data.date_range  # Reserved for future filtering

        # Fetch proceeding sittings for the session (limit to prevent infinite processing)
        if session_id:
            proceedings = await self.client.get_proceeding_sittings(
                term=int(session_id)
            )
        else:
            # Default to current term proceeding sittings
            proceedings = await self.client.get_proceeding_sittings()

        return SejmIngestionData(
            sejm_proceedings=proceedings,
            session_id=data.session_id,
            date_range=data.date_range,
            category=data.category,
            document_ids=data.document_ids,
        )


class ELIDataIngestionStep(PipelineStep):
    """Pipeline step for ingesting ELI legal documents."""

    def __init__(self):
        super().__init__("eli_ingestion")
        self.client = EliApiClient()

    async def process(self, data: SejmIngestionData) -> EliIngestionData:
        """Fetch and process ELI legal documents."""
        self.logger.info("Fetching ELI legal documents")

        # Extract parameters from input data
        document_ids = data.document_ids or []
        category = data.category

        # Fetch legal documents
        if document_ids:
            documents = await self.client.batch_get_documents(document_ids)
        else:
            # Search for recent documents in category
            documents = await self.client.search_documents(
                document_type=category, limit=10
            )

        return EliIngestionData(
            eli_documents=documents.documents,
            sejm_proceedings=data.sejm_proceedings,
            session_id=data.session_id,
            date_range=data.date_range,
            category=data.category,
            document_ids=data.document_ids,
        )


class TextProcessingStep(PipelineStep):
    """Pipeline step for text processing and cleaning."""

    def __init__(self):
        super().__init__("text_processing")
        self.processor = TextProcessor()

    async def process(self, data: EliIngestionData) -> TextProcessingData:
        """Process and clean text data."""
        self.logger.info("Processing text data")

        processed_sejm_proceedings = None
        processed_eli_documents = None

        # Process Sejm proceedings if available
        if data.sejm_proceedings:
            self.logger.info("Processing Sejm proceedings text")
            proceedings = data.sejm_proceedings
            processed_proceedings = []

            for proceeding in proceedings:
                # Extract text content from proceeding (agenda contains the main content)
                content = getattr(proceeding, "agenda", "") or ""
                processed_text = self.processor.clean_text(content)

                # Convert to ProcessedDocument
                proceeding_dict = proceeding.model_dump()
                proceeding_dict["processed_content"] = processed_text
                processed_proceedings.append(ProcessedDocument(**proceeding_dict))

            processed_sejm_proceedings = processed_proceedings

        # Process ELI documents if available
        if data.eli_documents:
            self.logger.info("Processing ELI documents text")
            documents = data.eli_documents
            processed_documents = []

            for document in documents:
                # Convert LegalDocument to dict and extract content
                document_dict = document.model_dump()
                content = (
                    document.content if hasattr(document, "content") else document.title
                )
                processed_text = self.processor.clean_text(content)
                document_dict["processed_content"] = processed_text
                processed_documents.append(ProcessedDocument(**document_dict))

            processed_eli_documents = processed_documents

        return TextProcessingData(
            processed_sejm_proceedings=processed_sejm_proceedings,
            processed_eli_documents=processed_eli_documents,
            sejm_proceedings=data.sejm_proceedings,
            eli_documents=data.eli_documents,
            session_id=data.session_id,
            date_range=data.date_range,
            category=data.category,
            document_ids=data.document_ids,
        )


class EmbeddingGenerationStep(PipelineStep):
    """Pipeline step for generating embeddings."""

    def __init__(self):
        super().__init__("embedding_generation")
        self.generator = BagEmbeddingsGenerator()

    async def process(self, data: TextProcessingData) -> EmbeddingGenerationData:
        """Generate embeddings for processed text."""
        self.logger.info("Generating embeddings")

        sejm_proceedings_embeddings = None
        eli_documents_embeddings = None

        # Generate embeddings for Sejm proceedings
        if data.processed_sejm_proceedings:
            proceedings = data.processed_sejm_proceedings
            proceedings_with_embeddings = []
            total_proceedings = len(proceedings)

            self.logger.info(
                f"Generating embeddings for {total_proceedings} proceedings"
            )

            for i, proceeding in enumerate(proceedings):
                text = proceeding.processed_content
                if text:
                    self.logger.info(
                        f"Processing proceeding {i + 1}/{total_proceedings}"
                    )
                    embedding = self.generator.generate_bag_embedding(text)
                    proceeding_dict = proceeding.model_dump()
                    proceeding_dict["embedding"] = embedding
                    proceedings_with_embeddings.append(
                        DocumentWithEmbedding(**proceeding_dict)
                    )

            sejm_proceedings_embeddings = proceedings_with_embeddings
            self.logger.info(
                f"Completed embedding generation for {len(proceedings_with_embeddings)} proceedings"
            )

        # Generate embeddings for ELI documents
        if data.processed_eli_documents:
            documents = data.processed_eli_documents
            documents_with_embeddings = []

            for document in documents:
                text = document.processed_content
                if text:
                    embedding = self.generator.generate_bag_embedding(text)
                    document_dict = document.model_dump()
                    document_dict["embedding"] = embedding
                    documents_with_embeddings.append(
                        DocumentWithEmbedding(**document_dict)
                    )

            eli_documents_embeddings = documents_with_embeddings

        return EmbeddingGenerationData(
            sejm_proceedings_embeddings=sejm_proceedings_embeddings,
            eli_documents_embeddings=eli_documents_embeddings,
            processed_sejm_proceedings=data.processed_sejm_proceedings,
            processed_eli_documents=data.processed_eli_documents,
            sejm_proceedings=data.sejm_proceedings,
            eli_documents=data.eli_documents,
            session_id=data.session_id,
            date_range=data.date_range,
            category=data.category,
            document_ids=data.document_ids,
        )


class DatabaseStorageStep(PipelineStep):
    """Pipeline step for storing data in database."""

    def __init__(self):
        super().__init__("database_storage")
        from sejm_whiz.database import get_database_config
        db_config = get_database_config()
        self.db = DocumentOperations(db_config)
        self.vector_db = VectorDBOperations()

    async def process(self, data: EmbeddingGenerationData) -> DatabaseStorageData:
        """Store processed data in database."""
        self.logger.info("Storing data in database")

        stored_sejm_proceedings = None
        stored_eli_documents = None

        # Store Sejm proceedings with embeddings
        if data.sejm_proceedings_embeddings:
            proceedings = data.sejm_proceedings_embeddings
            stored_ids = []

            for proceeding in proceedings:
                # Create document record (returns UUID)
                proceeding_dict = proceeding.model_dump(exclude={"embedding"})
                proceeding_id = self.db.create_document(
                    title=proceeding_dict.get("title", "Sejm Proceeding"),
                    content=proceeding.processed_content,
                    document_type="sejm_proceeding",
                    eli_identifier=None,
                    source_url=proceeding_dict.get("url"),
                    metadata=proceeding_dict,
                )

                # Store embedding in vector database
                if proceeding.embedding:
                    self.vector_db.create_document_embedding(
                        document_id=proceeding_id,  # Now proceeding_id is already a UUID
                        embedding=proceeding.embedding.document_embedding.tolist(),
                        model_name="allegro/herbert-base-cased",
                        model_version="1.0",
                        embedding_method="bag_of_embeddings",
                        token_count=len(proceeding.embedding.tokens)
                        if hasattr(proceeding.embedding, "tokens")
                        else None,
                    )

                stored_ids.append(proceeding_id)

            stored_sejm_proceedings = stored_ids

        # Store ELI documents with embeddings
        if data.eli_documents_embeddings:
            documents = data.eli_documents_embeddings
            stored_ids = []

            for document in documents:
                # Create document record (returns UUID)
                document_dict = document.model_dump(exclude={"embedding"})
                document_id = self.db.create_document(
                    title=document_dict.get("title", "ELI Document"),
                    content=document.processed_content,
                    document_type="eli_document",
                    eli_identifier=document_dict.get("eli_identifier"),
                    source_url=document_dict.get("url"),
                    metadata=document_dict,
                )

                # Store embedding in vector database
                if document.embedding:
                    self.vector_db.create_document_embedding(
                        document_id=document_id,  # Now document_id is already a UUID
                        embedding=document.embedding.document_embedding.tolist(),
                        model_name="allegro/herbert-base-cased",
                        model_version="1.0",
                        embedding_method="bag_of_embeddings",
                        token_count=len(document.embedding.tokens)
                        if hasattr(document.embedding, "tokens")
                        else None,
                    )

                stored_ids.append(document_id)

            stored_eli_documents = stored_ids

        return DatabaseStorageData(
            stored_sejm_proceedings=stored_sejm_proceedings,
            stored_eli_documents=stored_eli_documents,
            sejm_proceedings_embeddings=data.sejm_proceedings_embeddings,
            eli_documents_embeddings=data.eli_documents_embeddings,
            processed_sejm_proceedings=data.processed_sejm_proceedings,
            processed_eli_documents=data.processed_eli_documents,
            sejm_proceedings=data.sejm_proceedings,
            eli_documents=data.eli_documents,
            session_id=data.session_id,
            date_range=data.date_range,
            category=data.category,
            document_ids=data.document_ids,
        )


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
        # Run complete ingestion pipeline (both Sejm and ELI)
        pipeline = await create_full_ingestion_pipeline()

        # Sample input data
        from .models import DateRange

        input_data = PipelineInput(
            session_id="10",
            date_range=DateRange(start="2024-01-01", end="2024-01-31"),
            category="ustawa",  # For ELI documents
        )

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
            PipelineInput(
                session_id="10",
                date_range=DateRange(start="2024-01-01", end="2024-01-31"),
                category="ustawa",
            ),
            PipelineInput(
                session_id="11",
                date_range=DateRange(start="2024-02-01", end="2024-02-28"),
                category="rozporządzenie",
            ),
            PipelineInput(
                session_id="12",
                date_range=DateRange(start="2024-03-01", end="2024-03-31"),
                category="ustawa",
            ),
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
