#!/usr/bin/env python3
"""Simple but comprehensive data ingestion script."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project components to path
sys.path.insert(0, str(Path(__file__).parent / "components"))
sys.path.insert(0, str(Path(__file__).parent / "bases"))

from sejm_whiz.data_pipeline.core import DataPipeline, PipelineStep
from sejm_whiz.sejm_api import SejmApiClient
from sejm_whiz.eli_api import EliApiClient
from sejm_whiz.text_processing import TextProcessor
from sejm_whiz.embeddings import BagEmbeddingsGenerator
from sejm_whiz.vector_db import VectorDBOperations
from sejm_whiz.database import DocumentOperations


class SimpleSejmIngestion(PipelineStep):
    """Simple Sejm data ingestion."""

    def __init__(self):
        super().__init__("simple_sejm_ingestion")
        self.client = SejmApiClient()

    async def process(self, data):
        """Fetch Sejm data with error handling."""
        self.logger.info("Starting Sejm data ingestion")

        all_documents = []

        # Use known term (10 is current)
        term = 10

        try:
            # Get proceedings
            proceedings = await self.client.get_proceeding_sittings(term=term)
            # Limit to reasonable number
            proceedings = proceedings[:30]

            for proc in proceedings:
                doc = {
                    "title": getattr(
                        proc, "title", f"Proceeding {getattr(proc, 'id', '')}"
                    ),
                    "content": getattr(proc, "agenda", "") or "",
                    "document_type": "sejm_proceeding",
                    "metadata": proc.model_dump(),
                    "source": "sejm_api",
                }
                all_documents.append(doc)

            self.logger.info(f"Retrieved {len(proceedings)} proceedings")

        except Exception as e:
            self.logger.warning(f"Failed to get proceedings: {e}")

        return {**data, "sejm_documents": all_documents}


class SimpleELIIngestion(PipelineStep):
    """Simple ELI data ingestion."""

    def __init__(self):
        super().__init__("simple_eli_ingestion")
        self.client = EliApiClient()

    async def process(self, data):
        """Fetch ELI documents with error handling."""
        self.logger.info("Starting ELI data ingestion")

        all_documents = []

        # Get different document types
        doc_types = ["ustawa", "rozporządzenie", "kodeks"]

        for doc_type in doc_types:
            try:
                # Get recent documents
                recent_docs = await self.client.get_recent_documents(
                    days=60, document_types=[doc_type]
                )

                # Limit per type
                recent_docs = recent_docs[:20]

                for eli_doc in recent_docs:
                    doc = {
                        "title": eli_doc.title,
                        "content": getattr(eli_doc, "content", "") or eli_doc.title,
                        "document_type": f"eli_{doc_type}",
                        "eli_id": eli_doc.eli_id,
                        "metadata": eli_doc.model_dump(),
                        "source": "eli_api",
                    }
                    all_documents.append(doc)

                self.logger.info(f"Retrieved {len(recent_docs)} {doc_type} documents")

            except Exception as e:
                self.logger.warning(f"Failed to get {doc_type} documents: {e}")

        return {**data, "eli_documents": all_documents}


class SimpleTextProcessing(PipelineStep):
    """Simple text processing."""

    def __init__(self):
        super().__init__("simple_text_processing")
        self.processor = TextProcessor()

    async def process(self, data):
        """Process all documents."""
        self.logger.info("Starting text processing")

        processed_docs = []

        # Process Sejm documents
        if "sejm_documents" in data:
            for doc in data["sejm_documents"]:
                if doc.get("content"):
                    processed_content = self.processor.clean_text(doc["content"])
                    doc["processed_content"] = processed_content
                    processed_docs.append(doc)

        # Process ELI documents
        if "eli_documents" in data:
            for doc in data["eli_documents"]:
                if doc.get("content"):
                    processed_content = self.processor.clean_text(doc["content"])
                    doc["processed_content"] = processed_content
                    processed_docs.append(doc)

        self.logger.info(f"Processed {len(processed_docs)} documents")
        return {**data, "processed_documents": processed_docs}


class SimpleEmbeddingGeneration(PipelineStep):
    """Simple embedding generation."""

    def __init__(self):
        super().__init__("simple_embedding_generation")
        self.generator = BagEmbeddingsGenerator()

    async def process(self, data):
        """Generate embeddings for all documents."""
        self.logger.info("Starting embedding generation")

        if "processed_documents" not in data:
            return data

        documents = data["processed_documents"]
        docs_with_embeddings = []

        for i, doc in enumerate(documents):
            if doc.get("processed_content"):
                try:
                    embedding = self.generator.generate_bag_embedding(
                        doc["processed_content"]
                    )
                    doc["embedding"] = embedding
                    docs_with_embeddings.append(doc)

                    if (i + 1) % 10 == 0:
                        self.logger.info(
                            f"Generated embeddings for {i + 1}/{len(documents)} documents"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate embedding for document {i}: {e}"
                    )

        self.logger.info(
            f"Generated embeddings for {len(docs_with_embeddings)} documents"
        )
        return {**data, "documents_with_embeddings": docs_with_embeddings}


class SimpleStorage(PipelineStep):
    """Simple database storage."""

    def __init__(self):
        super().__init__("simple_storage")
        from sejm_whiz.database import get_database_config
        db_config = get_database_config()
        self.db = DocumentOperations(db_config)
        self.vector_db = VectorDBOperations()

    async def process(self, data):
        """Store documents and embeddings."""
        self.logger.info("Starting database storage")

        if "documents_with_embeddings" not in data:
            return data

        documents = data["documents_with_embeddings"]
        stored_ids = []

        for i, doc in enumerate(documents):
            try:
                # Store document
                doc_id = self.db.create_document(
                    title=doc["title"],
                    content=doc.get("processed_content", ""),
                    document_type=doc["document_type"],
                    eli_identifier=doc.get("eli_id"),
                    source_url=None,
                    metadata=doc.get("metadata", {}),
                )

                # Store embedding
                if "embedding" in doc:
                    self.vector_db.create_document_embedding(
                        document_id=doc_id,
                        embedding=doc["embedding"].document_embedding.tolist(),
                        model_name="allegro/herbert-base-cased",
                        model_version="1.0",
                        embedding_method="bag_of_embeddings",
                        token_count=len(doc["embedding"].tokens)
                        if hasattr(doc["embedding"], "tokens")
                        else None,
                    )

                stored_ids.append(doc_id)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Stored {i + 1}/{len(documents)} documents")

            except Exception as e:
                self.logger.warning(f"Failed to store document {i}: {e}")

        self.logger.info(f"Stored {len(stored_ids)} documents in database")
        return {**data, "stored_document_ids": stored_ids}


async def main():
    """Main execution."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("simple_full_ingestion")
    logger.info("Starting simple full dataset ingestion")

    try:
        # Create pipeline
        steps = [
            SimpleSejmIngestion(),
            SimpleELIIngestion(),
            SimpleTextProcessing(),
            SimpleEmbeddingGeneration(),
            SimpleStorage(),
        ]

        pipeline = DataPipeline("simple_full_ingestion", steps)

        # Run pipeline
        result = await pipeline.run({"started_at": datetime.now()})

        # Report results
        total_docs = len(result.get("stored_document_ids", []))
        logger.info(f"SUCCESS: Stored {total_docs} documents with embeddings")
        logger.info("Database is ready for semantic search testing!")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
