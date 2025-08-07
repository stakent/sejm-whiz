#!/usr/bin/env python3
"""Comprehensive 2025 document ingestion for full dataset search testing."""

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


class Comprehensive2025SejmIngestion(PipelineStep):
    """Ingest ALL available 2025 Sejm documents."""

    def __init__(self):
        super().__init__("comprehensive_2025_sejm")
        self.client = SejmApiClient()

    async def process(self, data):
        """Fetch comprehensive 2025 Sejm data."""
        self.logger.info("Starting comprehensive 2025 Sejm data ingestion")

        all_documents = []

        # Current term (10)
        term = 10

        try:
            # 1. Get ALL proceeding sittings for 2025
            self.logger.info("Fetching ALL 2025 proceeding sittings...")
            proceedings = await self.client.get_proceeding_sittings(term=term)

            # Filter for 2025 documents
            proceedings_2025 = []
            for proc in proceedings:
                # Check if proceeding has 2025 date info
                proc_data = proc.model_dump()
                if any("2025" in str(value) for value in proc_data.values() if value):
                    proceedings_2025.append(proc)

            self.logger.info(f"Found {len(proceedings_2025)} 2025 proceedings")

            for proc in proceedings_2025:
                doc = {
                    "title": getattr(
                        proc, "title", f"Proceeding {getattr(proc, 'id', '')}"
                    ),
                    "content": getattr(proc, "agenda", "") or "",
                    "document_type": "sejm_proceeding_2025",
                    "metadata": proc.model_dump(),
                    "source": "sejm_api",
                    "year": 2025,
                }
                all_documents.append(doc)

            # 2. Get ALL 2025 votings
            self.logger.info("Fetching ALL 2025 votings...")
            try:
                votings = await self.client.get_votings(
                    term=term, since="2025-01-01", until="2025-12-31"
                )

                self.logger.info(f"Found {len(votings)} 2025 votings")

                for voting in votings:
                    content = f"Voting: {getattr(voting, 'title', '')} - {getattr(voting, 'description', '')}"
                    doc = {
                        "title": getattr(
                            voting, "title", f'Voting {getattr(voting, "id", "")}'
                        ),
                        "content": content,
                        "document_type": "sejm_voting_2025",
                        "metadata": voting.model_dump(),
                        "source": "sejm_api",
                        "year": 2025,
                    }
                    all_documents.append(doc)

            except Exception as e:
                self.logger.warning(f"Failed to get 2025 votings: {e}")

            # 3. Get ALL 2025 interpellations
            self.logger.info("Fetching ALL 2025 interpellations...")
            try:
                interpellations = await self.client.get_interpellations(term=term)

                # Filter for 2025
                interpellations_2025 = []
                for interp in interpellations:
                    interp_data = interp.model_dump()
                    if any(
                        "2025" in str(value) for value in interp_data.values() if value
                    ):
                        interpellations_2025.append(interp)

                self.logger.info(
                    f"Found {len(interpellations_2025)} 2025 interpellations"
                )

                for interp in interpellations_2025:
                    content = (
                        f"{getattr(interp, 'title', '')}\n{getattr(interp, 'body', '')}"
                    )
                    doc = {
                        "title": getattr(
                            interp,
                            "title",
                            f'Interpellation {getattr(interp, "id", "")}',
                        ),
                        "content": content,
                        "document_type": "sejm_interpellation_2025",
                        "metadata": interp.model_dump(),
                        "source": "sejm_api",
                        "year": 2025,
                    }
                    all_documents.append(doc)

            except Exception as e:
                self.logger.warning(f"Failed to get 2025 interpellations: {e}")

            # 4. Get ALL 2025 committee sittings
            self.logger.info("Fetching ALL 2025 committee sittings...")
            try:
                committee_sittings = await self.client.get_committee_sittings_by_date(
                    date_from="2025-01-01", date_to="2025-12-31"
                )

                self.logger.info(
                    f"Found {len(committee_sittings)} 2025 committee sittings"
                )

                for sitting in committee_sittings:
                    content = f"Committee: {getattr(sitting, 'committee', '')} - {getattr(sitting, 'agenda', '')}"
                    doc = {
                        "title": f"Committee {getattr(sitting, 'committee', '')} - {getattr(sitting, 'date', '')}",
                        "content": content,
                        "document_type": "committee_sitting_2025",
                        "metadata": sitting.model_dump(),
                        "source": "sejm_api",
                        "year": 2025,
                    }
                    all_documents.append(doc)

            except Exception as e:
                self.logger.warning(f"Failed to get 2025 committee sittings: {e}")

        except Exception as e:
            self.logger.error(f"Major error in Sejm ingestion: {e}")

        self.logger.info(
            f"Comprehensive 2025 Sejm ingestion completed: {len(all_documents)} documents"
        )
        return {**data, "sejm_2025_documents": all_documents}


class Comprehensive2025ELIIngestion(PipelineStep):
    """Ingest ALL available 2025 ELI legal documents."""

    def __init__(self):
        super().__init__("comprehensive_2025_eli")
        self.client = EliApiClient()

    async def process(self, data):
        """Fetch comprehensive 2025 ELI documents."""
        self.logger.info("Starting comprehensive 2025 ELI data ingestion")

        all_legal_documents = []

        # All document types
        document_types = [
            "ustawa",  # Acts/Laws
            "rozporządzenie",  # Regulations
            "kodeks",  # Codes
            "konstytucja",  # Constitution
            "dekret",  # Decrees
            "uchwała",  # Resolutions
        ]

        # 2025 date range
        date_from = datetime(2025, 1, 1)
        date_to = datetime.now()  # Up to current date

        for doc_type in document_types:
            try:
                self.logger.info(f"Fetching ALL 2025 {doc_type} documents...")

                # Search for all 2025 documents of this type
                result = await self.client.search_documents(
                    document_type=doc_type,
                    date_from=date_from,
                    date_to=date_to,
                    limit=1000,  # High limit to get everything
                )

                documents = result.documents if hasattr(result, "documents") else result

                for eli_doc in documents:
                    doc = {
                        "title": eli_doc.title,
                        "content": getattr(eli_doc, "content", "") or eli_doc.title,
                        "document_type": f"eli_{doc_type}_2025",
                        "eli_id": eli_doc.eli_id,
                        "metadata": eli_doc.model_dump(),
                        "source": "eli_api",
                        "year": 2025,
                    }
                    all_legal_documents.append(doc)

                self.logger.info(
                    f"Retrieved {len(documents)} {doc_type} documents for 2025"
                )

            except Exception as e:
                self.logger.warning(f"Failed to get 2025 {doc_type} documents: {e}")

        # Also get recent documents to ensure we have the latest
        try:
            self.logger.info("Fetching recent 2025 documents...")
            recent_docs = await self.client.get_recent_documents(
                days=365,  # All of 2025 so far
                document_types=document_types,
            )

            # Filter for 2025 and avoid duplicates
            existing_eli_ids = {doc["eli_id"] for doc in all_legal_documents}

            for doc in recent_docs:
                if (
                    doc.eli_id not in existing_eli_ids
                    and hasattr(doc, "published_date")
                    and doc.published_date
                    and doc.published_date.year == 2025
                ):
                    legal_doc = {
                        "title": doc.title,
                        "content": getattr(doc, "content", "") or doc.title,
                        "document_type": f"eli_{doc.document_type.value}_2025",
                        "eli_id": doc.eli_id,
                        "metadata": doc.model_dump(),
                        "source": "eli_api",
                        "year": 2025,
                    }
                    all_legal_documents.append(legal_doc)

            self.logger.info(
                f"Added recent documents, total now: {len(all_legal_documents)}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to get recent 2025 documents: {e}")

        self.logger.info(
            f"Comprehensive 2025 ELI ingestion completed: {len(all_legal_documents)} documents"
        )
        return {**data, "eli_2025_documents": all_legal_documents}


class Enhanced2025TextProcessing(PipelineStep):
    """Process all 2025 documents with enhanced metadata."""

    def __init__(self):
        super().__init__("enhanced_2025_text_processing")
        self.processor = TextProcessor()

    async def process(self, data):
        """Process all 2025 documents."""
        self.logger.info("Starting enhanced 2025 text processing")

        processed_docs = []

        # Process Sejm 2025 documents
        if "sejm_2025_documents" in data:
            self.logger.info(
                f"Processing {len(data['sejm_2025_documents'])} Sejm 2025 documents"
            )
            for doc in data["sejm_2025_documents"]:
                if doc.get("content"):
                    processed_content = self.processor.clean_text(doc["content"])
                    doc["processed_content"] = processed_content
                    doc["content_length"] = len(processed_content)
                    processed_docs.append(doc)

        # Process ELI 2025 documents
        if "eli_2025_documents" in data:
            self.logger.info(
                f"Processing {len(data['eli_2025_documents'])} ELI 2025 documents"
            )
            for doc in data["eli_2025_documents"]:
                if doc.get("content"):
                    processed_content = self.processor.clean_text(doc["content"])
                    doc["processed_content"] = processed_content
                    doc["content_length"] = len(processed_content)
                    processed_docs.append(doc)

        self.logger.info(
            f"Enhanced 2025 text processing completed: {len(processed_docs)} documents"
        )
        return {**data, "processed_2025_documents": processed_docs}


class Enhanced2025EmbeddingGeneration(PipelineStep):
    """Generate embeddings for all 2025 documents."""

    def __init__(self):
        super().__init__("enhanced_2025_embedding_generation")
        self.generator = BagEmbeddingsGenerator()

    async def process(self, data):
        """Generate embeddings for all 2025 documents."""
        self.logger.info("Starting enhanced 2025 embedding generation")

        if "processed_2025_documents" not in data:
            return data

        documents = data["processed_2025_documents"]
        docs_with_embeddings = []

        total_docs = len(documents)
        self.logger.info(f"Generating embeddings for {total_docs} 2025 documents")

        for i, doc in enumerate(documents):
            if doc.get("processed_content"):
                try:
                    embedding = self.generator.generate_bag_embedding(
                        doc["processed_content"]
                    )
                    doc["embedding"] = embedding
                    docs_with_embeddings.append(doc)

                    if (i + 1) % 50 == 0 or i == total_docs - 1:
                        self.logger.info(
                            f"Generated embeddings for {i + 1}/{total_docs} documents ({((i+1)/total_docs)*100:.1f}%)"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate embedding for document {i}: {e}"
                    )

        self.logger.info(
            f"Enhanced 2025 embedding generation completed: {len(docs_with_embeddings)} documents"
        )
        return {**data, "documents_2025_with_embeddings": docs_with_embeddings}


class Enhanced2025Storage(PipelineStep):
    """Store all 2025 documents and embeddings."""

    def __init__(self):
        super().__init__("enhanced_2025_storage")
        from sejm_whiz.database import get_database_config
        db_config = get_database_config()
        self.db = DocumentOperations(db_config)
        self.vector_db = VectorDBOperations()

    async def process(self, data):
        """Store all 2025 documents and embeddings."""
        self.logger.info("Starting enhanced 2025 database storage")

        if "documents_2025_with_embeddings" not in data:
            return data

        documents = data["documents_2025_with_embeddings"]
        stored_ids = []

        total_docs = len(documents)
        self.logger.info(f"Storing {total_docs} 2025 documents with embeddings")

        for i, doc in enumerate(documents):
            try:
                # Store document
                doc_id = self.db.create_document(
                    title=doc["title"],
                    content=doc.get("processed_content", ""),
                    document_type=doc["document_type"],
                    eli_identifier=doc.get("eli_id"),
                    source_url=None,
                    metadata={
                        **doc.get("metadata", {}),
                        "year": doc.get("year", 2025),
                        "source": doc.get("source", ""),
                        "content_length": doc.get("content_length", 0),
                    },
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

                if (i + 1) % 25 == 0 or i == total_docs - 1:
                    self.logger.info(
                        f"Stored {i + 1}/{total_docs} documents ({((i+1)/total_docs)*100:.1f}%)"
                    )

            except Exception as e:
                self.logger.warning(f"Failed to store document {i}: {e}")

        self.logger.info(
            f"Enhanced 2025 storage completed: {len(stored_ids)} documents stored"
        )
        return {**data, "stored_2025_document_ids": stored_ids}


async def main():
    """Main execution for comprehensive 2025 dataset ingestion."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("full_2025_ingestion.log"),
        ],
    )

    logger = logging.getLogger("full_2025_ingestion")
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE 2025 DATASET INGESTION")
    logger.info("=" * 80)
    logger.info("This will ingest ALL available 2025 documents from:")
    logger.info("- Sejm API: Proceedings, Votings, Interpellations, Committee Sittings")
    logger.info(
        "- ELI API: Laws, Regulations, Codes, Constitution, Decrees, Resolutions"
    )

    start_time = datetime.now()

    try:
        # Create comprehensive 2025 pipeline
        steps = [
            Comprehensive2025SejmIngestion(),
            Comprehensive2025ELIIngestion(),
            Enhanced2025TextProcessing(),
            Enhanced2025EmbeddingGeneration(),
            Enhanced2025Storage(),
        ]

        pipeline = DataPipeline("comprehensive_2025_ingestion", steps)

        # Run pipeline
        input_data = {"started_at": start_time}
        result = await pipeline.run(input_data)

        # Calculate results
        total_docs = len(result.get("stored_2025_document_ids", []))
        elapsed_time = datetime.now() - start_time

        logger.info("=" * 80)
        logger.info("2025 DATASET INGESTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total processing time: {elapsed_time}")
        logger.info(f"Total 2025 documents stored: {total_docs}")

        # Break down by source
        sejm_count = len(result.get("sejm_2025_documents", []))
        eli_count = len(result.get("eli_2025_documents", []))

        logger.info("\nBreakdown:")
        logger.info(f"  Sejm 2025 documents: {sejm_count}")
        logger.info(f"  ELI 2025 documents: {eli_count}")

        # Pipeline metrics
        metrics = pipeline.get_metrics()
        logger.info("\nPipeline performance:")
        for step_name, step_metrics in metrics.items():
            duration = step_metrics.get("duration", 0)
            logger.info(f"  {step_name}: {duration:.2f}s")

        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE 2025 DATASET READY FOR SEMANTIC SEARCH!")
        logger.info("=" * 80)
        logger.info("You can now search through:")
        logger.info("🏛️  All 2025 Polish parliamentary proceedings")
        logger.info("🗳️  All 2025 voting records")
        logger.info("❓ All 2025 interpellations")
        logger.info("🏢 All 2025 committee sittings")
        logger.info("⚖️  All 2025 legal acts and laws")
        logger.info("📋 All 2025 regulations and decrees")
        logger.info("📚 All 2025 legal codes")
        logger.info("🏛️  Constitutional documents")

        return 0

    except Exception as e:
        elapsed_time = datetime.now() - start_time
        logger.error("=" * 80)
        logger.error("2025 DATASET INGESTION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error after {elapsed_time}: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    print("🚀 Starting comprehensive 2025 dataset ingestion...")
    print("📊 This will collect ALL available 2025 documents from Sejm and ELI APIs")
    print("📝 Progress will be logged to 'full_2025_ingestion.log'")
    print("\nPress Ctrl+C to cancel...")

    try:
        import time

        time.sleep(3)  # Give user a moment to cancel
        exit_code = asyncio.run(main())

        if exit_code == 0:
            print("\n✅ 2025 dataset ingestion completed successfully!")
            print("📄 Check 'full_2025_ingestion.log' for detailed logs")
            print("🔍 You can now test semantic search with the complete 2025 dataset")
        else:
            print("\n❌ 2025 dataset ingestion failed!")
            print("📄 Check 'full_2025_ingestion.log' for error details")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  Ingestion cancelled by user")
        sys.exit(1)
