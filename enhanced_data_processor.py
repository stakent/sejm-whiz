#!/usr/bin/env python3
"""Enhanced data processor for comprehensive document ingestion."""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from sejm_whiz.data_pipeline.core import (
    DataPipeline,
    PipelineStep,
    DataPipelineError,
)
from sejm_whiz.sejm_api import SejmApiClient
from sejm_whiz.eli_api import EliApiClient, DocumentType
from sejm_whiz.text_processing import TextProcessor
from sejm_whiz.embeddings import BagEmbeddingsGenerator
from sejm_whiz.vector_db import VectorDBOperations
from sejm_whiz.database import DocumentOperations


class ComprehensiveSejmDataStep(PipelineStep):
    """Enhanced Sejm data ingestion with multiple document types."""

    def __init__(self):
        super().__init__("comprehensive_sejm_ingestion")
        self.client = SejmApiClient()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch comprehensive Sejm data including multiple document types."""
        self.logger.info("Starting comprehensive Sejm data ingestion")

        # Get current term
        current_term = await self.client.get_current_term()
        terms_to_process = data.get("terms", [current_term])

        all_documents: Dict[str, List] = {
            "proceedings": [],
            "votings": [],
            "interpellations": [],
            "committee_sittings": [],
            "deputy_info": [],
        }

        for term in terms_to_process:
            self.logger.info(f"Processing term {term}")

            # 1. Get proceeding sittings (parliamentary sessions)
            try:
                proceedings = await self.client.get_proceeding_sittings(term=term)
                # Limit to prevent overwhelming processing
                limit = data.get("proceedings_limit", 20)
                proceedings = (
                    proceedings[:limit] if len(proceedings) > limit else proceedings
                )
                all_documents["proceedings"].extend(proceedings)
                self.logger.info(
                    f"Retrieved {len(proceedings)} proceedings for term {term}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get proceedings for term {term}: {e}")

            # 2. Get voting data
            try:
                # Get recent votings (last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                votings = await self.client.get_votings(
                    term=term,
                    since=start_date.strftime("%Y-%m-%d"),
                    until=end_date.strftime("%Y-%m-%d"),
                )
                voting_limit = data.get("votings_limit", 50)
                votings = (
                    votings[:voting_limit] if len(votings) > voting_limit else votings
                )
                all_documents["votings"].extend(votings)
                self.logger.info(f"Retrieved {len(votings)} votings for term {term}")
            except Exception as e:
                self.logger.warning(f"Failed to get votings for term {term}: {e}")

            # 3. Get interpellations (parliamentary questions)
            try:
                interpellations = await self.client.get_interpellations(term=term)
                interp_limit = data.get("interpellations_limit", 30)
                interpellations = (
                    interpellations[:interp_limit]
                    if len(interpellations) > interp_limit
                    else interpellations
                )
                all_documents["interpellations"].extend(interpellations)
                self.logger.info(
                    f"Retrieved {len(interpellations)} interpellations for term {term}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to get interpellations for term {term}: {e}"
                )

            # 4. Get committee sittings
            try:
                # Get recent committee sittings
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                committee_sittings = await self.client.get_committee_sittings_by_date(
                    date_from=start_date.strftime("%Y-%m-%d"),
                    date_to=end_date.strftime("%Y-%m-%d"),
                )
                committee_limit = data.get("committee_limit", 25)
                committee_sittings = (
                    committee_sittings[:committee_limit]
                    if len(committee_sittings) > committee_limit
                    else committee_sittings
                )
                all_documents["committee_sittings"].extend(committee_sittings)
                self.logger.info(
                    f"Retrieved {len(committee_sittings)} committee sittings"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get committee sittings: {e}")

        self.logger.info(
            f"Comprehensive Sejm ingestion completed. Total documents: "
            f"proceedings={len(all_documents['proceedings'])}, "
            f"votings={len(all_documents['votings'])}, "
            f"interpellations={len(all_documents['interpellations'])}, "
            f"committee_sittings={len(all_documents['committee_sittings'])}"
        )

        return {
            **data,
            "comprehensive_sejm_data": all_documents,
            "step_completed": "comprehensive_sejm_ingestion",
        }


class ComprehensiveELIDataStep(PipelineStep):
    """Enhanced ELI data ingestion with all document types."""

    def __init__(self):
        super().__init__("comprehensive_eli_ingestion")
        self.client = EliApiClient()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch comprehensive ELI legal documents of all types."""
        self.logger.info("Starting comprehensive ELI data ingestion")

        all_legal_documents = []

        # Get all document types
        document_types = [
            DocumentType.USTAWA,  # Acts/Laws
            DocumentType.ROZPORZADZENIE,  # Regulations
            DocumentType.KODEKS,  # Codes
            DocumentType.KONSTYTUCJA,  # Constitution
            DocumentType.DEKRET,  # Decrees
            DocumentType.UCHWALA,  # Resolutions
        ]

        # Date range for document collection
        days_back = data.get("eli_days_back", 90)  # Last 3 months by default
        date_from = datetime.now() - timedelta(days=days_back)

        for doc_type in document_types:
            try:
                self.logger.info(f"Fetching {doc_type.value} documents")

                # Search for documents of this type
                result = await self.client.search_documents(
                    document_type=doc_type.value,
                    date_from=date_from,
                    limit=data.get("eli_limit_per_type", 50),
                )

                documents = result.documents if hasattr(result, "documents") else result
                all_legal_documents.extend(documents)

                self.logger.info(
                    f"Retrieved {len(documents)} {doc_type.value} documents"
                )

                # Get recent documents of this type
                try:
                    recent_docs = await self.client.get_recent_documents(
                        days=30, document_types=[doc_type.value]
                    )
                    # Avoid duplicates by checking ELI IDs
                    existing_eli_ids = {doc.eli_id for doc in all_legal_documents}
                    new_docs = [
                        doc for doc in recent_docs if doc.eli_id not in existing_eli_ids
                    ]
                    all_legal_documents.extend(new_docs)

                    self.logger.info(
                        f"Added {len(new_docs)} recent {doc_type.value} documents"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to get recent {doc_type.value} documents: {e}"
                    )

            except Exception as e:
                self.logger.warning(f"Failed to get {doc_type.value} documents: {e}")

        # Also get document amendments for comprehensive coverage
        amendment_documents = []
        try:
            # Get sample of documents to check for amendments
            sample_docs = all_legal_documents[:10]  # Check first 10 documents
            for doc in sample_docs:
                try:
                    amendments = await self.client.get_document_amendments(doc.eli_id)
                    amendment_documents.extend(amendments)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get amendments for {doc.eli_id}: {e}"
                    )

            self.logger.info(
                f"Retrieved {len(amendment_documents)} amendment documents"
            )
        except Exception as e:
            self.logger.warning(f"Failed to process amendments: {e}")

        self.logger.info(
            f"Comprehensive ELI ingestion completed. "
            f"Total legal documents: {len(all_legal_documents)}, "
            f"amendments: {len(amendment_documents)}"
        )

        return {
            **data,
            "comprehensive_eli_documents": all_legal_documents,
            "eli_amendments": amendment_documents,
            "step_completed": "comprehensive_eli_ingestion",
        }


class EnhancedTextProcessingStep(PipelineStep):
    """Enhanced text processing for all document types."""

    def __init__(self):
        super().__init__("enhanced_text_processing")
        self.processor = TextProcessor()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text from all document types."""
        self.logger.info("Starting enhanced text processing")

        processed_data = {}

        # Process comprehensive Sejm data
        if "comprehensive_sejm_data" in data:
            sejm_data = data["comprehensive_sejm_data"]

            # Process proceedings
            if sejm_data.get("proceedings"):
                proceedings = sejm_data["proceedings"]
                processed_proceedings = []

                for proceeding in proceedings:
                    content = (
                        getattr(proceeding, "agenda", "")
                        or getattr(proceeding, "description", "")
                        or ""
                    )
                    if content:
                        processed_text = self.processor.clean_text(content)
                        proceeding_dict = proceeding.model_dump()
                        proceeding_dict["processed_content"] = processed_text
                        proceeding_dict["document_type"] = "sejm_proceeding"
                        processed_proceedings.append(proceeding_dict)

                processed_data["processed_proceedings"] = processed_proceedings
                self.logger.info(f"Processed {len(processed_proceedings)} proceedings")

            # Process votings
            if sejm_data.get("votings"):
                votings = sejm_data["votings"]
                processed_votings = []

                for voting in votings:
                    content = f"Voting: {getattr(voting, 'title', '')} - {getattr(voting, 'description', '')}"
                    processed_text = self.processor.clean_text(content)
                    voting_dict = voting.model_dump()
                    voting_dict["processed_content"] = processed_text
                    voting_dict["document_type"] = "sejm_voting"
                    processed_votings.append(voting_dict)

                processed_data["processed_votings"] = processed_votings
                self.logger.info(f"Processed {len(processed_votings)} votings")

            # Process interpellations
            if sejm_data.get("interpellations"):
                interpellations = sejm_data["interpellations"]
                processed_interpellations = []

                for interpellation in interpellations:
                    content = f"{getattr(interpellation, 'title', '')}\n{getattr(interpellation, 'body', '')}"
                    processed_text = self.processor.clean_text(content)
                    interp_dict = interpellation.model_dump()
                    interp_dict["processed_content"] = processed_text
                    interp_dict["document_type"] = "sejm_interpellation"
                    processed_interpellations.append(interp_dict)

                processed_data["processed_interpellations"] = processed_interpellations
                self.logger.info(
                    f"Processed {len(processed_interpellations)} interpellations"
                )

            # Process committee sittings
            if sejm_data.get("committee_sittings"):
                committee_sittings = sejm_data["committee_sittings"]
                processed_committee_sittings = []

                for sitting in committee_sittings:
                    content = f"Committee: {getattr(sitting, 'committee', '')} - {getattr(sitting, 'agenda', '')}"
                    processed_text = self.processor.clean_text(content)
                    sitting_dict = sitting.model_dump()
                    sitting_dict["processed_content"] = processed_text
                    sitting_dict["document_type"] = "committee_sitting"
                    processed_committee_sittings.append(sitting_dict)

                processed_data["processed_committee_sittings"] = (
                    processed_committee_sittings
                )
                self.logger.info(
                    f"Processed {len(processed_committee_sittings)} committee sittings"
                )

        # Process comprehensive ELI documents
        if "comprehensive_eli_documents" in data:
            documents = data["comprehensive_eli_documents"]
            processed_documents = []

            for document in documents:
                content = getattr(document, "content", "") or document.title
                processed_text = self.processor.clean_text(content)
                document_dict = document.model_dump()
                document_dict["processed_content"] = processed_text
                document_dict["document_type"] = f"eli_{document.document_type.value}"
                processed_documents.append(document_dict)

            processed_data["processed_eli_documents"] = processed_documents
            self.logger.info(f"Processed {len(processed_documents)} ELI documents")

        # Process amendments
        if "eli_amendments" in data:
            amendments = data["eli_amendments"]
            processed_amendments = []

            for amendment in amendments:
                content = getattr(amendment, "content", "") or getattr(
                    amendment, "description", ""
                )
                processed_text = self.processor.clean_text(content)
                amendment_dict = amendment.model_dump()
                amendment_dict["processed_content"] = processed_text
                amendment_dict["document_type"] = "eli_amendment"
                processed_amendments.append(amendment_dict)

            processed_data["processed_amendments"] = processed_amendments
            self.logger.info(f"Processed {len(processed_amendments)} amendments")

        total_processed = sum(
            len(v) for v in processed_data.values() if isinstance(v, list)
        )
        self.logger.info(
            f"Enhanced text processing completed. Total documents processed: {total_processed}"
        )

        return {**data, **processed_data, "step_completed": "enhanced_text_processing"}


class EnhancedEmbeddingGenerationStep(PipelineStep):
    """Enhanced embedding generation for all processed documents."""

    def __init__(self):
        super().__init__("enhanced_embedding_generation")
        self.generator = BagEmbeddingsGenerator()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for all processed documents."""
        self.logger.info("Starting enhanced embedding generation")

        embeddings_data = {}
        total_embeddings = 0

        # Process all document types
        document_collections = [
            ("processed_proceedings", "proceedings_embeddings"),
            ("processed_votings", "votings_embeddings"),
            ("processed_interpellations", "interpellations_embeddings"),
            ("processed_committee_sittings", "committee_sittings_embeddings"),
            ("processed_eli_documents", "eli_documents_embeddings"),
            ("processed_amendments", "amendments_embeddings"),
        ]

        for source_key, target_key in document_collections:
            if source_key in data:
                documents = data[source_key]
                documents_with_embeddings = []

                self.logger.info(
                    f"Generating embeddings for {len(documents)} {source_key}"
                )

                for i, document in enumerate(documents):
                    text = document.get("processed_content", "")
                    if text:
                        try:
                            embedding = self.generator.generate_bag_embedding(text)
                            documents_with_embeddings.append(
                                {**document, "embedding": embedding}
                            )
                            if (i + 1) % 10 == 0 or i == len(documents) - 1:
                                self.logger.info(
                                    f"Generated embeddings for {i + 1}/{len(documents)} {source_key}"
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to generate embedding for document {i}: {e}"
                            )

                embeddings_data[target_key] = documents_with_embeddings
                total_embeddings += len(documents_with_embeddings)
                self.logger.info(
                    f"Completed {len(documents_with_embeddings)} embeddings for {source_key}"
                )

        self.logger.info(
            f"Enhanced embedding generation completed. Total embeddings: {total_embeddings}"
        )

        return {
            **data,
            **embeddings_data,
            "step_completed": "enhanced_embedding_generation",
        }


class EnhancedDatabaseStorageStep(PipelineStep):
    """Enhanced database storage for all document types."""

    def __init__(self):
        super().__init__("enhanced_database_storage")
        self.db = DocumentOperations()
        self.vector_db = VectorDBOperations()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store all processed documents and embeddings in database."""
        self.logger.info("Starting enhanced database storage")

        storage_results = {}
        total_stored = 0

        # Store all document types with embeddings
        embedding_collections = [
            ("proceedings_embeddings", "sejm_proceeding"),
            ("votings_embeddings", "sejm_voting"),
            ("interpellations_embeddings", "sejm_interpellation"),
            ("committee_sittings_embeddings", "committee_sitting"),
            ("eli_documents_embeddings", None),  # Use document's own type
            ("amendments_embeddings", "eli_amendment"),
        ]

        for collection_key, default_doc_type in embedding_collections:
            if collection_key in data:
                documents = data[collection_key]
                stored_ids = []

                self.logger.info(f"Storing {len(documents)} {collection_key}")

                for i, document in enumerate(documents):
                    try:
                        # Determine document type
                        doc_type = document.get("document_type", default_doc_type)

                        # Create document record
                        document_id = self.db.create_document(
                            title=document.get("title", f"Document {i+1}"),
                            content=document.get("processed_content", ""),
                            document_type=doc_type,
                            eli_identifier=document.get("eli_identifier")
                            or document.get("eli_id"),
                            source_url=document.get("url")
                            or document.get("source_url"),
                            metadata=document,
                        )

                        # Store embedding if available
                        if "embedding" in document:
                            self.vector_db.create_document_embedding(
                                document_id=document_id,
                                embedding=document[
                                    "embedding"
                                ].document_embedding.tolist(),
                                model_name="allegro/herbert-base-cased",
                                model_version="1.0",
                                embedding_method="bag_of_embeddings",
                                token_count=len(document["embedding"].tokens)
                                if hasattr(document["embedding"], "tokens")
                                else None,
                            )

                        stored_ids.append(document_id)

                        if (i + 1) % 10 == 0 or i == len(documents) - 1:
                            self.logger.info(
                                f"Stored {i + 1}/{len(documents)} {collection_key}"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to store document {i} from {collection_key}: {e}"
                        )

                storage_results[f"stored_{collection_key}"] = stored_ids
                total_stored += len(stored_ids)

        self.logger.info(
            f"Enhanced database storage completed. Total documents stored: {total_stored}"
        )

        return {
            **data,
            **storage_results,
            "step_completed": "enhanced_database_storage",
        }


async def create_comprehensive_pipeline() -> DataPipeline:
    """Create a comprehensive data ingestion pipeline for all document types."""
    steps = [
        ComprehensiveSejmDataStep(),
        ComprehensiveELIDataStep(),
        EnhancedTextProcessingStep(),
        EnhancedEmbeddingGenerationStep(),
        EnhancedDatabaseStorageStep(),
    ]

    return DataPipeline("comprehensive_ingestion", steps)


async def main():
    """Main entry point for enhanced data processor."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("enhanced_data_processor")
    logger.info("Starting enhanced comprehensive data processor")

    try:
        # Create comprehensive pipeline
        pipeline = await create_comprehensive_pipeline()

        # Configuration for comprehensive data collection
        input_data = {
            "terms": [10],  # Current term (adjust as needed)
            "proceedings_limit": 50,  # Max proceedings per term
            "votings_limit": 100,  # Max votings per term
            "interpellations_limit": 75,  # Max interpellations per term
            "committee_limit": 50,  # Max committee sittings
            "eli_days_back": 180,  # Last 6 months of legal documents
            "eli_limit_per_type": 100,  # Max documents per legal document type
        }

        logger.info("Starting comprehensive pipeline execution")
        logger.info(f"Configuration: {input_data}")

        # Run comprehensive pipeline
        result = await pipeline.run(input_data)

        # Calculate total documents processed
        total_docs = 0
        for key, value in result.items():
            if "stored_" in key and isinstance(value, list):
                total_docs += len(value)
                logger.info(f"{key}: {len(value)} documents")

        logger.info("Comprehensive pipeline completed successfully!")
        logger.info(f"Total documents stored in database: {total_docs}")

        # Print detailed metrics
        metrics = pipeline.get_metrics()
        logger.info(f"Pipeline metrics: {metrics}")

        # Summary of what was collected
        logger.info("=== COLLECTION SUMMARY ===")
        if "comprehensive_sejm_data" in result:
            sejm_data = result["comprehensive_sejm_data"]
            logger.info(f"Sejm proceedings: {len(sejm_data.get('proceedings', []))}")
            logger.info(f"Sejm votings: {len(sejm_data.get('votings', []))}")
            logger.info(
                f"Sejm interpellations: {len(sejm_data.get('interpellations', []))}"
            )
            logger.info(
                f"Committee sittings: {len(sejm_data.get('committee_sittings', []))}"
            )

        if "comprehensive_eli_documents" in result:
            logger.info(
                f"ELI legal documents: {len(result['comprehensive_eli_documents'])}"
            )

        if "eli_amendments" in result:
            logger.info(f"ELI amendments: {len(result['eli_amendments'])}")

        logger.info("=== PIPELINE COMPLETE - READY FOR SEMANTIC SEARCH ===")

    except DataPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    logger.info("Enhanced data processor completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
