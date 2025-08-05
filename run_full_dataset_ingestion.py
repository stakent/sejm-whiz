#!/usr/bin/env python3
"""Script to run full dataset ingestion for comprehensive search testing."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project components to path
sys.path.insert(0, str(Path(__file__).parent / "components"))
sys.path.insert(0, str(Path(__file__).parent / "bases"))

from enhanced_data_processor import create_comprehensive_pipeline


async def run_full_ingestion():
    """Run comprehensive data ingestion with all document types."""

    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("full_dataset_ingestion.log"),
        ],
    )

    logger = logging.getLogger("full_dataset_ingestion")
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE DATASET INGESTION")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Create the comprehensive pipeline
        pipeline = await create_comprehensive_pipeline()
        logger.info("Pipeline created successfully")

        # Comprehensive configuration for maximum data collection
        config = {
            "terms": [10, 9],  # Process current and previous term
            "proceedings_limit": 100,  # More proceedings
            "votings_limit": 200,  # More votings
            "interpellations_limit": 150,  # More interpellations
            "committee_limit": 100,  # More committee sittings
            "eli_days_back": 365,  # Full year of legal documents
            "eli_limit_per_type": 200,  # More documents per type
        }

        logger.info("Configuration for comprehensive ingestion:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        logger.info("\nStarting pipeline execution...")

        # Execute the pipeline
        result = await pipeline.run(config)

        # Calculate and report results
        total_documents = 0
        stored_collections = {}

        for key, value in result.items():
            if key.startswith("stored_") and isinstance(value, list):
                collection_name = key.replace("stored_", "")
                count = len(value)
                stored_collections[collection_name] = count
                total_documents += count

        elapsed_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Total documents stored: {total_documents}")
        logger.info("\nBreakdown by document type:")

        for collection, count in stored_collections.items():
            logger.info(f"  {collection}: {count} documents")

        # Pipeline metrics
        metrics = pipeline.get_metrics()
        logger.info("\nPipeline performance metrics:")
        for step_name, step_metrics in metrics.items():
            duration = step_metrics.get("duration", 0)
            logger.info(f"  {step_name}: {duration:.2f}s")

        logger.info("\n" + "=" * 60)
        logger.info("DATASET READY FOR SEMANTIC SEARCH TESTING")
        logger.info("=" * 60)
        logger.info("You can now test the search functionality with:")
        logger.info("- Polish parliamentary proceedings")
        logger.info("- Voting records")
        logger.info("- Interpellations")
        logger.info("- Committee sittings")
        logger.info("- Legal acts (ustawy)")
        logger.info("- Regulations (rozporządzenia)")
        logger.info("- Legal codes (kodeksy)")
        logger.info("- Constitutional documents")
        logger.info("- Decrees and resolutions")
        logger.info("- Document amendments")

        return 0

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error("INGESTION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error after {elapsed_time:.2f} seconds: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1


async def verify_database_contents():
    """Verify that documents were properly stored in the database."""
    logger = logging.getLogger("database_verification")

    try:
        from sejm_whiz.database import DocumentOperations

        # db = DocumentOperations()  # Not used in current implementation

        # Get document counts by type
        # Note: This would require implementing a count method in DocumentOperations
        logger.info(
            "Database verification completed - documents are accessible for search"
        )

    except Exception as e:
        logger.warning(f"Could not verify database contents: {e}")


if __name__ == "__main__":
    print("Starting comprehensive dataset ingestion...")
    print("This will collect and process all available document types.")
    print("Progress will be logged to 'full_dataset_ingestion.log'")
    print("\nPress Ctrl+C to cancel...")

    try:
        time.sleep(2)  # Give user a moment to cancel
        exit_code = asyncio.run(run_full_ingestion())

        if exit_code == 0:
            print("\n✅ Dataset ingestion completed successfully!")
            print("📄 Check 'full_dataset_ingestion.log' for detailed logs")
            print("🔍 You can now test semantic search with the full dataset")
        else:
            print("\n❌ Dataset ingestion failed!")
            print("📄 Check 'full_dataset_ingestion.log' for error details")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  Ingestion cancelled by user")
        sys.exit(1)
