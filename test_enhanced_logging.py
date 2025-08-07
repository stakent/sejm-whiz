#!/usr/bin/env python3
"""Test script to demonstrate enhanced logging capabilities."""

import asyncio
import logging
from sejm_whiz.logging import setup_enhanced_logging, get_enhanced_logger
from sejm_whiz.eli_api.client import EliApiClient


async def main():
    """Test enhanced logging with ELI API client."""
    # Setup enhanced logging with file paths and timestamps
    setup_enhanced_logging(level=logging.INFO, include_source=True)

    logger = get_enhanced_logger("test_enhanced_logging")
    logger.info("Testing enhanced logging with ELI API client")

    # Test the enhanced error logging
    client = EliApiClient()

    try:
        # This will trigger the enhanced error logging we implemented
        # The error message will now show exactly where it comes from
        result = await client.search_documents(query="test", limit=5)
        logger.info(
            f"Search completed successfully. Found {len(result.documents)} documents"
        )

        # If we get documents with parsing errors, we'll see enhanced error messages
        logger.info("Enhanced logging test completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error during test: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
