# Data Processor

Batch processing project combining the data_pipeline base with ingestion components for processing Polish legal data.

## Overview

The data_processor project implements a configurable data pipeline system for batch processing of legal documents and parliamentary proceedings from Polish government APIs. It combines the data_pipeline base with various ingestion components to create flexible processing workflows.

## Features

- **Modular Pipeline Architecture**: Configurable steps that can be combined into different processing workflows
- **Batch Processing Support**: Process multiple items efficiently with configurable batch sizes
- **Error Handling and Metrics**: Comprehensive error handling with detailed execution metrics
- **Multiple Data Sources**: Support for both Sejm proceedings and ELI legal documents
- **Complete Processing Chain**: From data ingestion through text processing, embeddings, and database storage

## Pipeline Steps

### Available Pipeline Steps

1. **SejmDataIngestionStep**: Fetches parliamentary proceedings from Sejm API
2. **ELIDataIngestionStep**: Fetches legal documents from ELI API  
3. **TextProcessingStep**: Cleans and processes text content
4. **EmbeddingGenerationStep**: Generates embeddings using HerBERT
5. **DatabaseStorageStep**: Stores processed data and embeddings in databases

### Pre-configured Pipelines

- **Full Ingestion Pipeline**: Complete processing for both Sejm and ELI data
- **Sejm-Only Pipeline**: Processing pipeline for parliamentary proceedings only
- **ELI-Only Pipeline**: Processing pipeline for legal documents only

## Usage

### Basic Usage

```python
from data_processor.main import create_sejm_only_pipeline

# Create pipeline
pipeline = await create_sejm_only_pipeline()

# Process single item
input_data = {
    "session_id": "10",
    "date_range": {"start": "2024-01-01", "end": "2024-01-31"}
}

result = await pipeline.run(input_data)
```

### Batch Processing

```python
from data_processor.main import create_sejm_only_pipeline
from sejm_whiz.data_pipeline.core import BatchProcessor

# Create pipeline and batch processor
pipeline = await create_sejm_only_pipeline()
batch_processor = BatchProcessor(pipeline, batch_size=10)

# Process multiple items
batch_data = [
    {"session_id": "10", "date_range": {"start": "2024-01-01", "end": "2024-01-31"}},
    {"session_id": "11", "date_range": {"start": "2024-02-01", "end": "2024-02-28"}},
    # ... more items
]

results = await batch_processor.process_batch(batch_data)
```

### Running the Application

```bash
# Install dependencies
uv sync --dev

# Run the data processor
uv run python projects/data_processor/main.py
```

## Input Data Format

### Sejm Proceedings Input
```python
{
    "session_id": "10",                    # Session ID to process
    "date_range": {                        # Optional date filtering
        "start": "2024-01-01",
        "end": "2024-01-31"
    }
}
```

### ELI Documents Input
```python
{
    "document_ids": ["doc1", "doc2"],      # Specific document IDs
    "category": "ustawa"                   # Document category filter
}
```

## Output Data Format

Processed data includes:
- Original source data
- Processed text content
- Generated embeddings
- Database storage IDs
- Processing metadata

## Components Used

- **data_pipeline**: Base pipeline orchestration framework
- **document_ingestion**: Document processing utilities
- **sejm_api**: Sejm API client for parliamentary data
- **eli_api**: ELI API client for legal documents
- **text_processing**: Text cleaning and preprocessing
- **embeddings**: HerBERT-based embedding generation
- **vector_db**: PostgreSQL + pgvector storage
- **database**: Main database operations
- **redis**: Caching and session management

## Error Handling

The pipeline includes comprehensive error handling:
- Individual step validation (input/output)
- Pipeline-level error recovery
- Detailed logging and metrics collection
- Graceful handling of partial batch failures

## Metrics

Each pipeline execution provides metrics:
- Total runs, successes, and failures
- Execution timing information
- Last run and success timestamps
- Per-step performance data

## Configuration

The pipeline can be customized by:
- Adding or removing pipeline steps
- Configuring step parameters
- Adjusting batch processing settings
- Modifying logging levels and formats