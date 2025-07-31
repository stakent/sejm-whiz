# Sejm-Whiz: Polish Legal Change Prediction System

**Goal**: Predict changes in Polish law using data from Sejm (Polish Parliament) APIs
- **ELI API**: Effective law data from https://api.sejm.gov.pl/eli/openapi/
- **Sejm Proceedings API**: Parliamentary proceedings from https://api.sejm.gov.pl/sejm/openapi/

## Project Overview

This is a Python project structured as a Polylith workspace implementing an AI-driven legal prediction system using bag of embeddings for semantic similarity. The system monitors parliamentary proceedings and legal documents to predict future law changes with multi-act amendment detection and cross-reference analysis.

## Key Features

- **Multi-Act Amendment Detection**: Identifies complex omnibus legislation and cascading legal changes
- **Cross-Reference Analysis**: Maps relationships between legal acts and their dependencies
- **Semantic Search**: Uses HerBERT (Polish BERT) with bag of embeddings for document-level similarity
- **Real-Time Predictions**: Monitors parliamentary proceedings for early change indicators
- **User Interest Profiling**: Personalized notifications based on legal domain preferences
- **GPU-Optimized Inference**: Local processing on NVIDIA GTX 1060 6GB

## Architecture

Follows Polylith architecture pattern with planned components:

### Components

**✅ Implemented:**
- `database` - PostgreSQL + pgvector operations with Alembic migrations
- `eli_api` - ELI API integration with comprehensive legal document parsing, batch processing controls, and security features
- `sejm_api` - Sejm Proceedings API integration with comprehensive validation, rate limiting, and security features
- `vector_db` - Vector database operations with pgvector for semantic similarity search and embeddings storage
- `text_processing` - Polish legal text processing with cleaning, normalization, tokenization, and entity extraction
- `embeddings` - HerBERT embeddings with comprehensive Polish BERT implementation, bag-of-embeddings approach, batch processing, similarity calculations, and GPU optimization
- `redis` - Caching and queue management for background tasks and embedding operations
- `prediction_models` - ML models for law change predictions with ensemble methods, similarity-based predictors, and classification models

**🚧 Planned:**
- `document_ingestion` - Document processing pipeline and ingestion workflows
- `legal_nlp` - Legal document analysis with multi-act amendment detection
- `legal_graph` - Legal act dependency mapping and cross-reference analysis
- `semantic_search` - Embedding-based search and similarity
- `user_preferences` - User interest profiling and subscription management
- `notification_system` - Multi-channel notification delivery
- `dashboard` - Interactive prediction visualization

### Bases
- `web_api` - FastAPI web server base
- `data_pipeline` - Data processing base
- `ml_inference` - Model inference base

### Projects
- `api_server` - Main web API combining web_api base with user-facing components
- `data_processor` - Batch processing combining data_pipeline base with ingestion components
- `model_trainer` - ML training and validation workflows

## Technology Stack

- **Language**: Python 3.12+
- **Architecture**: Polylith monorepo with components and projects
- **Package Management**: uv with polylith-cli
- **Web Framework**: FastAPI with async support
- **Database**: PostgreSQL 17 with pgvector extension
- **Cache**: Redis 7+
- **ML Framework**: PyTorch with CUDA support
- **Embedding Models**: HerBERT (Polish BERT)
- **Orchestration**: k3s (single-node Kubernetes) with Helm charts
- **Container**: Docker with NVIDIA Container Toolkit

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Check workspace status**:
   ```bash
   uv run poly info
   ```

3. **Run tests**:
   ```bash
   uv run poly test
   ```

4. **Run database tests**:
   ```bash
   uv run python test_database.py
   ```

5. **Test embeddings system**:
   ```bash
   uv run pytest test/components/sejm_whiz/embeddings/ -v
   ```

6. **Test HerBERT embeddings**:
   ```bash
   # Test individual embedding components
   uv run pytest test/components/sejm_whiz/embeddings/test_herbert_encoder.py -v
   uv run pytest test/components/sejm_whiz/embeddings/test_bag_embeddings.py -v
   uv run pytest test/components/sejm_whiz/embeddings/test_similarity.py -v
   uv run pytest test/components/sejm_whiz/embeddings/test_batch_processor.py -v
   ```

7. **Deploy to k3s** (see `K3S_DEPLOYMENT.md` for full instructions):
   ```bash
   # Build and deploy containers
   docker build -t sejm-whiz-api:latest -f Dockerfile.api .
   docker build -t sejm-whiz-processor:latest -f Dockerfile.processor .
   ```

## Development Commands

### Package Management
- `uv sync --dev` - Install all dependencies including dev dependencies
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency

### Polylith Workspace Management
- `uv run poly info` - Show workspace summary
- `uv run poly check` - Validate the Polylith workspace
- `uv run poly sync` - Update pyproject.toml with missing bricks
- `uv run poly create component <name>` - Create a new component
- `uv run poly create base <name>` - Create a new base
- `uv run poly create project <name>` - Create a new project
- `uv run poly test` - Run tests across the workspace
- `uv run poly deps` - Visualize dependencies between bricks
- `uv run poly build` - Build packages

## Current State

**Phase 1 - Infrastructure Setup**: ✅ **COMPLETED**
- PostgreSQL database with pgvector extension configured
- Alembic migrations system in place
- Docker containerization with Dockerfile.api and Dockerfile.processor
- k3s deployment documentation ready

**Phase 2 - Core Components**: ✅ **COMPLETED**
- Database component implemented and functional ✅
- ELI API component completed with advanced legal document processing and security features ✅
- Sejm API component completed with comprehensive security features ✅
- Vector DB component completed with pgvector similarity search and embeddings storage ✅
- Text Processing component completed with Polish legal text processing pipeline ✅
- Embeddings component completed with comprehensive HerBERT Polish BERT implementation ✅
- Redis component completed with caching, queue management, and health monitoring ✅
- Prediction Models component completed with ensemble methods, similarity-based predictors, and classification models ✅

**Phase 3 - Advanced Components**: 🚧 **IN PROGRESS**
- Ready to implement document ingestion and legal NLP components
- Currently on `feature/legal_nlp` branch

See `IMPLEMENTATION_PLAN.md` for detailed development roadmap and `K3S_DEPLOYMENT.md` for deployment instructions.

## Hardware Requirements

- **GPU**: NVIDIA GeForce GTX 1060 6GB (minimum)
- **RAM**: 16GB+ recommended
- **Storage**: NVMe SSD for vector index performance
- **CUDA**: Version 11.8 or compatible

## Contributing

This project follows the Polylith architecture principles:
- Components should be small, reusable, and do one thing well
- Use the `sejm_whiz` namespace for all code
- Follow component isolation principles
- Test components independently using `poly test`

## License

[License information to be added]
