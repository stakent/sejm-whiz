# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Goal**: Predict changes in Polish law using data from Sejm (Polish Parliament) APIs
- **ELI API**: Effective law data from https://api.sejm.gov.pl/eli/openapi/
- **Sejm Proceedings API**: Parliamentary proceedings from https://api.sejm.gov.pl/sejm/openapi/

This is a Python project structured as a Polylith workspace implementing an AI-driven legal prediction system using bag of embeddings for semantic similarity. The system will monitor parliamentary proceedings and legal documents to predict future law changes with multi-act amendment detection and cross-reference analysis. Currently in initial setup phase.

## Key Commands

### Package Management
- `uv sync --dev` - Install all dependencies including dev dependencies (polylith-cli)
- `uv run python main.py` - Run the main application
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency

### Polylith Workspace Management
- `uv run poly info` - Show workspace summary (components, bases, projects)
- `uv run poly check` - Validate the Polylith workspace
- `uv run poly sync` - Update pyproject.toml with missing bricks
- `uv run poly create component <name>` - Create a new component
- `uv run poly create base <name>` - Create a new base
- `uv run poly create project <name>` - Create a new project
- `uv run poly test` - Run tests across the workspace
- `uv run poly deps` - Visualize dependencies between bricks
- `uv run poly build` - Build packages

### Testing
- `uv run poly test` - Run tests using Polylith's test system (enabled in workspace.toml)

## Code Quality and Formatting
- Format and lint all Python files using ruff.

## Architecture

This project follows the Polylith architecture pattern with planned components:

- **Namespace**: `sejm_whiz` - All code should use this Python namespace
- **Structure Theme**: "loose" - Allows flexible organization of code

### Implemented Components
- `sejm_api` - Sejm Proceedings API integration with rate limiting
- `eli_api` - ELI API integration for legal documents with parsing utilities
- `text_processing` - Text cleaning, legal parsing, normalization, and tokenization
- `embeddings` - Bag of embeddings with HerBERT encoder and batch processing
- `vector_db` - PostgreSQL + pgvector operations with migrations
- `legal_nlp` - Legal document analysis with relationship extraction and semantic analysis
- `prediction_models` - ML models with classification, ensemble methods, and similarity
- `semantic_search` - Embedding-based search with cross-register functionality
- `database` - Database operations with Alembic migrations and secure operations
- `document_ingestion` - Document ingestion pipeline with text processing
- `redis` - Redis cache and queue operations

### Implemented Bases
- `web_api` - FastAPI web server base
- `data_pipeline` - Data processing base

### Implemented Projects
- `api_server` - Main web API project with FastAPI
- `data_processor` - Batch processing project for data ingestion

### Not Yet Implemented
- `legal_graph` - Legal act dependency mapping (planned)
- `user_preferences` - User interest profiling (planned)
- `notification_system` - Multi-channel notifications (planned)
- `dashboard` - Interactive visualization (planned)
- `ml_inference` - Model inference base (planned)
- `model_trainer` - ML training project (planned)

The workspace is configured for:
- Python 3.12+ requirement
- GPU optimization for NVIDIA GTX 1060 6GB
- PostgreSQL 17 with pgvector extension for vector storage
- Bag of embeddings approach using HerBERT (Polish BERT)
- Multi-act amendment detection and omnibus legislation analysis

## Current State

The project has advanced significantly from initial setup:

### Completed Implementation
- **11 Components**: All core functionality components implemented including APIs, text processing, embeddings, vector database, NLP analysis, and prediction models
- **2 Bases**: Web API and data pipeline bases are implemented
- **2 Projects**: API server and data processor projects are configured and functional
- **Database Schema**: Alembic migrations set up with legal document models
- **Testing Infrastructure**: Comprehensive test suite across all components
- **Docker & Kubernetes**: Deployment configurations with Helm charts
- **Dependencies**: Full dependency stack including PyTorch, Transformers, PostgreSQL, Redis

### Key Features Implemented
- Legal document ingestion from ELI API
- Sejm proceedings data processing with rate limiting
- HerBERT-based embeddings for Polish legal text
- Vector similarity search with pgvector
- Legal NLP analysis with relationship extraction
- Multi-model prediction ensemble
- Redis caching and queue management
- Secure database operations

### Next Steps
The core system is functional. Remaining work includes UI components (dashboard), user management (preferences, notifications), advanced legal graph analysis, and ML training workflows.