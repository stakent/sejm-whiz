# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Planning Documents

- See `IMPLEMENTATION_PLAN.md` Phase 7-8 for deployment architecture and multi-cloud strategy

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
  - **Web Dashboard**: Real-time monitoring interface at `/dashboard`
    - Live log streaming with Server-Sent Events (SSE)
    - Pipeline status monitoring and document count tracking
    - Interactive controls for log management
    - Kubernetes pod log integration with local fallback
- `data_processor` - Batch processing project for data ingestion

### Not Yet Implemented

- `legal_graph` - Legal act dependency mapping (planned)
- `user_preferences` - User interest profiling (planned)
- `notification_system` - Multi-channel notifications (planned)
- `ml_inference` - Model inference base (planned)
- `model_trainer` - ML training project (planned)

The workspace is configured for:

- Python 3.12+ requirement
- GPU optimization for NVIDIA GTX 1060 6GB
- PostgreSQL 17 with pgvector extension for vector storage
- Bag of embeddings approach using HerBERT (Polish BERT)
- Multi-act amendment detection and omnibus legislation analysis

## Current State

**📋 [View detailed component status and progress → IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**

The project has advanced significantly from initial setup with 11 components, 2 bases, and 2 projects implemented. The core system is functional with comprehensive testing infrastructure, database operations, and deployment configurations.

For detailed implementation status, deployment assessment, and development roadmap, see the comprehensive status tracking in IMPLEMENTATION_PLAN.md.

## Deployment Configuration

### p7 Server Details

- **Project Directory**: `/root/tmp/sejm-whiz` (NOT `/tmp/sejm-whiz`)
- **Docker Compose File**: `docker-compose.dev-p7.yml`
- **API Port**: 8001 (mapped from container port 8000)
- **Database Port**: 5433 (mapped from container port 5432)

### Deployment Commands

```bash
# Deploy to p7
ssh root@p7 "cd /root/tmp/sejm-whiz && docker compose -f docker-compose.dev-p7.yml down"
# Copy updated files first, then:
ssh root@p7 "cd /root/tmp/sejm-whiz && docker compose -f docker-compose.dev-p7.yml up -d"
```
