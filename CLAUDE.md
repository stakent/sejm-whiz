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

### Planned Components
- `sejm_api` - Sejm Proceedings API integration
- `eli_api` - ELI API integration for legal documents
- `text_processing` - Text cleaning and preprocessing
- `embeddings` - Bag of embeddings document-level generation
- `vector_db` - PostgreSQL + pgvector operations
- `legal_nlp` - Legal document analysis with multi-act amendment detection
- `legal_graph` - Legal act dependency mapping and cross-reference analysis
- `prediction_models` - ML models for law change predictions
- `semantic_search` - Embedding-based search and similarity
- `user_preferences` - User interest profiling and subscription management
- `notification_system` - Multi-channel notification delivery
- `dashboard` - Interactive prediction visualization

### Planned Bases
- `web_api` - FastAPI web server base
- `data_pipeline` - Data processing base
- `ml_inference` - Model inference base

### Planned Projects
- `api_server` - Main web API combining web_api base with user-facing components
- `data_processor` - Batch processing combining data_pipeline base with ingestion components
- `model_trainer` - ML training and validation workflows

The workspace is configured for:
- Python 3.12+ requirement
- GPU optimization for NVIDIA GTX 1060 6GB
- PostgreSQL 17 with pgvector extension for vector storage
- Bag of embeddings approach using HerBERT (Polish BERT)
- Multi-act amendment detection and omnibus legislation analysis

## Current State

The project is in initial setup phase with only a basic main.py file. The Polylith workspace structure is configured but no components, bases, or projects have been created yet. Next step is to begin implementing the planned component architecture following the detailed implementation plan in sejm_whiz_plan.md.