# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Semantic versioning support with commitizen and bumpversion
- Code complexity analysis and refactoring roadmap
- Comprehensive code metrics with radon, mypy, bandit

## [0.1.0] - 2025-08-06

### Added

- Initial implementation of Polylith workspace architecture
- Core components: sejm_api, eli_api, text_processing, embeddings
- Vector database operations with PostgreSQL + pgvector
- Legal NLP processing with HerBERT embeddings
- Prediction models with ensemble methods
- Semantic search with cross-register functionality
- FastAPI web server with real-time dashboard
- Data processing pipeline with batch operations
- Redis caching and queue operations
- Database migrations with Alembic
- Document ingestion pipeline
- Comprehensive testing infrastructure
- Docker containerization and Kubernetes deployment configs
- Multi-cloud deployment support (p7 server, k3s)

### Development

- Polylith workspace with 11 components, 2 bases, 2 projects
- Pre-commit hooks with ruff formatting and linting
- GitHub Actions CI/CD pipeline
- Comprehensive developer setup documentation
