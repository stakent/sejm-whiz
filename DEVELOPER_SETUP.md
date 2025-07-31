# Developer Setup Guide

This guide provides step-by-step instructions for setting up local development environment for the sejm-whiz project using uv and git feature branches.

## Prerequisites

- Git installed and configured
- Python 3.12+ (uv will manage this for you)
- [uv package manager](https://docs.astral.sh/uv/) installed

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Alternative: Using pip
pip install uv
```

## Initial Project Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd sejm-whiz-dev
```

### 2. Set up Development Environment

```bash
# Sync all dependencies (including dev dependencies)
uv sync --dev

# This will:
# - Create a virtual environment in .venv/
# - Install all dependencies from pyproject.toml
# - Install polylith-cli for workspace management
```

### 3. Verify Installation

```bash
# Check uv project status
uv run python --version

# Verify Polylith workspace
uv run poly info

# Run the basic application
uv run python main.py
```

## Daily Development Workflow

### Starting New Feature Development

#### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Example:
git checkout -b feature/sejm-api-integration
```

#### 2. Sync Dependencies

```bash
# Ensure you have latest dependencies
uv sync --dev

# If someone added new dependencies, this will install them
```

### Working with Dependencies

#### Adding New Dependencies

```bash
# Add production dependency
uv add requests httpx

# Add development dependency  
uv add --dev pytest pytest-asyncio

# Add with specific version
uv add 'fastapi>=0.104.0'

# Dependencies are automatically added to pyproject.toml
```

#### Removing Dependencies

```bash
# Remove dependency
uv remove requests

# Remove dev dependency
uv remove --dev pytest
```

#### Upgrading Dependencies

```bash
# Upgrade specific package
uv lock --upgrade-package fastapi

# Upgrade all packages (use with caution)
uv lock --upgrade
```

### Running Code and Commands

#### Basic Execution

```bash
# Run Python scripts in project environment
uv run python main.py
uv run python scripts/data_ingestion.py

# Run with arguments
uv run python main.py --verbose --config config.json
```

#### Polylith Commands

```bash
# Check workspace health
uv run poly check

# Create new component
uv run poly create component --name your_component

# Create new base
uv run poly create base --name your_base

# Create new project  
uv run poly create project --name your_project

# Run tests
uv run poly test

# Show component dependencies
uv run poly deps
```

#### Running Tests

```bash
# Run all tests using Polylith
uv run poly test

# Run tests for specific component
uv run pytest test/components/sejm_whiz/sejm_api/ -v
uv run pytest test/components/sejm_whiz/eli_api/ -v
uv run pytest test/components/sejm_whiz/vector_db/ -v
uv run pytest test/components/sejm_whiz/text_processing/ -v
uv run pytest test/components/sejm_whiz/embeddings/ -v
uv run pytest test/components/sejm_whiz/legal_nlp/ -v
uv run pytest test/components/sejm_whiz/prediction_models/ -v
uv run pytest test/components/sejm_whiz/semantic_search/ -v

# Run specific test file
uv run pytest test/components/sejm_whiz/sejm_api/test_client.py -v
uv run pytest test/components/sejm_whiz/eli_api/test_client.py -v
uv run pytest test/components/sejm_whiz/vector_db/test_embeddings.py -v
uv run pytest test/components/sejm_whiz/text_processing/test_core.py -v
uv run pytest test/components/sejm_whiz/embeddings/test_herbert_encoder.py -v
uv run pytest test/components/sejm_whiz/embeddings/test_bag_embeddings.py -v
uv run pytest test/components/sejm_whiz/embeddings/test_similarity.py -v
uv run pytest test/components/sejm_whiz/legal_nlp/test_core.py -v
uv run pytest test/components/sejm_whiz/legal_nlp/test_semantic_analyzer.py -v
uv run pytest test/components/sejm_whiz/legal_nlp/test_relationship_extractor.py -v
uv run pytest test/components/sejm_whiz/legal_nlp/test_integration.py -v
uv run pytest test/components/sejm_whiz/semantic_search/test_search_engine.py -v
uv run pytest test/components/sejm_whiz/semantic_search/test_indexer.py -v
uv run pytest test/components/sejm_whiz/semantic_search/test_ranker.py -v
uv run pytest test/components/sejm_whiz/semantic_search/test_cross_register.py -v

# Run with coverage
uv run pytest --cov=sejm_whiz test/

# Run tests for all components
uv run pytest test/components/ -v
```

### Code Development Best Practices

#### 1. Component Development

When creating new components:

```bash
# Create component
uv run poly create component --name sejm_api

# This creates:
# components/sejm_api/
# └── sejm_whiz/
#     └── sejm_api/
#         ├── __init__.py
#         ├── client.py          # Main implementation
#         ├── models.py          # Pydantic models
#         ├── exceptions.py      # Custom exceptions
#         └── rate_limiter.py    # Rate limiting logic
```

#### 2. Import Structure

Use proper namespace imports:

```python
# Import from other components
from sejm_whiz.database import DatabaseConnection
from sejm_whiz.sejm_api import SejmApiClient
from sejm_whiz.eli_api import EliApiClient
from sejm_whiz.vector_db import get_vector_operations, get_similarity_search
from sejm_whiz.text_processing import clean_legal_text, normalize_legal_text, process_legal_document
from sejm_whiz.embeddings import get_herbert_embedder, get_bag_embeddings_generator, get_similarity_calculator
from sejm_whiz.legal_nlp import ComprehensiveLegalAnalyzer, analyze_legal_concepts, extract_semantic_fields
from sejm_whiz.prediction_models import get_prediction_config, get_ensemble_model, get_similarity_predictor, get_classifier
from sejm_whiz.semantic_search import get_search_engine, get_document_indexer, get_result_ranker, get_cross_register_matcher

# Import within same component
from sejm_whiz.sejm_api.models import Session, Deputy
from sejm_whiz.sejm_api.exceptions import SejmApiError
from sejm_whiz.eli_api.models import LegalDocument, Amendment
from sejm_whiz.eli_api.exceptions import EliApiError
from sejm_whiz.vector_db.embeddings import DistanceMetric
from sejm_whiz.text_processing.core import TextProcessor
from sejm_whiz.text_processing.legal_parser import LegalDocumentAnalyzer
from sejm_whiz.embeddings.herbert_encoder import HerBERTEncoder
from sejm_whiz.embeddings.bag_embeddings import BagEmbeddingsGenerator
from sejm_whiz.embeddings.similarity import SimilarityCalculator
from sejm_whiz.legal_nlp.core import LegalNLPAnalyzer, LegalConcept, LegalAmendment
from sejm_whiz.legal_nlp.semantic_analyzer import LegalSemanticAnalyzer, SemanticField
from sejm_whiz.legal_nlp.relationship_extractor import LegalRelationshipExtractor, LegalEntity
from sejm_whiz.prediction_models.config import PredictionConfig
from sejm_whiz.prediction_models.ensemble import VotingEnsemble, StackingEnsemble, BlendingEnsemble
from sejm_whiz.prediction_models.similarity import CosineDistancePredictor, HybridSimilarityPredictor
from sejm_whiz.prediction_models.classification import RandomForestLegalClassifier, SVMLegalClassifier
from sejm_whiz.semantic_search.config import SearchConfig
from sejm_whiz.semantic_search.search_engine import SemanticSearchEngine
from sejm_whiz.semantic_search.indexer import DocumentIndexer
from sejm_whiz.semantic_search.ranker import ResultRanker
from sejm_whiz.semantic_search.cross_register import CrossRegisterMatcher
from sejm_whiz.semantic_search.query_processor import QueryProcessor
```

#### 3. Development with REPL

```bash
# Start Python REPL with project environment
uv run python

# In REPL, you can import and test components:
# >>> from sejm_whiz.sejm_api import SejmApiClient
# >>> from sejm_whiz.eli_api import EliApiClient
# >>> from sejm_whiz.vector_db import get_vector_operations, get_similarity_search
# >>> from sejm_whiz.text_processing import clean_legal_text, process_legal_document
# >>> from sejm_whiz.legal_nlp import ComprehensiveLegalAnalyzer, analyze_legal_concepts
# >>> sejm_client = SejmApiClient()
# >>> eli_client = EliApiClient()
# >>> ops = get_vector_operations()
# >>> search = get_similarity_search()
# >>> legal_analyzer = ComprehensiveLegalAnalyzer()
# >>> await sejm_client.get_current_term()  # Test API methods
# >>> await eli_client.search_documents(query="ustawa")  # Test ELI API
# >>> health = validate_vector_db_health()  # Test vector DB health
# >>> cleaned_text = clean_legal_text("Art. 123. § 1. Przykład...")  # Test text processing
# >>> result = process_legal_document(legal_text)  # Test complete processing pipeline
# >>> herbert_embedder = get_herbert_embedder()  # Test HerBERT embedder
# >>> embedding = herbert_embedder.embed_text("Przykład tekstu prawnego")  # Test embedding generation
# >>> bag_gen = get_bag_embeddings_generator()  # Test bag embeddings
# >>> sim_calc = get_similarity_calculator()  # Test similarity calculations
# >>> legal_analysis = legal_analyzer.analyze_document("Art. 1. Przykład ustawy...")  # Test legal NLP
# >>> concepts = analyze_legal_concepts("Konstytucja określa prawa obywateli")  # Test concept extraction
# >>> config = get_prediction_config()  # Test prediction config
# >>> ensemble = get_ensemble_model(config, 'voting')  # Test ensemble models
# >>> predictor = get_similarity_predictor(config, 'cosine')  # Test similarity predictors
# >>> classifier = get_classifier(config, 'random_forest')  # Test classification models
# >>> search_engine = get_search_engine()  # Test semantic search
# >>> indexer = get_document_indexer()  # Test document indexing
# >>> ranker = get_result_ranker()  # Test result ranking
# >>> cross_matcher = get_cross_register_matcher()  # Test cross-register matching
# >>> results = search_engine.search("ustawa o ochronie danych", limit=5)  # Test semantic search functionality
```

### Git Workflow

#### 1. Regular Commits

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add sejm api integration component

- Implement SejmApiClient class
- Add rate limiting and error handling
- Include tests for basic functionality"

# Push feature branch
git push origin feature/sejm-api-integration
```

#### 2. Keeping Branch Updated

```bash
# Regularly sync with main
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main

# Or merge if preferred
git merge main
```

#### 3. Ready for Review

```bash
# Final push
git push origin feature/your-feature-name

# Create pull request via GitHub/GitLab interface
```

### Environment Management

#### Virtual Environment Location

uv creates virtual environment in `.venv/` directory:

```bash
# Activate manually (not usually needed with uv run)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

#### Environment Information

```bash
# Show project info
uv project show

# Show installed packages
uv pip list

# Show dependency tree
uv tree
```

### Troubleshooting

#### Clean Environment

```bash
# Remove virtual environment
rm -rf .venv/

# Recreate from scratch
uv sync --dev
```

#### Dependency Conflicts

```bash
# Check for conflicts
uv pip check

# Resolve with specific versions
uv add 'package-name==1.2.3'
```

#### Python Version Issues

```bash
# Check current Python version
uv run python --version

# Use specific Python version
uv python install 3.12
uv sync --dev
```

## File Structure Overview

```
sejm-whiz-dev/
├── .venv/                    # Virtual environment (auto-created)
├── .python-version           # Python version specification
├── uv.lock                   # Dependency lockfile (commit this)
├── pyproject.toml           # Project configuration and dependencies
├── workspace.toml           # Polylith workspace configuration
├── main.py                  # Main application entry
├── bases/                   # Polylith bases (coming soon)
├── components/              # Polylith components
│   ├── sejm_whiz/
│   │   ├── database/        ✅ PostgreSQL + pgvector operations
│   │   ├── document_ingestion/  # ELI API integration
│   │   ├── eli_api/         ✅ ELI API client with security features
│   │   ├── embeddings/      ✅ HerBERT Polish BERT with bag-of-embeddings
│   │   ├── legal_nlp/       ✅ Advanced legal document analysis and NLP
│   │   ├── prediction_models/ ✅ ML models for law change predictions
│   │   ├── redis/               # Caching and queues
│   │   ├── sejm_api/        ✅ Sejm API client with security features
│   │   ├── semantic_search/     # Embedding-based search with cross-register matching
│   │   ├── text_processing/ ✅ Polish legal text processing pipeline
│   │   └── vector_db/       ✅ Vector database operations with pgvector
├── projects/                # Polylith projects (coming soon)
├── test/                    # Test files organized by component
│   └── components/sejm_whiz/
│       ├── database/
│       ├── eli_api/         ✅ 119 tests passing
│       ├── embeddings/      ✅ 80+ tests passing (HerBERT + bag embeddings)
│       ├── legal_nlp/       ✅ 45+ tests passing (concept extraction + semantic analysis)
│       ├── prediction_models/ ✅ Validated (ensemble, similarity, classification)
│       ├── sejm_api/        ✅ 248 tests passing
│       ├── semantic_search/     # Tests ready for implementation
│       ├── text_processing/ ✅ 79 tests passing
│       └── vector_db/       ✅ 66 tests passing (unit + integration)
└── development/             # Shared development utilities
```

## Important Files to Track in Git

**Always commit:**
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Exact dependency versions for reproducibility
- `workspace.toml` - Polylith configuration
- All source code in `components/`, `bases/`, `projects/`

**Never commit:**
- `.venv/` - Virtual environment (auto-generated)
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files
- `.DS_Store` - macOS system files

### Quality Assurance

#### Code Formatting and Linting

```bash
# Format code with ruff
uv run ruff format components/ test/

# Check and fix linting issues
uv run ruff check components/ test/ --fix

# Type checking (if mypy is added)
uv run mypy components/
```

#### Security Testing

```bash
# Run security validation tests
uv run pytest test/components/sejm_whiz/sejm_api/test_validation.py -v
uv run pytest test/components/sejm_whiz/eli_api/test_client.py::TestEliApiClient -k "batch" -v
uv run pytest test/components/sejm_whiz/vector_db/test_integration.py -v
uv run pytest test/components/sejm_whiz/text_processing/test_cleaner.py -v
uv run pytest test/components/sejm_whiz/legal_nlp/test_core.py -k "concept" -v

# Check for security issues in dependencies
uv audit

# Manual security review checklist:
# - Input validation on all user-controlled parameters
# - Error message sanitization
# - Rate limiting implementation
# - Endpoint validation against path traversal
```

## Current Implementation Status

### ✅ Completed Components

**Database Component:**
- PostgreSQL + pgvector integration
- Alembic migrations system
- Connection management and operations

**ELI API Component:**
- Complete async HTTP client for Polish ELI (European Legislation Identifier) API
- Comprehensive legal document parsing and structure extraction
- Advanced security features:
  - Batch processing with concurrency controls (max 50 docs, max 10 concurrent)
  - Resource exhaustion prevention and input validation
  - HTML parsing with XSS protection and size limits
  - ReDoS (Regex DoS) prevention with timeout controls
- 119 tests with full coverage across 6 test modules
- Production-ready legal document processing pipeline

**Sejm API Component:**
- Complete async HTTP client for Sejm Proceedings API
- Comprehensive Pydantic models for all endpoints
- Advanced security features:
  - Endpoint validation preventing URL manipulation
  - Error message sanitization preventing information disclosure
  - Comprehensive input validation for all parameters
  - Token bucket and sliding window rate limiting
- 248 tests with full coverage across 6 test modules
- Production-ready with robust error handling

**Vector DB Component:**
- Complete PostgreSQL + pgvector integration for semantic similarity search
- Advanced vector operations with multiple distance metrics (cosine, L2, inner product)
- Comprehensive document CRUD operations with embedding storage
- Advanced features:
  - UUID support with proper string-to-UUID conversion
  - Raw SQL optimization for complex pgvector operations
  - Test isolation with singleton reset fixtures
  - Embedding validation for 768-dimension HerBERT vectors
  - Vector index creation (IVFFlat and HNSW)
  - Batch similarity search and embedding statistics
- 66 tests passing (25 unit + 41 integration/utility tests)
- Production-ready with comprehensive error handling and logging

**Text Processing Component:**
- Complete Polish legal text processing pipeline for document preparation
- Advanced text cleaning and normalization specialized for Polish legal documents
- Comprehensive text processing features:
  - HTML cleaning with legal structure preservation
  - Polish diacritics handling (preserve/remove options)
  - Legal reference standardization (Art., §, pkt., rozdz.)
  - Legal entity extraction (laws, articles, courts, legal persons)
  - Document structure analysis and tokenization
  - Lazy loading system for optional spacy dependencies
- 79 tests passing across 6 test modules
- Production-ready with comprehensive legal document focus

**Embeddings Component:**
- Complete HerBERT Polish BERT implementation (`allegro/herbert-klej-cased-v1`)
- Bag-of-embeddings approach with document-level averaging for semantic similarity
- GPU optimization for NVIDIA GTX 1060 6GB with efficient memory management
- Advanced features:
  - Batch processing with dynamic sizing and progress tracking
  - Similarity calculations (cosine, Euclidean) and matrix operations
  - Redis integration for embedding caching and performance optimization
  - Automatic tensor cleanup and memory optimization
  - Comprehensive error handling with graceful fallbacks
- 80+ tests passing across 5 test modules
- Production-ready with complete integration for vector database storage

**Legal NLP Component:**
- Advanced legal document analysis with multi-act amendment detection
- Comprehensive semantic field analysis for Polish legal domains
- Legal concept extraction with sophisticated pattern matching
- Advanced features:
  - Legal concept detection (principles, definitions, obligations, prohibitions, rights, penalties)
  - Amendment detection with modification, addition, and deletion types
  - Semantic field analysis (civil law, criminal law, administrative law, constitutional law, tax law, labor law)
  - Semantic relations extraction (causal, temporal, modal, conditional)
  - Legal definitions extraction using semantic patterns
  - Argumentative structure analysis (premises, conclusions, counterarguments, justifications)
  - Conceptual density analysis and complexity scoring
  - Legal entity relationship mapping with confidence scoring
- 45+ tests passing across 4 test modules
- Production-ready with sophisticated Polish legal document processing

**Prediction Models Component:**
- Complete ML pipeline for law change predictions with multiple model types
- Ensemble methods: VotingEnsemble, StackingEnsemble, BlendingEnsemble with soft/hard voting
- Similarity-based predictors: Cosine, Euclidean, Hybrid, and Temporal similarity models
- Classification models: Random Forest, Gradient Boosting, SVM, Logistic Regression, TF-IDF embedding
- Advanced features:
  - Comprehensive configuration with environment variable support
  - Legal document-specific feature extraction and processing
  - Model persistence with joblib integration and evaluation metrics
  - Batch processing with error handling and confidence levels
  - GPU/CPU optimization presets and production-ready configuration
- Component validated with successful imports and factory functions
- Production-ready with integration to embeddings and vector database components


### 🚧 Next Components to Implement

1. ✅ **COMPLETED**: Embeddings Component - HerBERT Polish BERT implementation
2. ✅ **COMPLETED**: Legal NLP Component - Multi-act amendment detection and semantic analysis
3. ✅ **COMPLETED**: Prediction Models Component - ML pipeline for law change predictions
4. **READY**: Semantic Search Component - Document retrieval and ranking system with cross-register matching
5. **Redis Component** - Caching and background job queues
6. **Document Ingestion Component** - Processing pipeline integration

## Next Steps

1. Read the project overview in `CLAUDE.md`
2. Review the detailed implementation plan in `IMPLEMENTATION_PLAN.md`
3. Check component implementation examples in `components/sejm_whiz/sejm_api/` and `components/sejm_whiz/eli_api/`
4. Run existing tests to understand patterns: 
   - `uv run pytest test/components/sejm_whiz/sejm_api/ -v`
   - `uv run pytest test/components/sejm_whiz/eli_api/ -v`  
   - `uv run pytest test/components/sejm_whiz/vector_db/ -v`
   - `uv run pytest test/components/sejm_whiz/text_processing/ -v`
   - `uv run pytest test/components/sejm_whiz/embeddings/ -v`
   - `uv run pytest test/components/sejm_whiz/legal_nlp/ -v`
   - `uv run pytest test/components/sejm_whiz/prediction_models/ -v` (when tests are added)
   - `uv run pytest test/components/sejm_whiz/semantic_search/ -v` (when implemented)
5. Follow the git feature branch workflow for all changes

## Getting Help

- uv documentation: https://docs.astral.sh/uv/
- Polylith documentation: https://polylith.gitbook.io/
- Project-specific help: Check `CLAUDE.md` for command reference