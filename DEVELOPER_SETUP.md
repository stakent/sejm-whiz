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
uv run pytest test/components/sejm_whiz/redis/ -v
uv run pytest test/components/sejm_whiz/document_ingestion/ -v

# Run tests for bases
uv run pytest test/bases/sejm_whiz/web_api/ -v

# Run API server
uv run python projects/api_server/main.py

# Or with uvicorn for development
uv run uvicorn projects.api_server.main:app --host 0.0.0.0 --port 8000 --reload

# Run data processor (batch processing pipeline)
uv run python projects/data_processor/main.py

# Run web UI monitoring dashboard
uv run python projects/web_ui/main.py

### Web UI Dashboard

The project includes a dedicated web monitoring dashboard for real-time pipeline monitoring:

```bash
# Start the web UI monitoring dashboard
uv run python projects/web_ui/main.py

# Access the web interface
# Home page: http://localhost:8000/
# Dashboard: http://localhost:8000/dashboard
# API Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

**Available Pages:**
- **🏠 Home** (`/` or `/home`): Landing page with project overview and feature descriptions
- **📊 Dashboard** (`/dashboard`): Real-time monitoring of data processor with live log streaming
- **📚 API Docs** (`/docs`): Interactive FastAPI/Swagger documentation with API testing
- **❤️ Health** (`/health`): System health status and service availability

**Dashboard Features:**
- **Fixed Top Navigation**: Easy access to all pages with visual active page indicators
- **Live Log Streaming**: Real-time logs from data processor with auto-scroll and color coding
- **Status Monitoring**: Current pipeline stage, document counts, and processor health
- **Interactive Controls**: Pause/resume streaming, clear logs, auto-scroll toggle
- **Fixed Container Height**: Logs scroll within fixed viewport without page layout expansion
- **Modern UI**: Gradient-styled interface with blur effects and responsive design
- **Automatic Reconnection**: Handles connection loss gracefully

**Web Interface API Endpoints:**
- `/` - Root endpoint (redirects to home page)
- `/home` - Landing page with project overview
- `/dashboard` - HTML dashboard interface
- `/api/logs/stream` - Server-Sent Events endpoint for real-time logs
- `/api/processor/status` - JSON status of data processor
- `/docs` - Interactive API documentation (Swagger UI)
- `/health` - Health check with structured response

**Technology Stack:**
- **Architecture**: Dedicated Polylith project (`projects/web_ui/`) using `web_api` base
- **Backend**: FastAPI with embedded HTML templates (no external dependencies)
- **Frontend**: Vanilla JavaScript with Server-Sent Events (SSE) for real-time updates
- **Styling**: Modern CSS with flexbox, gradients, and backdrop-filter effects
- **Log Sources**: Demo log generation with real-time streaming
- **Navigation**: Single-page application feel with fixed top navigation
- **Deployment**: Multi-stage Docker build with k3s deployment manifests

**Development Workflows:**

*Local Development:*
- Proper Polylith project structure using `web_api` base
- No external frontend dependencies required
- Dashboard works in both local development and Kubernetes environments
- For rapid iteration: `uv run uvicorn projects.web_ui.main:app --host 0.0.0.0 --port 8000 --reload`

*K3s Hot Reload Development:*
- **Speed**: ~5 seconds per change (eliminating 10+ minute Docker rebuilds)
- **URLs**: Production (http://192.168.0.200:30800/) and Development (http://192.168.0.200:30801/)
- **Workflow**:
  1. Edit files locally in `projects/web_ui/`
  2. Run: `./deployments/k3s/scripts/sync-web-ui.sh` (~1 second)
  3. Changes appear instantly via uvicorn auto-reload
- **Benefits**: Volume mounts eliminate container rebuilds, uvicorn hot-restarts on file changes
- **Key Files**:
  - `deployments/k3s/manifests/k3s-web-ui-deployment-dev.yaml` - Development deployment with volume mounts
  - `deployments/k3s/scripts/setup-web-ui-dev.sh` - One-time setup script
  - `deployments/k3s/scripts/sync-web-ui.sh` - Fast sync for changes

#### K3s Data Processor Hot Reload Scripts

For data processor development with Kubernetes hot reload (run from repo root directory):

**Configuration:**
The scripts use centralized configuration from `config.sh`:
```bash
# config.sh - Single source of truth for hot reload configuration
export MOUNT_PATH="/tmp/sejm-whiz"
export NAMESPACE="sejm-whiz"
export APP_LABEL="app=sejm-whiz-processor-gpu"
export WATCH_PATTERN="*.py$|*.html$|*.css$|*.js$|*.jinja2$|*.j2$"
```

**Manual Sync and Restart:**
```bash
# Using default configuration from config.sh
./sync-and-restart.sh

# Override specific settings via environment variables
MOUNT_PATH="/custom/path" ./sync-and-restart.sh
NAMESPACE="my-namespace" ./sync-and-restart.sh
```

**Automatic File Watching:**
```bash
# Watch files matching patterns in config.sh and auto-sync on changes
./watch-and-sync.sh

# Override watch pattern for specific use cases
WATCH_PATTERN="*.py$" ./watch-and-sync.sh  # Python only
WATCH_PATTERN="*" ./watch-and-sync.sh      # All files
```

**Configuration Parameters:**
- `REPO_PATH` - Local repository path (auto-detected via `pwd`)
- `MOUNT_PATH` - Remote mount/sync path (set in config.sh)
- `NAMESPACE` - Kubernetes namespace (set in config.sh)
- `APP_LABEL` - Pod selector label (set in config.sh)
- `WATCH_PATTERN` - File patterns to watch (set in config.sh, includes web UI files)

**Benefits:**
- **DRY Configuration**: Single config.sh file maintains all settings
- **Auto-Detection**: Repository path automatically detected from current directory
- **Web UI Support**: Watches Python, HTML, CSS, JS, and template files
- **Fast Iteration**: Automatic syncing with comprehensive file watching
- **Pod Management**: Automatic pod restart and log tailing
- **Flexible Overrides**: Environment variables can override config.sh settings

*Production Deployment:*
- Production-ready containerization with multi-stage Docker builds
- Separate production and development environments for proper service isolation

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
uv run pytest test/components/sejm_whiz/redis/test_connection.py -v
uv run pytest test/components/sejm_whiz/redis/test_cache.py -v
uv run pytest test/components/sejm_whiz/redis/test_queue.py -v
uv run pytest test/components/sejm_whiz/document_ingestion/test_pipeline.py -v
uv run pytest test/components/sejm_whiz/document_ingestion/test_processors.py -v
uv run pytest test/components/sejm_whiz/document_ingestion/test_workflows.py -v

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
# Import from bases
from sejm_whiz.web_api import create_app, get_app, HealthResponse, ErrorResponse

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
from sejm_whiz.redis import get_redis_client, get_cache_manager, get_queue_manager
from sejm_whiz.document_ingestion import get_ingestion_pipeline, get_document_processor

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
# >>> from sejm_whiz.web_api import create_app, get_app
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
# >>> app = get_app()  # Test web API creation
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
# >>> redis_client = get_redis_client()  # Test Redis connection
# >>> cache_manager = get_cache_manager()  # Test caching operations
# >>> queue_manager = get_queue_manager()  # Test background job queues
# >>> pipeline = get_ingestion_pipeline()  # Test document ingestion pipeline
# >>> processor = get_document_processor()  # Test document processing
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
├── bases/                   # Polylith bases
│   └── sejm_whiz/
│       ├── data_pipeline/   ✅ Data processing base with pipeline orchestration
│       └── web_api/         ✅ FastAPI web server base with comprehensive features
├── components/              # Polylith components
│   ├── sejm_whiz/
│   │   ├── database/        ✅ PostgreSQL + pgvector operations
│   │   ├── document_ingestion/ ✅ Document processing pipeline and ingestion workflows
│   │   ├── eli_api/         ✅ ELI API client with security features
│   │   ├── embeddings/      ✅ HerBERT Polish BERT with bag-of-embeddings
│   │   ├── legal_nlp/       ✅ Advanced legal document analysis and NLP
│   │   ├── prediction_models/ ✅ ML models for law change predictions
│   │   ├── redis/           ✅ Caching and background job queues
│   │   ├── sejm_api/        ✅ Sejm API client with security features
│   │   ├── semantic_search/ ✅ Embedding-based search with cross-register matching
│   │   ├── text_processing/ ✅ Polish legal text processing pipeline
│   │   └── vector_db/       ✅ Vector database operations with pgvector
├── projects/                # Polylith projects
│   ├── api_server/          ✅ Main web API server using web_api base
│   ├── data_processor/      ✅ Batch processing pipeline using data_pipeline base
│   └── web_ui/              ✅ Web monitoring dashboard using web_api base
├── test/                    # Test files organized by component
│   ├── bases/sejm_whiz/
│   │   ├── data_pipeline/   # Pipeline orchestration tests
│   │   └── web_api/         ✅ FastAPI base tests
│   └── components/sejm_whiz/
│       ├── database/
│       ├── eli_api/         ✅ 119 tests passing
│       ├── embeddings/      ✅ 80+ tests passing (HerBERT + bag embeddings)
│       ├── legal_nlp/       ✅ 45+ tests passing (concept extraction + semantic analysis)
│       ├── prediction_models/ ✅ Validated (ensemble, similarity, classification)
│       ├── redis/           ✅ 40+ tests passing (cache + queue operations)
│       ├── sejm_api/        ✅ 248 tests passing
│       ├── semantic_search/ ✅ 70+ tests passing (search engine + cross-register matching)
│       ├── text_processing/ ✅ 79 tests passing
│       ├── document_ingestion/ ✅ 50+ tests passing (pipeline + processors)
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

**Web API Base:**
- Complete FastAPI web server base implementation with production-ready features
- Comprehensive web interface with multiple pages and fixed top navigation
- Real-time monitoring dashboard with Server-Sent Events (SSE) log streaming
- Comprehensive error handling for all exception types with structured responses
- CORS middleware configuration with production considerations
- Multi-page web interface:
  - Home page (`/` or `/home`) with project overview and feature descriptions
  - Dashboard (`/dashboard`) with real-time data processor monitoring
  - API documentation (`/docs`) with interactive Swagger UI
  - Health check endpoint (`/health`) with structured JSON response
- Pydantic models for type-safe request/response handling (HealthResponse, ErrorResponse)
- Modern UI with gradient styling, fixed navigation, and responsive design
- Embedded HTML templates with no external dependencies for containerized deployment
- Logging integration for comprehensive error tracking and debugging
- Modular application factory pattern with `create_app()` and `get_app()` functions
- Ready for component integration and extensible route structure

**Data Pipeline Base:**
- Complete batch processing infrastructure with pipeline orchestration
- Modular step-based processing with error handling and recovery
- Progress tracking and metrics collection for large-scale operations
- Integration with all data processing components
- Pre-configured workflows for different data sources and processing types

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

**Semantic Search Component:**
- Complete semantic search pipeline with cross-register matching for legal vs parliamentary language
- HerBERT-powered embedding search with pgvector integration for fast similarity search
- Multi-factor relevance ranking combining semantic similarity, document metadata, and temporal relevance
- Query processing with legal term normalization and expansion for improved accuracy
- Advanced features:
  - Real-time search with caching and performance optimization
  - Legal domain awareness with specialized ranking for Polish legal system
  - Amendment tracking for legal change detection
  - Hybrid search combining semantic similarity and keyword matching
- 70+ tests passing across 7 comprehensive test modules
- Production-ready with seamless integration to all existing components

**Redis Component:**
- Complete Redis integration for caching and background job processing
- Connection management with automatic pooling and health monitoring
- Caching operations with TTL, JSON serialization, and batch operations
- Background job queue management with retry logic and error handling
- Advanced features:
  - Performance metrics and connection health checks
  - Task processing with configurable retry strategies
  - Memory-efficient operations with automatic cleanup
  - Integration with embeddings and document processing workflows
- 40+ tests passing across cache and queue functionality
- Production-ready with comprehensive error handling and logging

**Document Ingestion Component:**
- Complete document processing pipeline with advanced ingestion workflows
- Modular pipeline architecture with step-based processing and orchestration
- Individual processors for fetching, cleaning, embedding, and storage operations
- Pre-configured workflows for different document types (Sejm-only, ELI-only, Full ingestion)
- Advanced features:
  - Error recovery with comprehensive retry logic and partial failure recovery
  - Progress tracking with real-time processing metrics and monitoring
  - Batch processing optimization for large document collections
  - Integration with all existing components for end-to-end processing
- 50+ tests passing across pipeline and processor functionality
- Production-ready with seamless integration to data_processor project


### 🚧 Next Steps

**✅ Recently Completed:**
1. ✅ **COMPLETED**: Embeddings Component - HerBERT Polish BERT implementation
2. ✅ **COMPLETED**: Legal NLP Component - Multi-act amendment detection and semantic analysis
3. ✅ **COMPLETED**: Prediction Models Component - ML pipeline for law change predictions
4. ✅ **COMPLETED**: Semantic Search Component - Document retrieval and ranking system with cross-register matching
5. ✅ **COMPLETED**: Web API Base - FastAPI application factory with comprehensive features
6. ✅ **COMPLETED**: API Server Project - Main web API server combining web_api base with FastAPI application

**🚧 Next Priorities:**
1. ✅ **Redis Component** - Caching and background job queues (COMPLETED)
2. ✅ **Document Ingestion Component** - Processing pipeline integration (COMPLETED)
3. ✅ **Data Pipeline Base** - Batch processing infrastructure (COMPLETED)
4. ✅ **Data Processor Project** - Batch processing system (COMPLETED)
5. **Legal Graph Component** - Legal act dependency mapping and cross-reference analysis
6. **User Preferences Component** - User interest profiling and subscription management
7. **Notification System Component** - Multi-channel notification delivery
8. **Dashboard Component** - Interactive prediction visualization

## Next Steps

1. Read the project overview in `CLAUDE.md`
2. Review the detailed implementation plan in `IMPLEMENTATION_PLAN.md`
3. Check component implementation examples in `components/sejm_whiz/sejm_api/` and `components/sejm_whiz/eli_api/`
4. Run the projects to see them working:
   - **API server**: `uv run python projects/api_server/main.py`
     - Simple API server for backend services and health checks
     - Visit http://localhost:8000/docs for interactive API documentation
     - Visit http://localhost:8000/health for health check endpoint
   - **Web UI**: `uv run python projects/web_ui/main.py`
     - Complete monitoring dashboard with real-time features
     - Visit http://localhost:8000/ for home page with project overview
     - Visit http://localhost:8000/dashboard for real-time monitoring dashboard
     - Visit http://localhost:8000/docs for interactive API documentation
     - Visit http://localhost:8000/health for health check endpoint
   - **Data processor**: `uv run python projects/data_processor/main.py`
     - Demonstrates batch processing pipeline with modular steps
     - Shows integration of Sejm API, ELI API, text processing, and embeddings
     - Includes error handling and metrics collection
5. Run existing tests to understand patterns:
   - `uv run pytest test/bases/sejm_whiz/web_api/ -v`
   - `uv run pytest test/components/sejm_whiz/sejm_api/ -v`
   - `uv run pytest test/components/sejm_whiz/eli_api/ -v`
   - `uv run pytest test/components/sejm_whiz/vector_db/ -v`
   - `uv run pytest test/components/sejm_whiz/text_processing/ -v`
   - `uv run pytest test/components/sejm_whiz/embeddings/ -v`
   - `uv run pytest test/components/sejm_whiz/legal_nlp/ -v`
   - `uv run pytest test/components/sejm_whiz/prediction_models/ -v`
   - `uv run pytest test/components/sejm_whiz/semantic_search/ -v`
   - `uv run pytest test/components/sejm_whiz/redis/ -v`
   - `uv run pytest test/components/sejm_whiz/document_ingestion/ -v`
6. Follow the git feature branch workflow for all changes

## Getting Help

- uv documentation: https://docs.astral.sh/uv/
- Polylith documentation: https://polylith.gitbook.io/
- Project-specific help: Check `CLAUDE.md` for command reference
