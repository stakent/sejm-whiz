# Step-by-Step Implementation Plan

This document provides a detailed, actionable implementation plan for the sejm-whiz project, broken down into specific tasks with commands and deliverables.

## Prerequisites Checklist

- [x] Development environment set up (see `DEVELOPER_SETUP.md`)
- [x] `uv sync --dev` completed successfully
- [x] `uv run poly info` shows workspace is ready
- [x] Git repository is clean and on feature/database-setup branch

---

## Phase 1: Infrastructure & Core Setup (Weeks 1-4)

### Step 1.1: Database Setup

**Objective**: Set up PostgreSQL with pgvector extension for vector storage

**Tasks:**
```bash
# Create feature branch
git checkout -b feature/database-setup

# Add database dependencies
uv add psycopg2-binary sqlalchemy alembic
uv add pgvector

# Add development dependencies
uv add --dev pytest-asyncio pytest-postgresql
```

**Deliverables:**
- [x] PostgreSQL 17 installed with pgvector extension
- [x] Database connection configuration
- [x] Basic schema for legal documents and embeddings
- [x] Migration system setup with Alembic

**Validation:**
```bash
# Test database connection
uv run python -c "import psycopg2; print('PostgreSQL connection OK')"

# Verify pgvector extension
uv run python -c "import pgvector; print('pgvector extension OK')"
```

### Step 1.2: Container Environment

**Objective**: Set up Docker containers for development

**Tasks:**
```bash
# Create Docker configuration
touch Dockerfile.api Dockerfile.processor

# Containerization handled via Helm charts for k3s deployment
```

**Deliverables:**
- [x] Dockerfile for main application (Dockerfile.api, Dockerfile.processor)
- [x] Helm charts for PostgreSQL + pgvector deployment
- [x] Helm charts for Redis caching layer
- [x] k3s persistent volume configuration

**Validation:**
```bash
# Verify k3s deployment
kubectl get pods -n sejm-whiz
kubectl get services -n sejm-whiz
```

---

## Phase 2: Core API Components (Weeks 5-8)

### Step 2.1: Create sejm_api Component ✅ **COMPLETED**

**Objective**: Implement Sejm Proceedings API integration

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/sejm-api-component

# Create component
uv run poly create component --name sejm_api

# Add HTTP client dependencies
uv add httpx pydantic

# Dependencies added during implementation
```

**Component Structure:**
```
components/sejm_api/
└── sejm_whiz/
    └── sejm_api/
        ├── __init__.py
        ├── client.py          # Main API client
        ├── models.py          # Pydantic models
        ├── rate_limiter.py    # Rate limiting logic
        └── exceptions.py      # Custom exceptions
```

**Key Files Implemented:**
- [x] `client.py`: SejmApiClient class with async HTTP methods and comprehensive validation
- [x] `models.py`: Complete Pydantic models for all API responses
- [x] `rate_limiter.py`: Advanced rate limiting with token bucket and sliding window algorithms
- [x] `exceptions.py`: Comprehensive exception hierarchy with factory functions

**Security Features Added:**
- [x] Endpoint validation to prevent URL manipulation
- [x] Error message sanitization to prevent information disclosure
- [x] Comprehensive input validation for all parameters
- [x] Rate limiting with token bucket and sliding window algorithms

**Testing:**
```bash
# Comprehensive test suite created
test/components/sejm_whiz/sejm_api/
├── test_client.py         # API client tests
├── test_core.py          # Core functionality tests
├── test_exceptions.py    # Exception handling tests
├── test_models.py        # Pydantic model tests
├── test_rate_limiter.py  # Rate limiting tests
└── test_validation.py    # Security validation tests

# Run component tests
uv run pytest test/components/sejm_whiz/sejm_api/ -v
```

**Validation:**
- [x] API client can fetch proceedings data with comprehensive methods
- [x] Rate limiting works correctly with token bucket and sliding window
- [x] Error handling for API failures with sanitized messages
- [x] Security hardening against URL manipulation and information disclosure
- [x] Tests pass with comprehensive coverage (248 tests passing)
- [x] Code formatted and linted with ruff

### Step 2.2: Create eli_api Component

**Objective**: Implement ELI API integration for legal documents

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main  
git checkout -b feature/eli-api-component

# Create component
uv run poly create component --name eli_api

# Dependencies already added in previous step
```

**Component Structure:**
```
components/eli_api/
└── sejm_whiz/
    └── eli_api/
        ├── __init__.py
        ├── client.py          # ELI API client
        ├── models.py          # Legal document models
        ├── parser.py          # Document structure extraction
        └── utils.py           # Helper functions
```

**Key Files to Implement:**
- [ ] `client.py`: EliApiClient class
- [ ] `models.py`: LegalDocument, Amendment models
- [ ] `parser.py`: Extract structure from legal documents
- [ ] `utils.py`: Legal text processing utilities

**Validation:**
- [ ] Can fetch legal documents from ELI API
- [ ] Document parsing extracts structure correctly
- [ ] Models validate legal document data
- [ ] Tests pass with >90% coverage

### Step 2.3: Create vector_db Component

**Objective**: Implement PostgreSQL + pgvector operations

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/vector-db-component

# Create component
uv run poly create component --name vector_db

# Add vector database dependencies (already added)
```

**Component Structure:**
```
components/vector_db/
└── sejm_whiz/
    └── vector_db/
        ├── __init__.py
        ├── connection.py      # Database connection
        ├── operations.py      # CRUD operations
        ├── embeddings.py      # Vector operations
        ├── migrations/        # Database migrations
        └── models.py          # SQLAlchemy models
```

**Key Files to Implement:**
- [ ] `connection.py`: Database connection management
- [ ] `operations.py`: Insert, update, delete operations
- [ ] `embeddings.py`: Vector similarity search
- [ ] `models.py`: SQLAlchemy models with vector columns

**Database Schema:**
```sql
-- Legal documents table
CREATE TABLE legal_documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(768),  -- HerBERT embedding size
    created_at TIMESTAMP,
    document_type VARCHAR(50)
);

-- Create vector index
CREATE INDEX ON legal_documents USING ivfflat (embedding vector_cosine_ops);
```

**Validation:**
- [ ] Database connection works
- [ ] Can store and retrieve embeddings
- [ ] Vector similarity search functions correctly
- [ ] Migration system works

---

## Phase 3: Data Processing Components (Weeks 9-12)

### Step 3.1: Create text_processing Component

**Objective**: Implement text cleaning and preprocessing for Polish legal text

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/text-processing-component

# Create component
uv run poly create component --name text_processing

# Add NLP dependencies
uv add spacy nltk regex
uv add https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.7.0/pl_core_news_sm-3.7.0-py3-none-any.whl
```

**Component Structure:**
```
components/text_processing/
└── sejm_whiz/
    └── text_processing/
        ├── __init__.py
        ├── cleaner.py         # Text cleaning functions
        ├── normalizer.py      # Text normalization
        ├── tokenizer.py       # Polish tokenization
        ├── entities.py        # Named entity recognition
        └── legal_parser.py    # Legal document parsing
```

**Key Features:**
- [ ] Remove HTML tags and formatting
- [ ] Normalize Polish diacritics
- [ ] Extract legal references (article numbers, act names)
- [ ] Entity recognition for legal terms
- [ ] Sentence and paragraph segmentation

**Validation:**
```bash
# Test with sample legal text
uv run python -c "
from sejm_whiz.text_processing import clean_legal_text
text = 'Art. 123. § 1. Przykład tekstu prawnego...'
cleaned = clean_legal_text(text)
print(f'Cleaned: {cleaned}')
"
```

### Step 3.2: Create embeddings Component

**Objective**: Implement bag of embeddings approach with HerBERT

**Tasks:**
```bash
# Create feature branch  
git checkout main
git pull origin main
git checkout -b feature/embeddings-component

# Create component
uv run poly create component --name embeddings

# Add ML dependencies
uv add torch transformers tokenizers
uv add sentence-transformers
```

**Component Structure:**
```
components/embeddings/
└── sejm_whiz/
    └── embeddings/
        ├── __init__.py
        ├── herbert_encoder.py # HerBERT embedding generation
        ├── bag_embeddings.py  # Document-level averaging
        ├── similarity.py      # Cosine similarity functions
        └── batch_processor.py # Efficient batch processing
```

**Key Features:**
- [ ] Load HerBERT model for Polish text
- [ ] Generate token-level embeddings
- [ ] Average tokens to create document embeddings
- [ ] Cosine similarity calculation
- [ ] GPU optimization for GTX 1060

**Implementation Example:**
```python
# herbert_encoder.py
from transformers import AutoTokenizer, AutoModel
import torch

class HerBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-v1")
        self.model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
    
    def encode_document(self, text: str) -> torch.Tensor:
        # Tokenize and encode
        tokens = self.tokenizer(text, return_tensors="pt", truncate=True)
        outputs = self.model(**tokens)
        
        # Average token embeddings (bag of embeddings)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()
```

**Validation:**
- [ ] HerBERT model loads correctly
- [ ] Can generate embeddings for Polish legal text
- [ ] Bag of embeddings averaging works
- [ ] GPU utilization is efficient (<6GB VRAM)

### Step 3.3: Create legal_nlp Component

**Objective**: Legal document analysis with multi-act amendment detection

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/legal-nlp-component

# Create component
uv run poly create component --name legal_nlp

# Add advanced NLP dependencies
uv add scikit-learn networkx matplotlib
```

**Component Structure:**
```
components/legal_nlp/
└── sejm_whiz/
    └── legal_nlp/
        ├── __init__.py
        ├── amendment_detector.py  # Multi-act amendment detection
        ├── cross_reference.py     # Cross-reference analysis
        ├── impact_analyzer.py     # Impact assessment
        ├── sentiment.py           # Debate sentiment analysis
        └── topic_modeling.py      # Topic modeling for discussions
```

**Key Features:**
- [ ] Detect amendments affecting multiple legal acts
- [ ] Parse cross-references in omnibus legislation
- [ ] Analyze cascading impacts of legal changes
- [ ] Sentiment analysis of parliamentary debates
- [ ] Topic modeling for legal discussions

**Multi-Act Detection Algorithm:**
```python
# amendment_detector.py
import re
from typing import List, Dict

class MultiActAmendmentDetector:
    def __init__(self):
        self.act_patterns = [
            r"ustaw[aeęy]?\s+z\s+dnia\s+\d+.*?r\.",  # Polish law references
            r"kodeks[ua]?\s+\w+",                      # Code references
            r"rozporządzeni[aeęy]?\s+.*?"              # Regulation references
        ]
    
    def detect_multi_act_amendments(self, text: str) -> Dict[str, List[str]]:
        detected_acts = []
        for pattern in self.act_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            detected_acts.extend(matches)
        
        return {
            "affected_acts": detected_acts,
            "is_omnibus": len(detected_acts) > 1,
            "complexity_score": len(detected_acts)
        }
```

**Validation:**
- [ ] Multi-act amendments detected correctly
- [ ] Cross-references parsed accurately
- [ ] Impact analysis produces meaningful results

---

## Phase 4: ML Components (Weeks 13-16)

### Step 4.1: Create prediction_models Component

**Objective**: Implement ML models for law change predictions

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/prediction-models-component

# Create component
uv run poly create component --name prediction_models

# Add ML dependencies
uv add scikit-learn xgboost lightgbm
uv add optuna  # for hyperparameter optimization
```

**Component Structure:**
```
components/prediction_models/
└── sejm_whiz/
    └── prediction_models/
        ├── __init__.py
        ├── base_model.py       # Base prediction model class
        ├── embedding_model.py  # Embedding-based predictions
        ├── ensemble.py         # Model ensemble methods
        ├── trainer.py          # Model training utilities
        └── evaluator.py        # Model evaluation metrics
```

**Key Features:**
- [ ] Embedding-based similarity models
- [ ] Ensemble methods for robust predictions
- [ ] Model training and validation pipelines
- [ ] Performance metrics and evaluation

### Step 4.2: Create semantic_search Component

**Objective**: Implement semantic search using bag of embeddings

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/semantic-search-component

# Create component
uv run poly create component --name semantic_search

# Dependencies already available
```

**Component Structure:**
```
components/semantic_search/
└── sejm_whiz/
    └── semantic_search/
        ├── __init__.py
        ├── search_engine.py    # Main search interface
        ├── indexer.py          # Document indexing
        ├── ranker.py           # Result ranking
        └── cross_register.py   # Formal/informal matching
```

**Key Features:**
- [ ] Cosine similarity search on document embeddings
- [ ] Cross-register matching (legal vs parliamentary language)
- [ ] Real-time similarity scoring
- [ ] Efficient indexing and retrieval

---

## Phase 5: Project Assembly (Weeks 17-20)

### Step 5.1: Create web_api Base

**Objective**: Create FastAPI web server base

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/web-api-base

# Create base
uv run poly create base --name web_api

# Add web framework dependencies
uv add fastapi uvicorn pydantic-settings
uv add python-multipart  # for file uploads
```

**Base Structure:**
```
bases/web_api/
└── sejm_whiz/
    └── web_api/
        ├── __init__.py
        ├── main.py            # FastAPI application
        ├── routes/            # API route definitions
        ├── middleware.py      # Custom middleware
        └── config.py          # Configuration management
```

### Step 5.2: Create api_server Project

**Objective**: Assemble complete API server

**Tasks:**
```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/api-server-project

# Create project
uv run poly create project --name api_server
```

**Project Dependencies:**
Edit `projects/api_server/pyproject.toml`:
```toml
[project]
name = "sejm-whiz-api-server"
dependencies = [
    "sejm_whiz[web_api,sejm_api,eli_api,semantic_search,prediction_models]"
]
```

**API Endpoints:**
- [ ] `/api/v1/predictions` - Get law change predictions
- [ ] `/api/v1/search` - Semantic search endpoint
- [ ] `/api/v1/documents` - Legal document management
- [ ] `/api/v1/users` - User preference management

### Step 5.3: Create data_processor Project

**Objective**: Assemble batch data processing system

**Tasks:**
```bash
# Create feature branch
git checkout main  
git pull origin main
git checkout -b feature/data-processor-project

# Create project
uv run poly create project --name data_processor
```

**Processing Pipeline:**
- [ ] Automated API data ingestion
- [ ] Document preprocessing and cleaning
- [ ] Embedding generation and storage
- [ ] Multi-act amendment detection
- [ ] Legal taxonomy classification

---

## Phase 6: Testing & Quality Assurance (Weeks 21-22)

### Step 6.1: Component Testing

**Objective**: Comprehensive testing of all components

**Tasks:**
```bash
# Add testing dependencies
uv add --dev pytest pytest-cov pytest-mock
uv add --dev pytest-asyncio httpx

# Run all component tests
uv run poly test

# Generate coverage report
uv run pytest --cov=sejm_whiz --cov-report=html
```

**Testing Checklist:**
- [ ] All components have >90% test coverage
- [ ] Integration tests between components
- [ ] Mock tests for external APIs
- [ ] Performance tests for GPU components

### Step 6.2: End-to-End Testing

**Objective**: Full workflow validation

**Test Scenarios:**
- [ ] Complete document processing pipeline
- [ ] API request/response cycles
- [ ] Multi-act amendment detection accuracy
- [ ] Embedding similarity calculations
- [ ] Database operations and migrations

### Step 6.3: Polylith Validation

**Commands:**
```bash
# Validate workspace integrity
uv run poly check

# Verify all projects build successfully
uv run poly build

# Check component dependencies
uv run poly deps

# Validate project configurations
uv run poly sync
```

---

## Phase 7: Deployment Preparation (Weeks 23-24)

### Step 7.1: Docker Configuration

**Objective**: Containerize all services

**Tasks:**
```bash
# Production deployment handled via Helm charts
# Dockerfiles already exist: Dockerfile.api, Dockerfile.processor

# Create Helm chart templates for production
mkdir -p helm/charts/sejm-whiz-api
mkdir -p helm/charts/sejm-whiz-processor
```

### Step 7.2: GPU Optimization

**Objective**: Optimize for GTX 1060 6GB constraints

**Tasks:**
- [ ] Memory usage profiling
- [ ] Batch size optimization
- [ ] Model quantization if needed
- [ ] GPU monitoring setup

### Step 7.3: Production Readiness

**Checklist:**
- [ ] Environment configuration management
- [ ] Logging and monitoring setup
- [ ] Error handling and recovery
- [ ] Database backup strategy
- [ ] Security audit completed

---

## Success Metrics & Validation

### Technical Metrics
- [ ] Embedding generation: <200ms for 1000 tokens
- [ ] Vector similarity search: <50ms response time
- [ ] Multi-act amendment detection: >90% accuracy
- [ ] API response time: <500ms for complex queries
- [ ] GPU memory usage: <5.5GB peak
- [ ] Test coverage: >90% for all components

### Development Metrics
- [ ] All Polylith checks pass (`uv run poly check`)
- [ ] All projects build successfully (`uv run poly build`)
- [ ] No circular dependencies between components
- [ ] Consistent code style and documentation

### Deployment Metrics
- [ ] Docker containers build without errors
- [ ] Database migrations run successfully
- [ ] All services start and communicate properly
- [ ] Monitoring and logging operational

---

## Emergency Procedures

### Rollback Strategy
```bash
# Rollback to previous working state
git checkout main
git reset --hard HEAD~1
uv sync --dev
```

### Component Isolation
```bash
# Test individual component
uv run poly test --component component_name

# Check component in isolation
uv run poly check --component component_name
```

### Debug Commands
```bash
# Verbose workspace info
uv run poly info --verbose

# Dependency analysis
uv run poly deps --show-deps

# Build analysis
uv run poly build --verbose
```

---

## Next Steps

1. **Start with Phase 1**: Set up database and basic infrastructure
2. **Follow feature branch workflow**: Create branch for each component
3. **Test continuously**: Run `uv run poly test` after each component
4. **Validate frequently**: Use `uv run poly check` to ensure workspace health
5. **Document progress**: Update this plan with actual completion times

This implementation plan provides a structured approach to building the sejm-whiz system using Polylith architecture with concrete commands, deliverables, and validation steps.