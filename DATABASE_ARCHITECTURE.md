# Database Architecture - sejm_whiz Schema

## Overview

The sejm-whiz project implements a comprehensive PostgreSQL database schema with pgvector extension for storing and managing Polish legal documents with vector embeddings for semantic search.

## Schema Namespace Convention

All database objects follow the `sejm_whiz_*` naming convention for consistent namespace management:

### Database Tables

| Table Name                      | Purpose                              | Key Features                                                            |
| ------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `sejm_whiz_documents`           | Core legal document storage          | Vector embeddings (768-dim HerBERT), full text content, ELI identifiers |
| `sejm_whiz_amendments`          | Legal amendments and modifications   | Multi-act tracking, omnibus bill support, impact scoring                |
| `sejm_whiz_cross_references`    | Document relationships and citations | Semantic similarity scoring, reference type classification              |
| `sejm_whiz_document_embeddings` | Versioned embedding metadata         | Model versioning, preprocessing tracking, quality metrics               |
| `sejm_whiz_prediction_models`   | ML model performance tracking        | Model metadata, hyperparameters, deployment environment                 |

### Index Naming Convention

All indexes follow the pattern `idx_sejm_whiz_{table}_{column}`:

```sql
-- Document indexes
idx_sejm_whiz_documents_type
idx_sejm_whiz_documents_domain
idx_sejm_whiz_documents_published
idx_sejm_whiz_documents_embedding (ivfflat for vector similarity)

-- Amendment indexes
idx_sejm_whiz_amendments_document
idx_sejm_whiz_amendments_omnibus
idx_sejm_whiz_amendments_effective

-- Cross-reference indexes
idx_sejm_whiz_cross_refs_source
idx_sejm_whiz_cross_refs_target
idx_sejm_whiz_cross_refs_type

-- Embedding indexes
idx_sejm_whiz_embeddings_document
idx_sejm_whiz_embeddings_model
idx_sejm_whiz_embeddings_vector (ivfflat for vector operations)

-- Model indexes
idx_sejm_whiz_models_active
idx_sejm_whiz_models_type
idx_sejm_whiz_models_env
```

## Data Architecture

### Document Storage Model

```
sejm_whiz_documents (Main table)
├── Primary Key: UUID (uuid4)
├── Content: Full legal document text
├── Metadata: Title, type, domain, publication dates
├── Vector Embedding: 768-dimensional HerBERT vectors
├── Legal Classification: Act type, domain, amendment flags
└── Relationships: Foreign keys to amendments, cross-references
```

### Vector Search Architecture

The system uses pgvector with IVFFlat indexes for efficient similarity search:

- **Embedding Model**: HerBERT (768 dimensions) with bag-of-embeddings approach
- **Distance Metric**: Cosine similarity for semantic matching
- **Index Type**: IVFFlat for approximate nearest neighbor search
- **Performance**: GPU-accelerated embedding generation (20x speedup)

### Multi-Act Amendment Detection

```
sejm_whiz_amendments
├── Amendment Classification: change, addition, repeal
├── Multi-Act Tracking: omnibus_bill_id for grouped amendments
├── Impact Assessment: complexity scoring (0-100 scale)
└── Legal Reference: affected articles and paragraphs
```

## Migration History

### Migration: `94ff641a7af5_refactor_schema_naming_to_sejm_whiz_namespace`

**Date**: 2025-08-08
**Purpose**: Refactor all database objects to follow consistent `sejm_whiz` namespace

**Changes Applied**:

1. **Table Renaming** (dependency-aware order):

   ```sql
   legal_amendments       → sejm_whiz_amendments
   cross_references       → sejm_whiz_cross_references
   document_embeddings    → sejm_whiz_document_embeddings
   prediction_models      → sejm_whiz_prediction_models
   legal_documents        → sejm_whiz_documents  -- Renamed last due to FK dependencies
   ```

1. **Index Renaming**: All indexes updated to match table naming convention

1. **Data Preservation**: Zero data loss during migration with proper rollback support

## Performance Characteristics

### Query Performance

- **Document Retrieval**: O(1) lookup by UUID primary key
- **Vector Similarity**: O(log n) approximate with IVFFlat index
- **Text Search**: Full-text search with GIN indexes on content
- **Cross-Reference Queries**: Optimized with compound indexes

### Storage Efficiency

- **Document Storage**: Variable text compression with PostgreSQL TOAST
- **Vector Storage**: Efficient 32-bit float representation (768 × 4 bytes = 3KB per embedding)
- **Metadata Storage**: JSON columns for flexible legal document attributes

### Scalability Design

- **Partitioning Strategy**: Ready for date-based partitioning on `published_at`
- **Replica Support**: Read replicas for search-heavy workloads
- **Archival Strategy**: Older documents can be moved to slower storage tiers

## Data Flow Integration

### Input Sources

1. **Sejm API**: Parliamentary proceedings and legislative documents
1. **ELI API**: Official legal acts and regulations

### Processing Pipeline

1. **Document Ingestion**: `DocumentIngestionPipeline`
1. **Text Processing**: Polish legal text normalization
1. **Embedding Generation**: GPU-accelerated HerBERT processing
1. **Quality Validation**: Multi-tier content assessment
1. **Database Storage**: Transactional insertion with conflict resolution

### Output Interfaces

1. **REST API**: Semantic search endpoints
1. **Direct Database**: SQL queries for analytics
1. **Vector Search**: pgvector similarity operations

## Security & Compliance

### Data Protection

- **Encryption**: TLS for data in transit
- **Access Control**: Role-based database permissions
- **Audit Trail**: Comprehensive logging of document changes

### Legal Compliance

- **Document Integrity**: Immutable storage of original legal text
- **Version Control**: Full audit trail of document modifications
- **Source Attribution**: Trackable links to original government sources

## Future Architecture Considerations

### Scalability Enhancements

- **Distributed Storage**: Citus extension for horizontal scaling
- **Vector Index Optimization**: HNSW indexes for larger datasets
- **Multi-Model Support**: Support for multiple embedding models

### Feature Extensions

- **Document Versioning**: Temporal tables for document history
- **Advanced Analytics**: Time-series analysis of legal changes
- **Multi-Language Support**: Extension beyond Polish legal documents

## Development & Operations

### Local Development

```bash
# Database setup with Docker
docker-compose -f docker-compose.dev.yml up postgres

# Run migrations
DEPLOYMENT_ENV=local uv run alembic upgrade head

# Verify schema
psql -h localhost -p 5433 -U sejm_whiz_user -d sejm_whiz -c "\dt"
```

### Production Deployment

```bash
# Production migration (p7 environment)
DEPLOYMENT_ENV=p7 uv run alembic upgrade head

# Verify deployment
DEPLOYMENT_ENV=p7 uv run python -c "from sejm_whiz.database import get_db_session; print('✅ Database operational')"
```

This architecture provides a robust, scalable foundation for Polish legal document analysis with semantic search capabilities while maintaining clean namespace organization and high performance characteristics.
