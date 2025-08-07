# 🏛️ Sejm-Whiz CLI Guide

> **Comprehensive command-line interface for managing the Sejm-Whiz Polish Legal Document Analysis System**

## 🚀 Quick Start

```bash
# Install and setup
uv sync --dev
./scripts/install-completion.sh

# Basic usage
sejm-whiz-cli --help
sejm-whiz-cli info

# Check system health
sejm-whiz-cli system status

# Start ingesting documents
sejm-whiz-cli ingest documents --source eli --since 30d
```

## 📋 Command Overview

```
sejm-whiz-cli
├── 📊 info             # System overview
├── 🔧 system           # System management
├── 🗄️ db               # Database operations
├── 📥 ingest           # Data ingestion
├── 🔍 search           # Search operations
├── 🤖 model            # ML model management
├── ⚙️ config           # Configuration
└── 🛠️ dev             # Development tools
```

## 🔧 System Management

### Status & Health Monitoring

```bash
# Comprehensive system health check
sejm-whiz-cli system status

# Show system information
sejm-whiz-cli info
```

### Service Control

```bash
# Start all services
sejm-whiz-cli system start

# Start specific services
sejm-whiz-cli system start --services api,db

# Stop services
sejm-whiz-cli system stop

# Restart services
sejm-whiz-cli system restart
```

### Log Management

```bash
# View recent logs
sejm-whiz-cli system logs

# Follow logs in real-time
sejm-whiz-cli system logs --follow

# View specific service logs
sejm-whiz-cli system logs --service api --lines 100
```

## 🗄️ Database Operations

### Migrations

```bash
# Run database migrations
sejm-whiz-cli db migrate

# Migrate to specific revision
sejm-whiz-cli db migrate --revision abc123
```

### Data Management

```bash
# Seed with sample data
sejm-whiz-cli db seed

# Seed with full dataset
sejm-whiz-cli db seed --dataset full

# Check database status
sejm-whiz-cli db status
```

### Backup & Restore

```bash
# Create backup
sejm-whiz-cli db backup

# Create backup with custom name
sejm-whiz-cli db backup --output my-backup.sql

# Restore from backup
sejm-whiz-cli db restore backup-file.sql

# Reset database (⚠️ DANGEROUS)
sejm-whiz-cli db reset --yes
```

## 📥 Data Ingestion

### Document Ingestion

```bash
# Basic document ingestion
sejm-whiz-cli ingest documents --source eli

# Ingest with date range
sejm-whiz-cli ingest documents --from 2025-01-13 --to 2025-02-01

# Ingest from last 30 days
sejm-whiz-cli ingest documents --since 30d --limit 1000

# Force re-ingestion
sejm-whiz-cli ingest documents --force --batch-size 50
```

#### Date Parameters

| Parameter | Format      | Example             | Description                  |
| --------- | ----------- | ------------------- | ---------------------------- |
| `--from`  | YYYY-MM-DD  | `--from 2025-01-13` | Start date (absolute)        |
| `--to`    | YYYY-MM-DD  | `--to 2025-02-01`   | End date (defaults to today) |
| `--since` | Xd/Xw/Xm/Xy | `--since 30d`       | Relative date from now       |

**Relative Date Examples:**

- `1d` = 1 day ago
- `1w` = 1 week ago
- `30d` = 30 days ago
- `1m` = 1 month ago
- `1y` = 1 year ago

### Embedding Generation

```bash
# Generate embeddings for new documents
sejm-whiz-cli ingest embeddings

# Use CPU instead of GPU
sejm-whiz-cli ingest embeddings --cpu

# Custom batch size
sejm-whiz-cli ingest embeddings --batch-size 16 --model herbert
```

### Scheduled Jobs

```bash
# Schedule daily ingestion
sejm-whiz-cli ingest schedule --interval daily --time 02:00

# Check ingestion status
sejm-whiz-cli ingest status

# View ingestion logs
sejm-whiz-cli ingest logs --follow
```

## 🔍 Search Operations

### Document Search

```bash
# Semantic search
sejm-whiz-cli search query "ustawa o ochronie danych"

# Search with filters
sejm-whiz-cli search query "RODO" --limit 20 --min-score 0.8

# Search specific document type
sejm-whiz-cli search query "cyberbezpieczeństwo" --type ustawa
```

### Similar Documents

```bash
# Find similar documents
sejm-whiz-cli search similar DOC_ID_123

# Adjust similarity threshold
sejm-whiz-cli search similar DOC_ID_123 --threshold 0.7 --limit 10
```

### Search Management

```bash
# Rebuild search index
sejm-whiz-cli search reindex

# Force complete reindexing
sejm-whiz-cli search reindex --force

# Run search benchmarks
sejm-whiz-cli search benchmark --queries 500 --concurrent 20

# Check search system status
sejm-whiz-cli search status
```

## 🤖 Model Management

### Model Registry

```bash
# List available models
sejm-whiz-cli model list

# Filter by model type
sejm-whiz-cli model list --type embedding
```

### Model Training

```bash
# Train model with default settings
sejm-whiz-cli model train legal-classifier

# Custom training parameters
sejm-whiz-cli model train similarity-predictor \
  --dataset latest \
  --epochs 20 \
  --batch-size 64 \
  --gpu
```

### Model Evaluation & Deployment

```bash
# Evaluate model performance
sejm-whiz-cli model evaluate legal-classifier \
  --dataset test \
  --metrics accuracy,precision,recall

# Deploy model to staging
sejm-whiz-cli model deploy legal-classifier \
  --env staging \
  --replicas 2

# Check deployment status
sejm-whiz-cli model status --model herbert-embeddings
```

## ⚙️ Configuration Management

### View Configuration

```bash
# Show all configuration
sejm-whiz-cli config show

# Show specific section
sejm-whiz-cli config show --section database

# Export as JSON/YAML
sejm-whiz-cli config show --format json
sejm-whiz-cli config show --format yaml
```

### Update Configuration

```bash
# Set configuration value
sejm-whiz-cli config set database.pool_size 20

# Set with explicit type
sejm-whiz-cli config set api.debug true --type bool

# Validate configuration
sejm-whiz-cli config validate
```

### Import/Export Configuration

```bash
# Export configuration
sejm-whiz-cli config export --output config-backup.json

# Export as environment variables
sejm-whiz-cli config export --format env --output .env

# Import configuration
sejm-whiz-cli config import config.json --merge

# Reset to defaults
sejm-whiz-cli config reset --section database
```

## 🛠️ Development Tools

### Testing

```bash
# Run full test suite
sejm-whiz-cli dev test

# Test specific component
sejm-whiz-cli dev test --component embeddings

# Run with coverage
sejm-whiz-cli dev test --coverage --verbose
```

### Code Quality

```bash
# Run linting
sejm-whiz-cli dev lint

# Auto-fix issues
sejm-whiz-cli dev lint --fix

# Format code
sejm-whiz-cli dev format

# Check formatting only
sejm-whiz-cli dev format --check
```

### Complexity Analysis

```bash
# Check code complexity
sejm-whiz-cli dev complexity

# Set complexity threshold
sejm-whiz-cli dev complexity --threshold B

# Check specific component
sejm-whiz-cli dev complexity --component search
```

### Version Management

```bash
# Show current version
sejm-whiz-cli dev version show

# Bump version
sejm-whiz-cli dev version bump --type patch
sejm-whiz-cli dev version bump --type minor
sejm-whiz-cli dev version bump --type major

# Create git tag
sejm-whiz-cli dev version tag
```

## 🐚 Shell Completion

### Install Completion

```bash
# Auto-install for current shell
./scripts/install-completion.sh

# Manual installation
sejm-whiz-cli --install-completion bash
sejm-whiz-cli --install-completion zsh
```

### Using Completion

```bash
# Show all commands
sejm-whiz-cli <TAB><TAB>

# Show subcommands
sejm-whiz-cli system <TAB>
sejm-whiz-cli db <TAB>

# Complete parameters
sejm-whiz-cli ingest documents --<TAB>
```

## 🎯 Common Workflows

### Daily Operations

```bash
# Morning health check
sejm-whiz-cli system status
sejm-whiz-cli ingest status

# Check for new documents
sejm-whiz-cli ingest documents --since 1d --source eli

# Monitor search performance
sejm-whiz-cli search status
```

### Development Workflow

```bash
# Pre-commit checks
sejm-whiz-cli dev test --component mycomponent
sejm-whiz-cli dev lint --fix
sejm-whiz-cli dev complexity

# Database development
sejm-whiz-cli db migrate
sejm-whiz-cli db seed --dataset sample

# Model development
sejm-whiz-cli model train my-model --epochs 5
sejm-whiz-cli model evaluate my-model
```

### Production Deployment

```bash
# Pre-deployment checks
sejm-whiz-cli config validate
sejm-whiz-cli system status
sejm-whiz-cli dev test

# Deploy new model
sejm-whiz-cli model deploy classifier-v2 --env production
sejm-whiz-cli model status

# Post-deployment verification
sejm-whiz-cli search benchmark
sejm-whiz-cli system logs --follow
```

## 🚨 Troubleshooting

### Common Issues

**Services not starting:**

```bash
sejm-whiz-cli system status
sejm-whiz-cli config validate
sejm-whiz-cli system logs --service api
```

**Database connection issues:**

```bash
sejm-whiz-cli db status
sejm-whiz-cli config show --section database
```

**Search not working:**

```bash
sejm-whiz-cli search status
sejm-whiz-cli search reindex
```

**Ingestion failures:**

```bash
sejm-whiz-cli ingest status
sejm-whiz-cli ingest logs --lines 100
```

### Debug Mode

```bash
# Enable verbose output for any command
sejm-whiz-cli --verbose system status
sejm-whiz-cli --verbose ingest documents --source eli
```

## 🔗 Integration Examples

### Cron Jobs

```bash
# Daily ingestion at 2 AM
0 2 * * * /path/to/venv/bin/sejm-whiz-cli ingest documents --since 1d

# Weekly full reindex on Sunday
0 3 * * 0 /path/to/venv/bin/sejm-whiz-cli search reindex
```

### CI/CD Pipeline

```bash
#!/bin/bash
# Pre-deployment script
sejm-whiz-cli config validate
sejm-whiz-cli dev test
sejm-whiz-cli dev lint
sejm-whiz-cli dev complexity --threshold B
```

### Docker Integration

```dockerfile
# Add CLI to Docker image
COPY sejm-whiz-cli.py /usr/local/bin/sejm-whiz-cli
RUN chmod +x /usr/local/bin/sejm-whiz-cli

# Health check
HEALTHCHECK CMD sejm-whiz-cli system status || exit 1
```

## 📚 Additional Resources

- **API Documentation**: `/docs` endpoint when API server is running
- **Configuration Schema**: `sejm-whiz-cli config show --format yaml`
- **Component Architecture**: `sejm-whiz-cli info`
- **Performance Metrics**: `sejm-whiz-cli search benchmark`

## 💡 Tips & Best Practices

1. **Always validate configuration** before deploying: `sejm-whiz-cli config validate`
1. **Use date filtering** for efficient ingestion: `--since 1d` instead of full ingestion
1. **Monitor system health** regularly: `sejm-whiz-cli system status`
1. **Set up shell completion** for faster workflow: `./scripts/install-completion.sh`
1. **Use batch processing** for large operations: `--batch-size 100`
1. **Check logs** when troubleshooting: `--follow` for real-time monitoring

______________________________________________________________________

*For additional help on any command, use `--help`:*

```bash
sejm-whiz-cli COMMAND --help
sejm-whiz-cli COMMAND SUBCOMMAND --help
```
