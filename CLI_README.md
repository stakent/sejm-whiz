# 🏛️ Sejm-Whiz CLI Guide

> **Comprehensive command-line interface for managing the Sejm-Whiz Polish Legal Document Analysis System**

## 🚀 Quick Start

```bash
# Install and setup
uv sync --dev
./scripts/install-completion.sh

# Basic usage (run with uv)
uv run python uv run python sejm-whiz-cli.py.py --help
uv run python uv run python sejm-whiz-cli.py.py info

# Or create alias for convenience
alias uv run python sejm-whiz-cli.py="uv run python ./uv run python sejm-whiz-cli.py.py"

# Check system health
uv run python uv run python sejm-whiz-cli.py.py system status

# Start ingesting documents
uv run python uv run python sejm-whiz-cli.py.py ingest documents --source eli --since 30d
```

## 📋 Command Overview

```
uv run python sejm-whiz-cli.py
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
uv run python uv run python sejm-whiz-cli.py.py system status

# Show system information
uv run python uv run python sejm-whiz-cli.py.py info
```

### Service Control

```bash
# Start all services
uv run python uv run python sejm-whiz-cli.py.py system start

# Start specific services
uv run python uv run python sejm-whiz-cli.py.py system start --services api,db

# Stop services
uv run python uv run python sejm-whiz-cli.py.py system stop

# Restart services
uv run python uv run python sejm-whiz-cli.py.py system restart
```

### Log Management

```bash
# View recent logs
uv run python uv run python sejm-whiz-cli.py.py system logs

# Follow logs in real-time
uv run python uv run python sejm-whiz-cli.py.py system logs --follow

# View specific service logs
uv run python uv run python sejm-whiz-cli.py.py system logs --service api --lines 100
```

## 🗄️ Database Operations

### Migrations

```bash
# Run database migrations
uv run python uv run python sejm-whiz-cli.py.py db migrate

# Migrate to specific revision
uv run python sejm-whiz-cli.py db migrate --revision abc123
```

### Data Management

```bash
# Seed with sample data
uv run python sejm-whiz-cli.py db seed

# Seed with full dataset
uv run python sejm-whiz-cli.py db seed --dataset full

# Check database status
uv run python sejm-whiz-cli.py db status
```

### Backup & Restore

```bash
# Create backup
uv run python sejm-whiz-cli.py db backup

# Create backup with custom name
uv run python sejm-whiz-cli.py db backup --output my-backup.sql

# Restore from backup
uv run python sejm-whiz-cli.py db restore backup-file.sql

# Reset database (⚠️ DANGEROUS)
uv run python sejm-whiz-cli.py db reset --yes
```

## 📥 Data Ingestion

### Document Ingestion

```bash
# Basic document ingestion
uv run python sejm-whiz-cli.py ingest documents --source eli

# Ingest with date range
uv run python sejm-whiz-cli.py ingest documents --from 2025-01-13 --to 2025-02-01

# Ingest from last 30 days
uv run python sejm-whiz-cli.py ingest documents --since 30d --limit 1000

# Force re-ingestion
uv run python sejm-whiz-cli.py ingest documents --force --batch-size 50
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
uv run python sejm-whiz-cli.py ingest embeddings

# Use CPU instead of GPU
uv run python sejm-whiz-cli.py ingest embeddings --cpu

# Custom batch size
uv run python sejm-whiz-cli.py ingest embeddings --batch-size 16 --model herbert
```

### Scheduled Jobs

```bash
# Schedule daily ingestion
uv run python sejm-whiz-cli.py ingest schedule --interval daily --time 02:00

# Check ingestion status
uv run python sejm-whiz-cli.py ingest status

# View ingestion logs
uv run python sejm-whiz-cli.py ingest logs --follow
```

## 🔍 Search Operations

### Document Search

```bash
# Semantic search
uv run python sejm-whiz-cli.py search query "ustawa o ochronie danych"

# Search with filters
uv run python sejm-whiz-cli.py search query "RODO" --limit 20 --min-score 0.8

# Search specific document type
uv run python sejm-whiz-cli.py search query "cyberbezpieczeństwo" --type ustawa
```

### Similar Documents

```bash
# Find similar documents
uv run python sejm-whiz-cli.py search similar DOC_ID_123

# Adjust similarity threshold
uv run python sejm-whiz-cli.py search similar DOC_ID_123 --threshold 0.7 --limit 10
```

### Search Management

```bash
# Rebuild search index
uv run python sejm-whiz-cli.py search reindex

# Force complete reindexing
uv run python sejm-whiz-cli.py search reindex --force

# Run search benchmarks
uv run python sejm-whiz-cli.py search benchmark --queries 500 --concurrent 20

# Check search system status
uv run python sejm-whiz-cli.py search status
```

## 🤖 Model Management

### Model Registry

```bash
# List available models
uv run python sejm-whiz-cli.py model list

# Filter by model type
uv run python sejm-whiz-cli.py model list --type embedding
```

### Model Training

```bash
# Train model with default settings
uv run python sejm-whiz-cli.py model train legal-classifier

# Custom training parameters
uv run python sejm-whiz-cli.py model train similarity-predictor \
  --dataset latest \
  --epochs 20 \
  --batch-size 64 \
  --gpu
```

### Model Evaluation & Deployment

```bash
# Evaluate model performance
uv run python sejm-whiz-cli.py model evaluate legal-classifier \
  --dataset test \
  --metrics accuracy,precision,recall

# Deploy model to staging
uv run python sejm-whiz-cli.py model deploy legal-classifier \
  --env staging \
  --replicas 2

# Check deployment status
uv run python sejm-whiz-cli.py model status --model herbert-embeddings
```

## ⚙️ Configuration Management

### View Configuration

```bash
# Show all configuration
uv run python sejm-whiz-cli.py config show

# Show specific section
uv run python sejm-whiz-cli.py config show --section database

# Export as JSON/YAML
uv run python sejm-whiz-cli.py config show --format json
uv run python sejm-whiz-cli.py config show --format yaml
```

### Update Configuration

```bash
# Set configuration value
uv run python sejm-whiz-cli.py config set database.pool_size 20

# Set with explicit type
uv run python sejm-whiz-cli.py config set api.debug true --type bool

# Validate configuration
uv run python sejm-whiz-cli.py config validate
```

### Import/Export Configuration

```bash
# Export configuration
uv run python sejm-whiz-cli.py config export --output config-backup.json

# Export as environment variables
uv run python sejm-whiz-cli.py config export --format env --output .env

# Import configuration
uv run python sejm-whiz-cli.py config import config.json --merge

# Reset to defaults
uv run python sejm-whiz-cli.py config reset --section database
```

## 🛠️ Development Tools

### Testing

```bash
# Run full test suite
uv run python sejm-whiz-cli.py dev test

# Test specific component
uv run python sejm-whiz-cli.py dev test --component embeddings

# Run with coverage
uv run python sejm-whiz-cli.py dev test --coverage --verbose
```

### Code Quality

```bash
# Run linting
uv run python sejm-whiz-cli.py dev lint

# Auto-fix issues
uv run python sejm-whiz-cli.py dev lint --fix

# Format code
uv run python sejm-whiz-cli.py dev format

# Check formatting only
uv run python sejm-whiz-cli.py dev format --check
```

### Complexity Analysis

```bash
# Check code complexity
uv run python sejm-whiz-cli.py dev complexity

# Set complexity threshold
uv run python sejm-whiz-cli.py dev complexity --threshold B

# Check specific component
uv run python sejm-whiz-cli.py dev complexity --component search
```

### Version Management

```bash
# Show current version
uv run python sejm-whiz-cli.py dev version show

# Bump version
uv run python sejm-whiz-cli.py dev version bump --type patch
uv run python sejm-whiz-cli.py dev version bump --type minor
uv run python sejm-whiz-cli.py dev version bump --type major

# Create git tag
uv run python sejm-whiz-cli.py dev version tag
```

## 🐚 Shell Completion

### Install Completion

```bash
# Auto-install for current shell
./scripts/install-completion.sh

# Manual installation
uv run python sejm-whiz-cli.py --install-completion bash
uv run python sejm-whiz-cli.py --install-completion zsh
```

### Using Completion

```bash
# Show all commands
uv run python sejm-whiz-cli.py <TAB><TAB>

# Show subcommands
uv run python sejm-whiz-cli.py system <TAB>
uv run python sejm-whiz-cli.py db <TAB>

# Complete parameters
uv run python sejm-whiz-cli.py ingest documents --<TAB>
```

## 🎯 Common Workflows

### Daily Operations

```bash
# Morning health check
uv run python sejm-whiz-cli.py system status
uv run python sejm-whiz-cli.py ingest status

# Check for new documents
uv run python sejm-whiz-cli.py ingest documents --since 1d --source eli

# Monitor search performance
uv run python sejm-whiz-cli.py search status
```

### Development Workflow

```bash
# Pre-commit checks
uv run python sejm-whiz-cli.py dev test --component mycomponent
uv run python sejm-whiz-cli.py dev lint --fix
uv run python sejm-whiz-cli.py dev complexity

# Database development
uv run python sejm-whiz-cli.py db migrate
uv run python sejm-whiz-cli.py db seed --dataset sample

# Model development
uv run python sejm-whiz-cli.py model train my-model --epochs 5
uv run python sejm-whiz-cli.py model evaluate my-model
```

### Production Deployment

```bash
# Pre-deployment checks
uv run python sejm-whiz-cli.py config validate
uv run python sejm-whiz-cli.py system status
uv run python sejm-whiz-cli.py dev test

# Deploy new model
uv run python sejm-whiz-cli.py model deploy classifier-v2 --env production
uv run python sejm-whiz-cli.py model status

# Post-deployment verification
uv run python sejm-whiz-cli.py search benchmark
uv run python sejm-whiz-cli.py system logs --follow
```

## 🚨 Troubleshooting

### Common Issues

**Services not starting:**

```bash
uv run python sejm-whiz-cli.py system status
uv run python sejm-whiz-cli.py config validate
uv run python sejm-whiz-cli.py system logs --service api
```

**Database connection issues:**

```bash
uv run python sejm-whiz-cli.py db status
uv run python sejm-whiz-cli.py config show --section database
```

**Search not working:**

```bash
uv run python sejm-whiz-cli.py search status
uv run python sejm-whiz-cli.py search reindex
```

**Ingestion failures:**

```bash
uv run python sejm-whiz-cli.py ingest status
uv run python sejm-whiz-cli.py ingest logs --lines 100
```

### Debug Mode

```bash
# Enable verbose output for any command
uv run python sejm-whiz-cli.py --verbose system status
uv run python sejm-whiz-cli.py --verbose ingest documents --source eli
```

## 🔗 Integration Examples

### Cron Jobs

```bash
# Daily ingestion at 2 AM
0 2 * * * /path/to/venv/bin/uv run python sejm-whiz-cli.py ingest documents --since 1d

# Weekly full reindex on Sunday
0 3 * * 0 /path/to/venv/bin/uv run python sejm-whiz-cli.py search reindex
```

### CI/CD Pipeline

```bash
#!/bin/bash
# Pre-deployment script
uv run python sejm-whiz-cli.py config validate
uv run python sejm-whiz-cli.py dev test
uv run python sejm-whiz-cli.py dev lint
uv run python sejm-whiz-cli.py dev complexity --threshold B
```

### Docker Integration

```dockerfile
# Add CLI to Docker image
COPY uv run python sejm-whiz-cli.py.py /usr/local/bin/uv run python sejm-whiz-cli.py
RUN chmod +x /usr/local/bin/uv run python sejm-whiz-cli.py

# Health check
HEALTHCHECK CMD uv run python sejm-whiz-cli.py system status || exit 1
```

## 📚 Additional Resources

- **API Documentation**: `/docs` endpoint when API server is running
- **Configuration Schema**: `uv run python sejm-whiz-cli.py config show --format yaml`
- **Component Architecture**: `uv run python sejm-whiz-cli.py info`
- **Performance Metrics**: `uv run python sejm-whiz-cli.py search benchmark`

## 💡 Tips & Best Practices

1. **Always validate configuration** before deploying: `uv run python sejm-whiz-cli.py config validate`
1. **Use date filtering** for efficient ingestion: `--since 1d` instead of full ingestion
1. **Monitor system health** regularly: `uv run python sejm-whiz-cli.py system status`
1. **Set up shell completion** for faster workflow: `./scripts/install-completion.sh`
1. **Use batch processing** for large operations: `--batch-size 100`
1. **Check logs** when troubleshooting: `--follow` for real-time monitoring

______________________________________________________________________

*For additional help on any command, use `--help`:*

```bash
uv run python sejm-whiz-cli.py COMMAND --help
uv run python sejm-whiz-cli.py COMMAND SUBCOMMAND --help
```
