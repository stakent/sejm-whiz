# P7 Database Management Scripts

This directory contains scripts for managing the sejm-whiz database specifically on the **p7 server**.

## Scripts

### 🗄️ setup-p7-database.sh

**Purpose**: Creates a fresh database on p7 server with proper schema and extensions

**Features**:

- ✅ Exits safely if database already exists on p7
- ✅ Creates PostgreSQL database and user on p7
- ✅ Installs pgvector extension
- ✅ Runs Alembic migrations
- ✅ Verifies complete setup
- ✅ Provides clear success/error messages
- 🎯 **P7-specific** - hardcoded for p7 server configuration

**Usage**:

```bash
# Setup database on p7 server
./scripts/setup-p7-database.sh
```

**Configuration** (hardcoded for p7):

- `DB_HOST=p7` - P7 database server
- `DB_NAME=sejm_whiz` - Database name
- `DB_USER=sejm_whiz_user` - Database user
- `DB_PASSWORD=sejm_whiz_password` - User password
- `SSH_HOST=root@p7` - SSH connection to p7

### 🗑️ cleanup-p7-database.sh

**Purpose**: Completely removes database and user from p7 server (⚠️ DESTRUCTIVE)

**Features**:

- ⚠️ Requires explicit "yes" confirmation
- 🗑️ Drops p7 database and all data
- 👤 Removes database user from p7
- ✅ Verifies complete cleanup
- 🎯 **P7-specific** - only affects p7 server

**Usage**:

```bash
# Interactive cleanup with confirmation
./scripts/cleanup-p7-database.sh

# The script will prompt:
# "Are you sure you want to continue? (type 'yes' to confirm):"
```

## Typical P7 Workflow

### Fresh P7 Database Setup

```bash
# 1. Run p7 setup script
./scripts/setup-p7-database.sh

# 2. Verify empty p7 database
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py db

# 3. Ingest sample data to p7
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --limit 10

# 4. Test p7 search functionality
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py search status
```

### Reset P7 Database (Development)

```bash
# 1. Clean up existing p7 database
./scripts/cleanup-p7-database.sh

# 2. Create fresh p7 database
./scripts/setup-p7-database.sh

# 3. Continue with data ingestion to p7...
```

## Requirements

- SSH access to **p7 server** as root
- PostgreSQL running on **p7**
- pgvector extension available on **p7**
- uv and Python environment set up locally
- Network connectivity to p7 (192.168.0.200:5432)

## Database Schema

After setup, the following tables are created:

| Table                 | Purpose                |
| --------------------- | ---------------------- |
| `alembic_version`     | Migration tracking     |
| `legal_documents`     | Legal document storage |
| `document_embeddings` | Vector embeddings      |
| `cross_references`    | Document relationships |
| `legal_amendments`    | Amendment tracking     |
| `prediction_models`   | ML model metadata      |

## Troubleshooting

### "Database already exists" Error

- Expected behavior - script protects existing data on p7
- Use `cleanup-p7-database.sh` first if you want to recreate

### "Permission denied for schema public"

- Script automatically grants proper permissions
- Ensures user owns database and schema

### "pgvector extension not found"

- Ensure pgvector is installed on p7 PostgreSQL server
- Usually available via package manager on Debian/Ubuntu

### Migration Failures

- Check PYTHONPATH and working directory
- Ensure Alembic configuration is correct
- Verify database connectivity to p7

## Security Notes

- **P7-specific** hardcoded credentials for development use
- Database password is hardcoded in scripts
- SSH connection uses key-based authentication to p7
- Scripts require root access to p7 server
- ⚠️ **Not for production** - credentials should be externalized for production deployments
