# P7CLI Setup Guide

## Problem

CLI commands like `DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --source eli --since 30d` run locally on x230 but connect to p7 database, causing resource bottlenecks on the local machine.

## Solution

Use the `p7cli` tool to execute commands directly on p7 server using the sejm-whiz user account.

## Setup

### 1. Verify P7 Access as sejm-whiz User

```bash
ssh sejm-whiz@p7
```

### 2. Verify Sejm-Whiz Installation on P7

```bash
ssh sejm-whiz@p7 "ls -la /home/sejm-whiz/"
# Should show: sejm-whiz-baremetal directory with sejm-whiz-cli.py
```

### 3. Make p7cli Executable (if not already)

```bash
chmod +x ./p7cli
```

## Usage

### Basic Commands

```bash
# Check system status
./p7cli system status

# Ingest documents from ELI API
./p7cli ingest documents --source eli --since 30d --limit 50

# Search documents
./p7cli search query "ustawa RODO" --limit 3

# Database operations
./p7cli db status
```

### Common Operations

```bash
# Recent document ingestion (both sources)
./p7cli ingest documents --since 7d

# ELI-only ingestion with date range
./p7cli ingest documents --source eli --from 2025-01-01 --to 2025-01-31

# Check ingestion status
./p7cli ingest status
```

## Architecture

- **Execution Location**: p7 server
- **User**: sejm-whiz (not root)
- **Working Directory**: `/home/sejm-whiz/sejm-whiz-baremetal/`
- **Environment**: `DEPLOYMENT_ENV=p7`
- **Python Path**: `PYTHONPATH=components:bases`

## Benefits

1. **Full p7 Resources**: CPU-intensive processing uses p7's resources
1. **Local Database**: All database/Redis connections are local to p7
1. **KISS Principle**: Simple SSH execution, no complex wrappers
1. **Proper User Context**: Runs as sejm-whiz user with correct permissions
1. **Development Friendly**: Easy to debug and modify commands

## Troubleshooting

### SSH Connection Issues

```bash
# Test SSH connectivity as sejm-whiz user
ssh sejm-whiz@p7 "echo 'SSH working'"
```

### Directory Issues

```bash
# Verify sejm-whiz directory structure
ssh sejm-whiz@p7 "ls -la /home/sejm-whiz/sejm-whiz-baremetal/"
```

### uv Installation Issues

```bash
# Check if uv is installed for sejm-whiz user
ssh sejm-whiz@p7 "~/.local/bin/uv --version"
# If not found, it will be auto-installed on first use
```

## Optional: Create Alias

Add to `~/.bashrc` for convenience:

```bash
alias p7='./p7cli'
```

Then use: `p7 system status` instead of `./p7cli system status`
