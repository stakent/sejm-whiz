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

# Run specific test file (when tests are created)
uv run pytest tests/test_specific.py

# Run with coverage
uv run pytest --cov=sejm_whiz tests/
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
#         └── core.py
```

#### 2. Import Structure

Use proper namespace imports:

```python
# Import from other components
from sejm_whiz.eli_api import EliApiClient
from sejm_whiz.text_processing import clean_legal_text

# Import within same component
from sejm_whiz.sejm_api.utils import parse_response
```

#### 3. Development with REPL

```bash
# Start Python REPL with project environment
uv run python

# In REPL, you can import and test components:
# >>> from sejm_whiz.sejm_api import SejmApiClient
# >>> client = SejmApiClient()
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
├── components/              # Polylith components (coming soon)
├── projects/                # Polylith projects (coming soon)
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

## Next Steps

1. Read the project overview in `CLAUDE.md`
2. Review the detailed implementation plan in `sejm_whiz_plan.md`
3. Start with creating your first component using `uv run poly create component`
4. Follow the git feature branch workflow for all changes

## Getting Help

- uv documentation: https://docs.astral.sh/uv/
- Polylith documentation: https://polylith.gitbook.io/
- Project-specific help: Check `CLAUDE.md` for command reference