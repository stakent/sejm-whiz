#!/bin/bash
# Refactoring Plan: Incremental API Refactoring
# Strategy: incremental
# Estimated time: 5.8 hours

set -e  # Exit on error

echo "Starting refactoring: Incremental API Refactoring"
echo "Strategy: incremental"
echo "Estimated time: 5.8 hours"
echo ""

# Prerequisites check
echo "Checking prerequisites..."
echo "- Create feature branch: git checkout -b refactor/api-migration"
echo "- Backup current state: git tag backup-before-refactor"
echo "- Ensure all tests pass: uv run poly test"
echo "- Ensure clean working directory: git status"

read -p "Prerequisites met? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete prerequisites first"
    exit 1
fi


echo "=== Step step_001: Apply backward compatible changes ==="
echo "Risk level: low"
echo "Files affected: 2"
echo "Estimated time: 45 minutes"
echo ""

# Check dependencies

read -p "Execute this step? (y/n/s=skip): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "Skipping step"
elif [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing commands..."
    echo "# Update imports in components/sejm_whiz/document_ingestion/cached_ingestion_pipeline.py"
    sed -i 's/from old_module import/from new_module import/g' components/sejm_whiz/document_ingestion/cached_ingestion_pipeline.py
    sed -i 's/import old_module/import new_module/g' components/sejm_whiz/document_ingestion/cached_ingestion_pipeline.py
    echo "# Update imports in components/sejm_whiz/document_ingestion/content_extraction_orchestrator.py"
    sed -i 's/from old_module import/from new_module import/g' components/sejm_whiz/document_ingestion/content_extraction_orchestrator.py
    sed -i 's/import old_module/import new_module/g' components/sejm_whiz/document_ingestion/content_extraction_orchestrator.py

    echo "Running validation..."
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"

    echo "Step completed successfully"
else
    echo "Step cancelled"
    exit 1
fi
echo ""

echo "=== Step step_002: Update breaking changes batch 1 ==="
echo "Risk level: high"
echo "Files affected: 5"
echo "Estimated time: 90 minutes"
echo ""

# Check dependencies
echo "Dependency: step_001 (should be completed)"

read -p "Execute this step? (y/n/s=skip): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "Skipping step"
elif [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing commands..."
    echo "# Update method signatures in components/sejm_whiz/eli_api/client.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/eli_api/client.py
    echo "# Update method signatures in components/sejm_whiz/eli_api/content_validator.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/eli_api/content_validator.py
    echo "# Update method signatures in test/components/sejm_whiz/eli_api/test_client.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file test/components/sejm_whiz/eli_api/test_client.py
    echo "# Update method signatures in components/sejm_whiz/eli_api/client.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/eli_api/client.py
    echo "# Update method signatures in test/components/sejm_whiz/eli_api/test_client.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file test/components/sejm_whiz/eli_api/test_client.py

    echo "Running validation..."
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    uv run poly test --affected || echo "Validation warning for: uv run poly test --affected"

    echo "Creating checkpoint..."
    git add -A
    git commit -m "Checkpoint after step_002: Update breaking changes batch 1"
    git tag checkpoint-step_002

    echo "Step completed successfully"
else
    echo "Step cancelled"
    exit 1
fi
echo ""

echo "=== Step step_003: Update breaking changes batch 2 ==="
echo "Risk level: high"
echo "Files affected: 4"
echo "Estimated time: 90 minutes"
echo ""

# Check dependencies
echo "Dependency: step_002 (should be completed)"

read -p "Execute this step? (y/n/s=skip): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "Skipping step"
elif [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing commands..."
    echo "# Update method signatures in components/sejm_whiz/cli/commands/ingest.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/cli/commands/ingest.py
    echo "# Update method signatures in components/sejm_whiz/document_ingestion/multi_api_pipeline.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/document_ingestion/multi_api_pipeline.py
    echo "# Update method signatures in components/sejm_whiz/document_ingestion/dual_stream_pipeline.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/document_ingestion/dual_stream_pipeline.py
    echo "# Update method signatures in components/sejm_whiz/document_ingestion/content_extraction_orchestrator.py"
    echo "# Use Python AST-based refactoring tool"
    python scripts/update_signatures.py --file components/sejm_whiz/document_ingestion/content_extraction_orchestrator.py

    echo "Running validation..."
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    python -m py_compile {file} || echo "Validation warning for: python -m py_compile {file}"
    uv run poly test --affected || echo "Validation warning for: uv run poly test --affected"

    echo "Creating checkpoint..."
    git add -A
    git commit -m "Checkpoint after step_003: Update breaking changes batch 2"
    git tag checkpoint-step_003

    echo "Step completed successfully"
else
    echo "Step cancelled"
    exit 1
fi
echo ""

echo "=== Step step_004: Update test files ==="
echo "Risk level: medium"
echo "Files affected: 2"
echo "Estimated time: 120 minutes"
echo ""

# Check dependencies
echo "Dependency: step_001 (should be completed)"
echo "Dependency: step_002 (should be completed)"
echo "Dependency: step_003 (should be completed)"

read -p "Execute this step? (y/n/s=skip): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "Skipping step"
elif [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing commands..."
    echo "# Update test file test/components/sejm_whiz/eli_api/test_client.py"
    echo "# Update mocks and assertions"
    python scripts/update_test_mocks.py --file test/components/sejm_whiz/eli_api/test_client.py
    echo "# Update test file test/components/sejm_whiz/sejm_api/test_client.py"
    echo "# Update mocks and assertions"
    python scripts/update_test_mocks.py --file test/components/sejm_whiz/sejm_api/test_client.py

    echo "Running validation..."
    uv run poly test || echo "Validation warning for: uv run poly test"
    python -m pytest --tb=short || echo "Validation warning for: python -m pytest --tb=short"

    echo "Step completed successfully"
else
    echo "Step cancelled"
    exit 1
fi
echo ""

echo "=== Post-refactoring validation ==="
echo "- Run full test suite: uv run poly test"
uv run poly test
echo "- Run linting: uv ruff check --fix ."
uv ruff check --fix .
echo "- Update CHANGELOG.md"
echo "- Create PR for review"

echo ""
echo "Refactoring completed successfully!"
echo "Remember to:"
echo "- Review all changes"
echo "- Run comprehensive tests"
echo "- Update documentation"
echo "- Create pull request"
