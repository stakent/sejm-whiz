#!/bin/bash
# Version Management Testing Script

echo "=== Semantic Versioning Test Script ==="
echo

echo "Current version in __init__.py:"
grep "__version__" components/sejm_whiz/__init__.py

echo
echo "=== Testing Commitizen ==="
echo "Commitizen version:"
uv run cz version 2>/dev/null || echo "Commitizen not configured yet"

echo
echo "=== Testing version bump commands (dry-run) ==="
echo "Note: These commands will be available after committing the semantic versioning setup"

echo
echo "Available commands after setup:"
echo "  uv run cz bump --increment PATCH    # 0.1.0 → 0.1.1"
echo "  uv run cz bump --increment MINOR    # 0.1.0 → 0.2.0"
echo "  uv run cz bump --increment MAJOR    # 0.1.0 → 1.0.0"
echo "  uv run cz bump                      # Auto-detect from commits"
echo
echo "  uv run bumpversion patch            # 0.1.0 → 0.1.1"
echo "  uv run bumpversion minor            # 0.1.0 → 0.2.0"
echo "  uv run bumpversion major            # 0.1.0 → 1.0.0"

echo
echo "=== Changelog Preview ==="
if [ -f "CHANGELOG.md" ]; then
    echo "CHANGELOG.md created successfully"
    head -20 CHANGELOG.md
else
    echo "CHANGELOG.md not found"
fi
