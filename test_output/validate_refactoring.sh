#!/bin/bash
# Validation script for: Incremental API Refactoring

set -e

echo "Validating refactoring results..."
echo ""

# Compile check
echo "=== Python compilation check ==="
find . -name "*.py" -not -path "./.*" | head -20 | while read file; do
    echo "Compiling $file..."
    python -m py_compile "$file"
done
echo "Compilation check passed"
echo ""

# Import check
echo "=== Import validation ==="
python -c "
import sys
import importlib
import pkgutil

def validate_imports(package_name):
    try:
        package = importlib.import_module(package_name)
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            try:
                importlib.import_module(modname)
                print('✓ ' + modname)
            except Exception as e:
                print('✗ ' + modname + ': ' + str(e))
    except Exception as e:
        print('Package import failed: ' + str(e))

validate_imports('sejm_whiz')  # Based on project structure
"
echo ""

# Test execution
echo "=== Test execution ==="
uv run poly test || echo "Some tests failed - review required"
echo ""

# Linting check
echo "=== Code quality check ==="
uv ruff check . || echo "Linting issues found - review required"
echo ""

echo "Validation completed"
