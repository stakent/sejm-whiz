#!/bin/sh

# find . -name '*.py' | entr ./run-tests.sh
find . -type f \( -name "*.py" -o -name "*.html" -o -name "*.toml" -o -name "*.sh" -o -name "Dockerfile.*" \) | entr ./rsync-files-to-p7.sh
