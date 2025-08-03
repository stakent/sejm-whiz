#!/bin/sh

# print empty lines to mark start of new run
# echo "********************************************************"
echo "\n\n\n\n\n\n\n\n\n\n\n\n\n"

rsync \
    -rv \
    --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=.git \
    . d@p7:`pwd`
