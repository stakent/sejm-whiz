#!/bin/bash
# Development Configuration
# Single source of truth for hot reload scripts

# Paths
export MOUNT_PATH="/tmp/sejm-whiz"

# Kubernetes Configuration
export NAMESPACE="sejm-whiz"
export APP_LABEL="app=sejm-whiz-processor-gpu"

# File Watching Configuration
export WATCH_PATTERN="*.py$|*.html$|*.css$|*.js$|*.jinja2$|*.j2$"
