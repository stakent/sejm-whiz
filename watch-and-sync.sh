#!/bin/bash
set -e

# Load configuration from config.sh
source ./config.sh 2>/dev/null || true

# Configuration - customize these paths for your environment
REPO_PATH="${REPO_PATH:-$(pwd)}"
MOUNT_PATH="${MOUNT_PATH:-/tmp/sejm-whiz}"
NAMESPACE="${NAMESPACE:-sejm-whiz}"
APP_LABEL="${APP_LABEL:-app=sejm-whiz-processor-gpu}"
WATCH_PATTERN="${WATCH_PATTERN:-*.py$}"

echo "🔍 Starting file watcher for Python files..."
echo "📁 Watching: $REPO_PATH"
echo "🎯 Syncing to: $MOUNT_PATH"
echo "🔍 Pattern: $WATCH_PATTERN"
echo "⚠️  Press Ctrl+C to stop"
echo ""

while inotifywait -r -e modify,create,delete --include="$WATCH_PATTERN" "$REPO_PATH/"; do
    echo "🔄 File changed, syncing..."
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' --exclude='*.pyc' "$REPO_PATH/" "$MOUNT_PATH/"
    echo "✅ Sync complete at $(date)"
    echo "💡 Run 'kubectl delete pod -n $NAMESPACE -l $APP_LABEL' to restart pod"
    echo ""
done
