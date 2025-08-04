#!/bin/bash
set -e

# Load configuration from config.sh
source ./config.sh 2>/dev/null || true

# Configuration - customize these paths for your environment
REPO_PATH="${REPO_PATH:-$(pwd)}"
MOUNT_PATH="${MOUNT_PATH:-/tmp/sejm-whiz}"
NAMESPACE="${NAMESPACE:-sejm-whiz}"
APP_LABEL="${APP_LABEL:-app=sejm-whiz-processor-gpu}"

echo "🔄 Syncing code to hot-reload mount..."
echo "📁 Source: $REPO_PATH"
echo "🎯 Target: $MOUNT_PATH"
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' --exclude='*.pyc' "$REPO_PATH/" "$MOUNT_PATH/"

echo "🔃 Restarting processor pod..."
kubectl delete pod -n "$NAMESPACE" -l "$APP_LABEL"

echo "⏳ Waiting for new pod..."
kubectl wait --for=condition=Ready pod -l "$APP_LABEL" -n "$NAMESPACE" --timeout=60s

echo "✅ Hot-reload complete! Tailing logs..."
kubectl logs -n "$NAMESPACE" -l "$APP_LABEL" --tail=10 -f
