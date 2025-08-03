#!/bin/bash
# Development deployment script for web UI with hot reload
# No Docker build needed - uses volume mounts for instant updates

set -e

echo "🚀 Setting up Web UI development deployment with hot reload"

# Check if we're in the right directory
if [[ ! -f "workspace.toml" ]]; then
    echo "❌ Error: Please run this script from the sejm-whiz project root"
    exit 1
fi

# 1. Initial sync of files to p7
echo "📁 Performing initial sync of project files to p7..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='models' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    . root@p7:/tmp/sejm-whiz/

# 2. Create namespace if it doesn't exist
echo "📁 Ensuring sejm-whiz namespace..."
ssh root@p7 "kubectl create namespace sejm-whiz --dry-run=client -o yaml | kubectl apply -f -"

# 3. Delete existing production deployment if exists
echo "🔄 Removing production deployment if exists..."
ssh root@p7 "kubectl delete deployment sejm-whiz-web-ui -n sejm-whiz --ignore-not-found=true"

# 4. Deploy development version with volume mounts
echo "🚀 Deploying development web UI with hot reload..."
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-web-ui-deployment-dev.yaml

# 5. Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready (may take a minute for pip install)..."
ssh root@p7 "kubectl wait --for=condition=available --timeout=120s deployment/sejm-whiz-web-ui-dev -n sejm-whiz" || true

# 6. Check deployment status
echo "📊 Checking deployment status..."
ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui-dev"

# 7. Show logs to see startup
echo "📋 Showing startup logs..."
POD_NAME=$(ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui-dev -o jsonpath='{.items[0].metadata.name}'")
if [[ -n "$POD_NAME" ]]; then
    echo "Pod: $POD_NAME"
    ssh root@p7 "kubectl logs -n sejm-whiz $POD_NAME --tail=20"
fi

echo ""
echo "✅ Development deployment complete!"
echo ""
echo "📋 Access Information:"
echo "  - Dev URL: http://192.168.0.200:30801/"
echo "  - Dashboard: http://192.168.0.200:30801/dashboard"
echo "  - API Docs: http://192.168.0.200:30801/docs"
echo "  - Health: http://192.168.0.200:30801/health"
echo ""
echo "🔄 Hot Reload Workflow:"
echo "  1. Edit files locally in projects/web_ui/"
echo "  2. Run: ./deployments/k3s/scripts/sync-web-ui.sh"
echo "  3. Changes appear instantly (uvicorn auto-reloads)"
echo ""
echo "📋 Useful Commands:"
echo "  - Sync changes: ./deployments/k3s/scripts/sync-web-ui.sh"
echo "  - View logs: ssh root@p7 'kubectl logs -n sejm-whiz deployment/sejm-whiz-web-ui-dev -f'"
echo "  - Restart pod: ssh root@p7 'kubectl rollout restart deployment/sejm-whiz-web-ui-dev -n sejm-whiz'"
echo ""
echo "⚡ No more Docker builds! Just sync and refresh browser!"