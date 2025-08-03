#!/bin/bash
# Web UI deployment script for sejm-whiz
# Builds and deploys the web UI monitoring dashboard
# Runs all commands on p7 host via SSH

set -e

echo "🌐 Setting up Sejm Whiz Web UI deployment"

# Check if we're in the right directory
if [[ ! -f "workspace.toml" ]]; then
    echo "❌ Error: Please run this script from the sejm-whiz project root"
    exit 1
fi

# 1. Create namespace if it doesn't exist
echo "📁 Ensuring sejm-whiz namespace..."
ssh root@p7 "kubectl create namespace sejm-whiz --dry-run=client -o yaml | kubectl apply -f -"

# 2. Transfer files to p7 for Docker build
echo "📁 Transferring project files to p7..."
rsync -avz --exclude='.git' --exclude='models' --exclude='.venv' . root@p7:/tmp/sejm-whiz/

# 3. Build Web UI Docker image on p7
echo "🔨 Building Web UI image on p7..."
echo "⏳ This may take a few minutes..."
ssh root@p7 "cd /tmp/sejm-whiz && timeout 900 docker build --progress=plain -t sejm-whiz-web-ui:latest -f projects/web_ui/Dockerfile . || (echo '❌ Docker build failed or timed out after 15 minutes' && exit 1)"
echo "✅ Docker build completed successfully"

# 4. Import image to k3s on p7
echo "📥 Importing image to k3s..."
echo "⏳ Saving Docker image..."
ssh root@p7 "docker save sejm-whiz-web-ui:latest -o /tmp/sejm-whiz-web-ui.tar"
echo "⏳ Importing to k3s containerd..."
ssh root@p7 "k3s ctr images import /tmp/sejm-whiz-web-ui.tar 2>&1 | while IFS= read -r line; do echo \"📦 \$line\"; done || (echo '❌ Image import failed' && exit 1)"
echo "🧹 Cleaning up temporary image file..."
ssh root@p7 "rm /tmp/sejm-whiz-web-ui.tar"
echo "✅ Image import complete"

# 5. Deploy Web UI
echo "🚀 Deploying Web UI..."
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-web-ui-deployment.yaml

# 6. Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
ssh root@p7 "kubectl wait --for=condition=available --timeout=300s deployment/sejm-whiz-web-ui -n sejm-whiz"

# 7. Check deployment status
echo "📊 Checking deployment status..."
ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui"

# 8. Get service information
echo "🌐 Getting service information..."
ssh root@p7 "kubectl get services -n sejm-whiz sejm-whiz-web-ui"

# 9. Test health endpoint
echo "🏥 Testing health endpoint..."
POD_NAME=$(ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui -o jsonpath='{.items[0].metadata.name}'")

if [[ -n "$POD_NAME" ]]; then
    echo "Testing health endpoint in pod: $POD_NAME"
    ssh root@p7 "kubectl exec -n sejm-whiz $POD_NAME -- curl -s http://localhost:8000/health | head -3"
else
    echo "⚠️  No running pod found for health check"
fi

echo "✅ Web UI deployment complete!"
echo ""
echo "📋 Access Information:"
echo "  - External URL: http://192.168.0.200:30800/"
echo "  - Dashboard: http://192.168.0.200:30800/dashboard"
echo "  - API Docs: http://192.168.0.200:30800/docs"
echo "  - Health: http://192.168.0.200:30800/health"
echo ""
echo "📋 Management Commands:"
echo "  - Monitor pod logs: ssh root@p7 'kubectl logs -n sejm-whiz deployment/sejm-whiz-web-ui -f'"
echo "  - Check pod status: ssh root@p7 'kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui'"
echo "  - Scale deployment: ssh root@p7 'kubectl scale deployment sejm-whiz-web-ui --replicas=1 -n sejm-whiz'"
echo "  - Clean up: ssh root@p7 'rm -rf /tmp/sejm-whiz'"