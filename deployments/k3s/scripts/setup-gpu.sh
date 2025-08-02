#!/bin/bash
# GPU-enabled k3s setup script for sejm-whiz
# Based on working k3s-docker-gpu example
# Runs all commands on p7 host via SSH

set -e

echo "🚀 Setting up GPU-enabled sejm-whiz deployment"

# Check if we're in the right directory
if [[ ! -f "workspace.toml" ]]; then
    echo "❌ Error: Please run this script from the sejm-whiz project root"
    exit 1
fi

# 1. Apply NVIDIA runtime class
echo "📦 Applying NVIDIA runtime class..."
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/nvidia-runtime-class.yaml

# 2. Label the GPU node (assuming single node p7)
echo "🏷️  Labeling GPU node..."
ssh root@p7 "kubectl label nodes p7 gpu=gtx1060 --overwrite"

# 3. Create namespace if it doesn't exist
echo "📁 Ensuring sejm-whiz namespace..."
ssh root@p7 "kubectl create namespace sejm-whiz --dry-run=client -o yaml | kubectl apply -f -"

# 4. Apply model cache PVC
echo "💾 Setting up model cache storage..."
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-model-cache-pvc.yaml

# 5. Transfer files to p7 for Docker build
echo "📁 Transferring project files to p7..."
rsync -avz --exclude='.git' --exclude='models' --exclude='.venv' . root@p7:/tmp/sejm-whiz/

# 6. Build GPU-enabled Docker image on p7
echo "🔨 Building GPU-enabled processor image on p7..."
echo "⏳ This may take several minutes for CUDA base image..."
echo "💡 You can monitor progress on p7 with: ssh root@p7 'docker logs \$(docker ps -q --filter ancestor=nvidia/cuda)'"
ssh root@p7 "cd /tmp/sejm-whiz && timeout 1800 docker build --progress=plain -t sejm-whiz-processor:gpu-latest -f projects/data_processor/Dockerfile . || (echo '❌ Docker build failed or timed out after 30 minutes' && exit 1)"
echo "✅ Docker build completed successfully"

# 7. Import image to k3s on p7
echo "📥 Importing image to k3s..."
echo "⏳ Saving Docker image..."
ssh root@p7 "docker save sejm-whiz-processor:gpu-latest -o /tmp/sejm-whiz-processor-gpu.tar"
echo "⏳ Importing to k3s containerd..."
echo "💡 This may take a few minutes for large CUDA images..."
ssh root@p7 "k3s ctr images import /tmp/sejm-whiz-processor-gpu.tar 2>&1 | while IFS= read -r line; do echo \"📦 \$line\"; done || (echo '❌ Image import failed' && exit 1)"
echo "🧹 Cleaning up temporary image file..."
ssh root@p7 "rm /tmp/sejm-whiz-processor-gpu.tar"
echo "✅ Image import complete"

# 8. Deploy GPU-enabled processor
echo "🚀 Deploying GPU-enabled processor..."
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-processor-deployment-gpu.yaml

# 9. Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
ssh root@p7 "kubectl wait --for=condition=available --timeout=300s deployment/sejm-whiz-processor-gpu -n sejm-whiz"

# 10. Check deployment status
echo "📊 Checking deployment status..."
ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-processor-gpu"

# 11. Run GPU validation test
echo "🧪 Running GPU validation test..."
POD_NAME=$(ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-processor-gpu -o jsonpath='{.items[0].metadata.name}'")

if [[ -n "$POD_NAME" ]]; then
    echo "Running GPU validation in pod: $POD_NAME"
    ssh root@p7 "kubectl exec -n sejm-whiz $POD_NAME -- python /app/deployments/k3s/scripts/gpu_validation.py"
else
    echo "⚠️  No running pod found for GPU validation"
fi

echo "✅ GPU-enabled sejm-whiz deployment complete!"
echo ""
echo "📋 Next steps:"
echo "  - Monitor pod logs: ssh root@p7 'kubectl logs -n sejm-whiz deployment/sejm-whiz-processor-gpu -f'"
echo "  - Check GPU utilization: ssh root@p7 'kubectl exec -n sejm-whiz $POD_NAME -- nvidia-smi'"
echo "  - Scale deployment: ssh root@p7 'kubectl scale deployment sejm-whiz-processor-gpu --replicas=1 -n sejm-whiz'"
echo "  - Clean up: ssh root@p7 'rm -rf /tmp/sejm-whiz'"