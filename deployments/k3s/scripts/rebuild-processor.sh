#!/bin/bash
# Rebuild and redeploy processor with pipeline fixes

set -e

echo "🔨 Rebuilding processor with pipeline fixes..."

# Build updated processor image on p7
echo "Building processor image on p7..."
ssh root@p7 "cd /tmp/sejm-whiz && docker build -t sejm-whiz-processor:optimized -f projects/data_processor/Dockerfile ."

# Import to k3s containerd
echo "Importing image to k3s..."
ssh root@p7 "docker save sejm-whiz-processor:optimized | k3s ctr images import -"

# Delete current processor pod to trigger restart with new image
echo "Restarting processor deployment..."
kubectl delete pod -n sejm-whiz -l app=sejm-whiz-processor-gpu

echo "✅ Processor rebuild and restart completed!"
echo "Monitor logs with: kubectl logs -f -n sejm-whiz deployment/sejm-whiz-processor-gpu"
