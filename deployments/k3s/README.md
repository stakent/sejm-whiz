# K3s Deployment for Sejm-Whiz

This directory contains all k3s-specific deployment configurations for the Sejm-Whiz project, organized according to the hybrid deployment strategy outlined in `hybrid_deployment_summary.md`.

## Directory Structure

```
deployments/k3s/
├── manifests/          # Kubernetes YAML manifests
│   ├── k3s-processor-deployment-gpu.yaml  # GPU-enabled processor deployment
│   ├── k3s-processor-deployment-cpu.yaml  # CPU-only fallback deployment
│   ├── k3s-model-cache-pvc.yaml          # Persistent volume for model cache
│   └── nvidia-runtime-class.yaml          # NVIDIA runtime configuration
├── scripts/            # Deployment and validation scripts
│   ├── setup-gpu.sh    # Main GPU deployment script
│   └── gpu_validation.py # GPU validation and testing script
└── helm/               # Helm charts (future)
```

## Quick Start

### GPU-Enabled Deployment

From the project root directory:

```bash
# Run the complete GPU setup
./deployments/k3s/scripts/setup-gpu.sh
```

This script will:
1. Apply NVIDIA runtime class
2. Label GPU nodes
3. Create namespace and storage
4. Build GPU-enabled Docker image on p7
5. Import to k3s containerd
6. Deploy the processor with GPU support
7. Run validation tests

### Manual Deployment

```bash
# Apply individual components
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/nvidia-runtime-class.yaml
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-model-cache-pvc.yaml
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-processor-deployment-gpu.yaml
```

### Validation

Test GPU access in running pod:
```bash
POD_NAME=$(ssh root@p7 "kubectl get pods -n sejm-whiz -l app=sejm-whiz-processor-gpu -o jsonpath='{.items[0].metadata.name}'")
ssh root@p7 "kubectl exec -n sejm-whiz $POD_NAME -- nvidia-smi"
ssh root@p7 "kubectl exec -n sejm-whiz $POD_NAME -- python /app/deployments/k3s/scripts/gpu_validation.py"
```

## Key Components

### GPU Processor Deployment
- **File**: `manifests/k3s-processor-deployment-gpu.yaml`
- **Dockerfile**: `projects/data_processor/Dockerfile`
- **Image**: Uses NVIDIA CUDA 12.2 base image
- **Runtime**: `nvidia` runtime class for GPU access
- **Node Selection**: Targets nodes labeled with `gpu=gtx1060`

### Model Cache Storage
- **File**: `manifests/k3s-model-cache-pvc.yaml`
- **Size**: 10Gi persistent volume
- **Purpose**: Caches HerBERT models to avoid repeated downloads

### NVIDIA Runtime
- **File**: `manifests/nvidia-runtime-class.yaml`
- **Handler**: Enables NVIDIA container runtime for GPU access
- **Note**: Requires NVIDIA Container Toolkit on host

## Monitoring

Check deployment status:
```bash
ssh root@p7 "kubectl get pods -n sejm-whiz"
ssh root@p7 "kubectl logs -n sejm-whiz deployment/sejm-whiz-processor-gpu"
```

Monitor GPU usage:
```bash
ssh root@p7 "kubectl exec -n sejm-whiz deployment/sejm-whiz-processor-gpu -- nvidia-smi"
```

## Troubleshooting

### Pod in CrashLoopBackOff
- Check logs: `kubectl logs -n sejm-whiz <pod-name>`
- Verify image: `k3s ctr images ls | grep sejm-whiz`
- Check events: `kubectl describe pod -n sejm-whiz <pod-name>`

### GPU Not Available
- Verify runtime class: `kubectl get runtimeclass`
- Check node labels: `kubectl get nodes --show-labels`
- Test NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

### Image Pull Issues
- Import manually: `k3s ctr images import <image.tar>`
- Check disk space: `df -h /var/lib/rancher/k3s`

## Notes

- This deployment is designed for single-node k3s with GPU support
- Requires NVIDIA drivers and Container Toolkit on host
- Optimized for GTX 1060 6GB GPU
- Part of hybrid deployment strategy supporting multiple cloud providers