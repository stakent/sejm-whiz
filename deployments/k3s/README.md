# K3s Deployment for Sejm-Whiz [WIP]

**⚠️ WORK IN PROGRESS**: This K3s deployment is under active development and may be incomplete or unstable. Consider using simpler deployment methods like Docker Compose for local development.

This directory contains all k3s-specific deployment configurations for the Sejm-Whiz project, organized according to the hybrid deployment strategy outlined in `hybrid_deployment_summary.md`.

## Directory Structure

```
deployments/k3s/
├── manifests/          # Kubernetes YAML manifests
│   ├── k3s-processor-deployment-gpu.yaml  # GPU-enabled processor deployment
│   ├── k3s-processor-deployment-cpu.yaml  # CPU-only fallback deployment
│   ├── k3s-web-ui-deployment.yaml         # Web UI monitoring dashboard
│   ├── k3s-model-cache-pvc.yaml          # Persistent volume for model cache
│   └── nvidia-runtime-class.yaml          # NVIDIA runtime configuration
├── scripts/            # Deployment and validation scripts
│   ├── setup-gpu.sh    # Main GPU deployment script
│   ├── setup-web-ui.sh # Web UI deployment script
│   ├── setup-production.sh # Production-ready deployment with monitoring
│   ├── run-migrations.sh # Database migration script
│   └── gpu_validation.py # GPU validation and testing script
└── helm/               # Helm charts (future)
```

## Quick Start

### Prerequisites - Database Setup

Before deploying the processor, ensure the database is properly initialized:

```bash
# Run database migrations to create required tables
./deployments/k3s/scripts/run-migrations.sh
```

This will create all necessary tables including:

- `legal_documents` - Main document storage
- `document_embeddings` - Vector embeddings with pgvector
- `legal_amendments` - Amendment tracking
- `cross_references` - Document relationships
- `prediction_models` - ML model metadata

### Production-Ready Deployment

From the project root directory:

```bash
# Run the complete production setup (recommended)
./deployments/k3s/scripts/setup-production.sh
```

This script provides:

1. Health checks and readiness probes for all components
1. Prometheus metrics collection and alerting rules
1. Error recovery mechanisms and restart policies
1. Resource optimization with GPU limits
1. Comprehensive validation and testing
1. Production monitoring endpoints

### GPU-Enabled Deployment (Basic)

For development/testing only:

```bash
# Run the basic GPU setup
./deployments/k3s/scripts/setup-gpu.sh
```

This script will:

1. Apply NVIDIA runtime class
1. Label GPU nodes
1. Create namespace and storage
1. Build GPU-enabled Docker image on p7
1. Import to k3s containerd
1. Deploy the processor with GPU support
1. Run validation tests
1. Optionally deploy Web UI monitoring dashboard

### Web UI Deployment

Deploy the monitoring dashboard separately:

```bash
# Deploy Web UI only
./deployments/k3s/scripts/setup-web-ui.sh
```

This will:

1. Build Web UI Docker image
1. Import to k3s containerd
1. Deploy Web UI with NodePort service
1. Test health endpoint

**Access URLs:**

- Home: http://192.168.0.200:30800/
- Dashboard: http://192.168.0.200:30800/dashboard
- API Docs: http://192.168.0.200:30800/docs
- Health: http://192.168.0.200:30800/health

### Manual Deployment

```bash
# Apply individual components
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/nvidia-runtime-class.yaml
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-model-cache-pvc.yaml
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-processor-deployment-gpu.yaml
ssh root@p7 "kubectl apply -f -" < deployments/k3s/manifests/k3s-web-ui-deployment.yaml
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

### Production Monitoring

Production deployment includes comprehensive monitoring:

```bash
# Check all deployment status
kubectl get all -n sejm-whiz

# Monitor pipeline processing (real-time logs)
curl -N http://192.168.0.200:30800/api/logs/stream

# Check health endpoints
curl http://192.168.0.200:30800/health

# View monitoring dashboard
firefox http://192.168.0.200:30800/dashboard
```

### Component Monitoring

Check individual components:

```bash
# Processor status and logs
kubectl get pods -n sejm-whiz -l app=sejm-whiz-processor-gpu
kubectl logs -f -n sejm-whiz deployment/sejm-whiz-processor-gpu

# Database status and documents
kubectl exec -n sejm-whiz deployment/postgresql-pgvector -- psql -U sejm_whiz_user -d sejm_whiz -c "SELECT COUNT(*) FROM legal_documents;"

# GPU utilization
kubectl exec -n sejm-whiz deployment/sejm-whiz-processor-gpu -- nvidia-smi

# Redis cache status
kubectl exec -n sejm-whiz deployment/redis -- redis-cli info memory
```

### Metrics Collection

If Prometheus is installed, metrics are automatically collected at:

- `/metrics` endpoint on processor (port 8080)
- ServiceMonitor and PodMonitor configured
- Alert rules for failures and resource usage

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
