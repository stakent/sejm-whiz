# K3s Deployment Guide for sejm-whiz

This guide covers deployment to the k3s single-node cluster on the 'p7' host using Helm charts.

## Prerequisites

- Access to the 'p7' host with k3s cluster running
- `kubectl` configured to connect to the k3s cluster
- `helm` installed and configured
- GPU node scheduling configured for GTX 1060

## Deployment Architecture

The sejm-whiz application will be deployed as microservices on k3s:

```
k3s cluster (p7 host)
├── PostgreSQL with pgvector (persistent storage)
├── Redis (caching layer)
├── sejm-whiz-api (FastAPI web server)
├── sejm-whiz-processor (batch data processing)
└── GPU workloads (ML inference)
```

## Step 1: Deploy PostgreSQL with pgvector

### Install PostgreSQL Helm Chart

```bash
# Navigate to the project directory
cd /path/to/sejm-whiz-dev

# Deploy PostgreSQL with pgvector
helm install postgresql-pgvector ./helm/charts/postgresql-pgvector \
  --namespace sejm-whiz \
  --create-namespace \
  --values ./helm/charts/postgresql-pgvector/values.yaml

# Check deployment status
kubectl get pods -n sejm-whiz
kubectl get pvc -n sejm-whiz
```

### Verify PostgreSQL Installation

```bash
# Check if PostgreSQL is running
kubectl logs -n sejm-whiz deployment/postgresql-pgvector-postgresql

# Test pgvector extension
kubectl exec -n sejm-whiz deployment/postgresql-pgvector -- psql -U sejm_whiz_user -d sejm_whiz -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Database Connection Details

```yaml
# Connection configuration for applications
database:
  host: postgresql-pgvector.sejm-whiz.svc.cluster.local
  port: 5432
  database: sejm_whiz
  username: sejm_whiz_user
  password: sejm_whiz_password  # Use k8s secrets in production
```

## Step 2: Deploy Redis

```bash
# Deploy Redis (will be created in next step)
helm install redis ./helm/charts/redis \
  --namespace sejm-whiz \
  --values ./helm/charts/redis/values.yaml
```

## Step 3: Deploy sejm-whiz Applications

### API Server Deployment

````bash
# Build and push Docker image to local registry or registry accessible by k3s

### Option 1: Using k3s Built-in Registry (Recommended for Development)

k3s comes with containerd that can load images directly without a registry:

```bash
# Build the Docker image locally
docker build -t sejm-whiz-api:latest -f Dockerfile.api .
docker build -t sejm-whiz-processor:latest -f projects/data_processor/Dockerfile .

# Save images to tar files
docker save sejm-whiz-api:latest -o sejm-whiz-api.tar
docker save sejm-whiz-processor:latest -o sejm-whiz-processor.tar

# Copy images to k3s node (if building on different machine)
scp sejm-whiz-api.tar user@p7:/tmp/
scp sejm-whiz-processor.tar user@p7:/tmp/

# Import images directly into k3s containerd
sudo k3s ctr images import /tmp/sejm-whiz-api.tar
sudo k3s ctr images import /tmp/sejm-whiz-processor.tar

# Verify images are available
sudo k3s ctr images ls | grep sejm-whiz
````

### Option 2: Using Local Docker Registry

Set up a local registry accessible by k3s:

```bash
# Run a local registry container
docker run -d -p 5000:5000 --name registry registry:2

# Build and tag images for local registry
docker build -t localhost:5000/sejm-whiz-api:latest -f Dockerfile.api .
docker build -t localhost:5000/sejm-whiz-processor:latest -f projects/data_processor/Dockerfile .

# Push to local registry
docker push localhost:5000/sejm-whiz-api:latest
docker push localhost:5000/sejm-whiz-processor:latest

# Configure k3s to use insecure registry (add to /etc/rancher/k3s/registries.yaml)
sudo mkdir -p /etc/rancher/k3s
sudo tee /etc/rancher/k3s/registries.yaml << EOF
mirrors:
  "localhost:5000":
    endpoint:
      - "http://localhost:5000"
configs:
  "localhost:5000":
    tls:
      insecure_skip_verify: true
EOF

# Restart k3s to pick up registry config
sudo systemctl restart k3s
```

### Option 3: Using External Registry (Production)

For production deployments, use a proper container registry:

```bash
# Login to your registry (Docker Hub, GitHub Container Registry, etc.)
docker login ghcr.io -u your-username

# Build and tag for external registry
docker build -t ghcr.io/your-username/sejm-whiz-api:v1.0.0 -f Dockerfile.api .
docker build -t ghcr.io/your-username/sejm-whiz-processor:v1.0.0 -f projects/data_processor/Dockerfile .

# Push to external registry
docker push ghcr.io/your-username/sejm-whiz-api:v1.0.0
docker push ghcr.io/your-username/sejm-whiz-processor:v1.0.0

# Update Helm values to use external registry
# In helm/charts/sejm-whiz-api/values.yaml:
image:
  repository: ghcr.io/your-username/sejm-whiz-api
  tag: v1.0.0
  pullPolicy: Always
```

### Dockerfile Configuration

The project includes two Docker images for different services:

**Dockerfile.api** (for web API server):

```dockerfile
FROM python:3.12-slim-bookworm AS base

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev



FROM base
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Run the application using system Python
CMD ["python", "-m", "uvicorn", "bases.web_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**projects/data_processor/Dockerfile** (GPU-enabled data processing):

- Located in `projects/data_processor/Dockerfile`
- Uses NVIDIA CUDA 12.2 base image for GPU support
- Installs Python 3.12 via uv
- Configured for GPU acceleration with HerBERT embeddings

### Verify Image Deployment

```bash
# Check if images are available in k3s
kubectl run test-pod --image=sejm-whiz-api:latest --rm -it --restart=Never -- /bin/bash

# Or check with k3s directly
sudo k3s ctr images ls | grep sejm-whiz
```

# Then deploy the API server

helm install sejm-whiz-api ./helm/charts/sejm-whiz-api \
--namespace sejm-whiz \
--values ./helm/charts/sejm-whiz-api/values.yaml

````

### Data Processor Deployment

```bash
# Deploy the batch data processor
helm install sejm-whiz-processor ./helm/charts/sejm-whiz-processor \
  --namespace sejm-whiz \
  --values ./helm/charts/sejm-whiz-processor/values.yaml
````

## GPU Configuration for GTX 1060

### Node Labeling and GPU Scheduling

```bash
# Label the p7 node for GPU workloads
kubectl label nodes p7 gpu=gtx1060

# Verify GPU resources are available
kubectl describe node p7 | grep -A 5 "nvidia.com/gpu"
```

### GPU Resource Requests in Helm Values

```yaml
# In values.yaml for GPU-intensive components
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "2Gi"
  limits:
    nvidia.com/gpu: 1
    memory: "5Gi"  # Leave 1GB for system

nodeSelector:
  gpu: gtx1060
```

## Storage Configuration

### k3s Local Path Provisioner

The PostgreSQL chart uses k3s default storage class:

```yaml
postgresql:
  primary:
    persistence:
      storageClass: "local-path"  # k3s default
      size: 10Gi
```

### Persistent Volume Management

```bash
# Check storage usage
kubectl get pv
kubectl get pvc -n sejm-whiz

# Backup database (add to cron)
kubectl exec -n sejm-whiz deployment/postgresql-pgvector -- pg_dump -U sejm_whiz_user sejm_whiz > backup.sql
```

## Networking and Ingress

### Service Exposure

```bash
# Expose API server (example with NodePort for development)
kubectl patch service sejm-whiz-api -n sejm-whiz -p '{"spec":{"type":"NodePort"}}'

# Get external access
kubectl get service sejm-whiz-api -n sejm-whiz
```

### Traefik Ingress (k3s default)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sejm-whiz-ingress
  namespace: sejm-whiz
spec:
  rules:
  - host: sejm-whiz.p7.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sejm-whiz-api
            port:
              number: 8000
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check all services
kubectl get all -n sejm-whiz

# Check logs
kubectl logs -n sejm-whiz deployment/postgresql-pgvector
kubectl logs -n sejm-whiz deployment/sejm-whiz-api

# Check resource usage
kubectl top nodes
kubectl top pods -n sejm-whiz
```

### Database Maintenance

```bash
# Run database migrations
kubectl exec -n sejm-whiz deployment/sejm-whiz-api -- python -m alembic upgrade head

# Check vector index performance
kubectl exec -n sejm-whiz deployment/postgresql-pgvector -- psql -U sejm_whiz_user -d sejm_whiz -c "\\d+ legal_documents"
```

## Development Workflow

### Local Development with k3s

```bash
# Port forward for local development
kubectl port-forward -n sejm-whiz service/postgresql-pgvector 5432:5432
kubectl port-forward -n sejm-whiz service/redis 6379:6379

# Update local configuration
export DATABASE_URL="postgresql://sejm_whiz_user:sejm_whiz_password@localhost:5432/sejm_whiz"
export REDIS_URL="redis://localhost:6379"
```

### Deployment Updates

```bash
# Update Helm releases
helm upgrade postgresql-pgvector ./helm/charts/postgresql-pgvector \
  --namespace sejm-whiz \
  --values ./helm/charts/postgresql-pgvector/values.yaml

# Rollback if needed
helm rollback postgresql-pgvector 1 --namespace sejm-whiz
```

## Troubleshooting

### Common Issues

1. **Pod stuck in Pending**: Check node resources and scheduling constraints
1. **Database connection failed**: Verify service DNS and credentials
1. **GPU not available**: Check NVIDIA device plugin and node labels
1. **Storage issues**: Check PVC status and local-path provisioner

### Debug Commands

```bash
# Get detailed pod information
kubectl describe pod -n sejm-whiz <pod-name>

# Check events
kubectl get events -n sejm-whiz --sort-by=.metadata.creationTimestamp

# Test network connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup postgresql-pgvector.sejm-whiz.svc.cluster.local
```

## Security Considerations

### Production Hardening

```bash
# Use Kubernetes secrets for sensitive data
kubectl create secret generic postgres-credentials \
  --from-literal=password=your-secure-password \
  --namespace sejm-whiz

# Network policies (example)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sejm-whiz-network-policy
  namespace: sejm-whiz
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## Backup and Recovery

### Automated Backups

```bash
# Create backup job
kubectl create job postgres-backup --from=cronjob/postgres-backup -n sejm-whiz

# Restore from backup
kubectl exec -n sejm-whiz deployment/postgresql-pgvector -- psql -U sejm_whiz_user -d sejm_whiz < backup.sql
```

This deployment guide provides the foundation for running sejm-whiz on the k3s cluster at the 'p7' host with proper GPU support and vector database capabilities.
