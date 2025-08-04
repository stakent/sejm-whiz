# Docker Compose Deployment

This directory contains Docker Compose configurations for different deployment scenarios of the sejm-whiz project.

## Available Configurations

### 1. Development (`docker-compose.dev.yml`)

**Purpose**: Local development with hot reload and debugging capabilities

**Features**:

- Source code mounted for hot reload
- **Configurable GPU/CPU processing** (GPU enabled by default)
- Development-friendly logging
- Port forwarding for all services

**Usage**:

```bash
# Navigate to docker deployment directory
cd deployments/docker

# Choose your development mode:
# Option 1: GPU development (default, recommended)
cp .env.dev.gpu .env

# Option 2: CPU-only development (lightweight)
cp .env.dev.cpu .env

# Option 3: Custom configuration
cp .env.example .env
# Edit .env with your preferred settings

# Start all services
docker compose -f docker-compose.dev.yml up -d

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Stop services
docker compose -f docker-compose.dev.yml down
```

**Exposed Ports**:

- `8000`: Web UI dashboard
- `5432`: PostgreSQL database
- `6379`: Redis cache

### 2. Production (`docker-compose.prod.yml`)

**Purpose**: Single-server production deployment with GPU support

**Features**:

- Optimized resource limits
- Health checks and auto-restart
- GPU support (when available)
- Environment variable configuration
- Persistent data volumes

**Usage**:

```bash
# Navigate to docker deployment directory
cd deployments/docker

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f processor

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### 3. CI Testing (`docker-compose.ci.yml`)

**Purpose**: Automated testing in CI/CD pipelines

**Features**:

- Minimal resource usage
- tmpfs storage for speed
- CPU-only processing
- Automated test execution

**Usage**:

```bash
# Navigate to docker deployment directory
cd deployments/docker

# Run tests
docker-compose -f docker-compose.ci.yml up --abort-on-container-exit

# Cleanup
docker-compose -f docker-compose.ci.yml down -v
```

## Services Overview

### PostgreSQL (`postgres`)

- **Image**: `pgvector/pgvector:pg17`
- **Purpose**: Primary database with vector extension
- **Extensions**: pgvector, uuid-ossp
- **Data**: Persistent volume for production, tmpfs for CI

### Redis (`redis`)

- **Image**: `redis:7.2-alpine`
- **Purpose**: Caching and job queues
- **Config**: Memory limits and LRU eviction

### Web UI (`web-ui`)

- **Build**: `projects/web_ui/Dockerfile`
- **Purpose**: Monitoring dashboard and API
- **Port**: 8000
- **Health Check**: `/health` endpoint

### Data Processor (`processor`)

- **Build**: `projects/data_processor/Dockerfile`
- **Purpose**: Batch processing and ML inference
- **GPU**: Configurable via environment variables

## Environment Configuration

### Development GPU/CPU Configuration

The development environment supports both GPU and CPU processing modes:

#### GPU Mode (Default, Recommended)

- **Faster processing** with CUDA acceleration
- **Better performance** for embedding generation
- **Production-like** environment

```bash
cd deployments/docker
cp .env.dev.gpu .env
docker compose -f docker-compose.dev.yml up -d
```

#### CPU Mode (Lightweight)

- **Faster startup** and lower resource usage
- **No GPU dependencies** required
- **Simpler debugging** environment

```bash
cd deployments/docker
cp .env.dev.cpu .env
docker compose -f docker-compose.dev.yml up -d
```

#### Custom Configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key variables:

- `DEV_EMBEDDING_DEVICE`: `cuda` (GPU) or `cpu` (CPU-only) for development
- `DEV_GPU_RUNTIME`: Set to `nvidia` for GPU support
- `DEV_GPU_COUNT`: Number of GPUs to use (0 for CPU-only)
- `POSTGRES_PASSWORD`: Database password
- `CUDA_VISIBLE_DEVICES`: GPU selection

## GPU Support

For GPU-enabled processing:

1. Install [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
1. Set environment variables in `.env`:
   ```
   EMBEDDING_DEVICE=cuda
   NVIDIA_VISIBLE_DEVICES=all
   CUDA_VISIBLE_DEVICES=0
   ```
1. Uncomment `runtime: nvidia` in production compose file

## Data Persistence

**Development**: Named volumes with `-dev` suffix
**Production**: Named volumes for data persistence
**CI**: tmpfs for speed (data not persisted)

## Monitoring

Access the web dashboard at `http://localhost:8000` to monitor:

- Processing pipeline status
- Document ingestion progress
- System health and logs
- Database and cache metrics

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change port mappings in compose file
1. **Memory issues**: Adjust resource limits in compose file
1. **GPU not detected**: Check NVIDIA runtime installation
1. **Database connection fails**: Wait for health checks to pass

### Useful Commands

```bash
# View service logs
docker-compose logs -f [service-name]

# Execute commands in running container
docker-compose exec web-ui bash

# Rebuild services
docker-compose build --no-cache

# Reset all data (WARNING: destructive)
docker-compose down -v
```

## Migration from k3s

If migrating from k3s deployment:

1. Export data from k3s PostgreSQL
1. Stop k3s services
1. Start docker-compose services
1. Import data to docker-compose PostgreSQL
1. Update any hardcoded service URLs

## Performance Tuning

### Database Optimization

- Adjust PostgreSQL `shared_buffers` and `work_mem`
- Configure connection pooling for high-concurrency workloads

### Redis Optimization

- Tune `maxmemory` and eviction policies
- Enable persistence if needed for production

### Processor Optimization

- Adjust resource limits based on available hardware
- Configure batch sizes for optimal GPU utilization
