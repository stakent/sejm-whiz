# Sejm-Whiz Baremetal Deployment

This directory contains scripts and configuration for deploying Sejm-Whiz directly on p7 server (baremetal) following the KISS principle.

## Overview

The baremetal deployment runs all projects as systemd services on the host system:

- **api_server**: FastAPI web service (port 8001)
- **data_processor**: Batch processing with systemd timer (every 6 hours)
- **web_ui**: Monitoring dashboard (port 8002)

## Directory Structure

```
deployments/baremetal/
├── config/
│   └── environment.env          # Environment configuration template
├── scripts/
│   ├── install-system.sh        # Initial system setup
│   ├── deploy-api-server.sh     # Deploy API server
│   ├── deploy-data-processor.sh # Deploy data processor
│   ├── deploy-web-ui.sh         # Deploy web UI
│   ├── deploy-all.sh            # Deploy all services
│   └── manage-services.sh       # Service management utilities
└── README.md                    # This file
```

## Prerequisites

- **Ubuntu/Debian 12 system** with systemd (tested on Debian 12)
- **PostgreSQL 17** with pgvector extension (see [Infrastructure Setup](#infrastructure-setup))
- **Redis 7.0+** server (see [Infrastructure Setup](#infrastructure-setup))
- Python 3.12+
- Root access for installation

## Infrastructure Setup

### Architecture Overview

The baremetal deployment requires two core infrastructure services:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   PostgreSQL    │    │      Redis      │
│     Host        │    │   Database      │    │     Cache       │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • API Server    │◄───┤ • Document data │    │ • Job queues    │
│ • Data Processor│    │ • Vector embeds │    │ • Search cache  │
│ • Web UI        │    │ • ML models     │◄───┤ • Sessions      │
│                 │    │ • pgvector ext  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Debian 12              Debian 12              Debian 12
```

### Infrastructure Components

For **Debian 12** hosts, comprehensive setup guides are provided:

#### Database Layer

- **[PostgreSQL 17 + pgvector Setup](docs/P7_POSTGRESQL_SETUP.md)**
  - Production-grade configuration for AI workloads
  - Network accessibility from application hosts
  - Performance tuning for vector operations (HerBERT embeddings)
  - Memory optimization for large document datasets
  - Security hardening and monitoring

#### Caching Layer

- **[Redis 7.0 Setup](docs/P7_REDIS_SETUP.md)**
  - High-performance configuration for caching and job queues
  - Network accessibility for distributed deployments
  - Memory management for AI cache patterns
  - Job queue optimization for document processing
  - System-level performance tuning

### Configuration Requirements

Both infrastructure components **must be configured** for network access if running on separate hosts:

| Component  | Network Config           | Performance Config   | Security Config     |
| ---------- | ------------------------ | -------------------- | ------------------- |
| PostgreSQL | `listen_addresses = '*'` | Memory tuning for AI | `pg_hba.conf` rules |
| Redis      | `bind 0.0.0.0 ::`        | Cache policies       | `protected-mode no` |

> **⚠️ Critical**: The default installations listen only on localhost. The setup guides include all necessary network configuration changes to enable distributed deployment.

### Tested Environment

- **OS**: Debian 12 (Bookworm)
- **PostgreSQL**: 17.5 with pgvector 0.8.0
- **Redis**: 7.0.15
- **Python**: 3.12+
- **Architecture**: x86_64

## Installation Steps

### 1. Copy Files to p7

First, copy the entire project to p7:

```bash
# From local development machine
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  /home/d/project/sejm-whiz/sejm-whiz-dev/ root@p7:/root/tmp/sejm-whiz-baremetal/
```

### 2. Initial System Setup

```bash
# Run as root on p7
ssh root@p7
cd /root/tmp/sejm-whiz-baremetal
./deployments/baremetal/scripts/install-system.sh
```

This script will:

- Create system user `sejm-whiz`
- Install system dependencies (Python 3.12, uv, etc.)
- Create directories (`/opt/sejm-whiz`, `/etc/sejm-whiz`, etc.)
- Copy source code
- Install Python dependencies
- Set up log rotation and firewall rules

### 3. Configure Environment

Edit the configuration file:

```bash
sudo vim /etc/sejm-whiz/environment.env
```

Update database and Redis connection settings:

```bash
DATABASE_HOST=localhost  # or your PostgreSQL server
DATABASE_PORT=5432
DATABASE_NAME=sejm_whiz
DATABASE_USER=sejm_whiz_user
DATABASE_PASSWORD=your_password

REDIS_HOST=localhost     # or your Redis server
REDIS_PORT=6379
```

### 4. Run Database Migrations

```bash
cd /opt/sejm-whiz
sudo -u sejm-whiz uv run alembic upgrade head
```

### 5. Deploy Services

Deploy all services at once:

```bash
sudo ./deployments/baremetal/scripts/deploy-all.sh
```

Or deploy individually:

```bash
sudo ./deployments/baremetal/scripts/deploy-api-server.sh
sudo ./deployments/baremetal/scripts/deploy-data-processor.sh
sudo ./deployments/baremetal/scripts/deploy-web-ui.sh
```

## Service Management

Use the management script for common operations:

```bash
# Check status of all services
sudo ./deployments/baremetal/scripts/manage-services.sh status

# Start all services
sudo ./deployments/baremetal/scripts/manage-services.sh start

# Stop all services
sudo ./deployments/baremetal/scripts/manage-services.sh stop

# Restart all services
sudo ./deployments/baremetal/scripts/manage-services.sh restart

# View recent logs
sudo ./deployments/baremetal/scripts/manage-services.sh logs

# Enable services to start on boot
sudo ./deployments/baremetal/scripts/manage-services.sh enable
```

Or use systemctl directly:

```bash
# Individual service management
sudo systemctl status sejm-whiz-api
sudo systemctl restart sejm-whiz-web-ui
sudo systemctl start sejm-whiz-processor.timer

# View logs
sudo journalctl -u sejm-whiz-api -f
sudo journalctl -u sejm-whiz-processor -n 50
```

## Service Details

### API Server (sejm-whiz-api)

- **Port**: 8001
- **Type**: Long-running service
- **Endpoints**: `/docs`, `/health`, etc.
- **Dependencies**: PostgreSQL, Redis

### Data Processor (sejm-whiz-processor)

- **Type**: Scheduled task (systemd timer)
- **Schedule**: Every 6 hours
- **Manual run**: `sudo systemctl start sejm-whiz-processor`
- **Dependencies**: PostgreSQL, Redis, external APIs

### Web UI (sejm-whiz-web-ui)

- **Port**: 8002
- **Type**: Long-running service
- **Endpoints**: `/dashboard`, `/home`, `/health`
- **Dependencies**: None (standalone)

## File Locations

- **Source code**: `/opt/sejm-whiz/`
- **Configuration**: `/etc/sejm-whiz/environment.env`
- **Logs**: `/var/log/sejm-whiz/` and `journalctl`
- **Data**: `/var/lib/sejm-whiz/`
- **Services**: `/etc/systemd/system/sejm-whiz-*`

## Access URLs

After deployment:

- **API Server**: http://p7:8001
- **API Documentation**: http://p7:8001/docs
- **Web UI Home**: http://p7:8002/home
- **Monitoring Dashboard**: http://p7:8002/dashboard

## Troubleshooting

### Check Service Status

```bash
sudo systemctl status sejm-whiz-api
sudo systemctl status sejm-whiz-processor.timer
sudo systemctl status sejm-whiz-web-ui
```

### View Logs

```bash
# Real-time logs
sudo journalctl -u sejm-whiz-api -f

# Recent logs
sudo journalctl -u sejm-whiz-processor -n 100

# All services logs
sudo ./deployments/baremetal/scripts/manage-services.sh logs
```

### Common Issues

1. **Service won't start**: Check environment configuration and dependencies
1. **Database connection issues**:
   - Verify PostgreSQL settings in environment.env
   - Ensure PostgreSQL is configured for network access (see [PostgreSQL Setup Guide](docs/P7_POSTGRESQL_SETUP.md))
   - Test connection: `psql -h <host> -U sejm_whiz_user -d sejm_whiz`
1. **Redis connection issues**:
   - Verify Redis settings in environment.env
   - Ensure Redis is configured for network access (see [Redis Setup Guide](docs/P7_REDIS_SETUP.md))
   - Test connection: `redis-cli -h <host> ping`
1. **Permission errors**: Ensure sejm-whiz user has proper permissions
1. **Port conflicts**: Check if ports 8001/8002 are available

### Infrastructure Troubleshooting

For database and cache issues, refer to the comprehensive troubleshooting sections in:

- [PostgreSQL Troubleshooting](docs/P7_POSTGRESQL_SETUP.md#troubleshooting-commands)
- [Redis Troubleshooting](docs/P7_REDIS_SETUP.md#troubleshooting-commands)

### Manual Service Updates

To update code without full redeployment:

```bash
# Copy new code
sudo rsync -av --delete projects/api_server/ /opt/sejm-whiz/projects/api_server/
sudo chown -R sejm-whiz:sejm-whiz /opt/sejm-whiz/

# Restart service
sudo systemctl restart sejm-whiz-api
```

## Monitoring

- **Service status**: Built into management script
- **Logs**: Available via journalctl and web dashboard
- **System resources**: Use htop, systemctl status
- **Application metrics**: Available at web dashboard (http://p7:8002/dashboard)

## Security Considerations

- Services run as non-root user `sejm-whiz`
- Configuration files have restricted permissions (640)
- Firewall rules are configured for required ports
- Log rotation is configured to prevent disk space issues
