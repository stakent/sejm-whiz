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

- Ubuntu/Debian system with systemd
- PostgreSQL server running (local or remote)
- Redis server running (local or remote)
- Python 3.12+
- Root access for installation

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
2. **Database connection issues**: Verify PostgreSQL settings in environment.env
3. **Permission errors**: Ensure sejm-whiz user has proper permissions
4. **Port conflicts**: Check if ports 8001/8002 are available

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