# Baremetal Infrastructure Documentation

This directory contains comprehensive setup guides for deploying Sejm-Whiz infrastructure on Debian 12 systems.

## Infrastructure Components

### Database Layer

- **[PostgreSQL 17 + pgvector Setup](P7_POSTGRESQL_SETUP.md)**
  - Complete PostgreSQL configuration for AI workloads
  - Network accessibility configuration
  - Performance optimization for vector embeddings
  - Security hardening and monitoring setup
  - Production-ready configuration for document analysis

### Caching Layer

- **[Redis 7.0 Setup](P7_REDIS_SETUP.md)**
  - High-performance Redis configuration
  - Network accessibility for distributed deployments
  - Memory management and eviction policies
  - Job queue optimization
  - System-level performance tuning

## Quick Reference

### Critical Configuration Changes

Both services require network configuration to accept external connections:

#### PostgreSQL

```bash
# /etc/postgresql/17/main/postgresql.conf
listen_addresses = '*'

# /etc/postgresql/17/main/pg_hba.conf
host all all 192.168.0.0/24 scram-sha-256
```

#### Redis

```bash
# /etc/redis/redis.conf
bind 0.0.0.0 ::
protected-mode no
```

### Service Commands

#### PostgreSQL

```bash
# Service management
sudo systemctl status postgresql@17-main
sudo systemctl restart postgresql@17-main

# Connection testing
psql -h <host> -U sejm_whiz_user -d sejm_whiz

# Check listening interfaces
ss -tlnp | grep :5432
```

#### Redis

```bash
# Service management
sudo systemctl status redis-server
sudo systemctl restart redis-server

# Connection testing
redis-cli -h <host> ping

# Check listening interfaces
ss -tlnp | grep :6379
```

## Architecture Context

These infrastructure components support the Sejm-Whiz AI legal document analysis system:

- **PostgreSQL**: Stores legal documents, vector embeddings, ML models
- **Redis**: Provides caching, job queues, session storage
- **Application Layer**: FastAPI services, data processors, web UI

## Target Environment

- **Operating System**: Debian 12 (Bookworm)
- **Architecture**: x86_64
- **Network**: Internal network deployment with cross-host communication
- **Workload**: AI/ML document processing with vector similarity search

## Related Documentation

- [Baremetal Deployment Guide](../README.md) - Complete deployment instructions
- [Project Overview](../../../README.md) - Main project documentation
- [CLI Management](../../../CLI_README.md) - Command-line tools for administration
