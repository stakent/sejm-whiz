# PostgreSQL 17 Configuration for Debian 12 - Sejm-Whiz Infrastructure

> **Part of**: [Sejm-Whiz Baremetal Deployment](../README.md)
> **Target OS**: Debian 12 (Bookworm)
> **PostgreSQL Version**: 17.x with pgvector extension

## Overview

This document provides the complete PostgreSQL 17 configuration for Debian 12 hosts to handle the sejm-whiz AI workload with vector embeddings, external network connections, and production performance requirements.

**Target System**: Debian 12 server with PostgreSQL 17
**Use Case**: AI legal document analysis with vector embeddings (pgvector)
**Expected Workload**: ~500K+ documents, vector embeddings, ML model data
**Network Requirements**: Accept connections from sejm-whiz application hosts

## 1. Current Status Assessment

```bash
# Service status - ✅ RUNNING
systemctl status postgresql@17-main

# Listening interfaces - ❌ LOCALHOST ONLY
ss -tlnp | grep :5432
# Current: 127.0.0.1:5432 (localhost only)
# Required: 0.0.0.0:5432 (network accessible)

# Cluster info
pg_lsclusters
# Ver: 17, Cluster: main, Port: 5432, Status: online
```

## 2. Core Configuration Files

### 2.1 Main Configuration: `/etc/postgresql/17/main/postgresql.conf`

```bash
# Backup original configuration
sudo cp /etc/postgresql/17/main/postgresql.conf /etc/postgresql/17/main/postgresql.conf.backup

# Edit configuration
sudo nano /etc/postgresql/17/main/postgresql.conf
```

**Critical Settings for Network Access:**

```ini
#==============================================================================
# CONNECTION AND AUTHENTICATION
#==============================================================================

# Network configuration
listen_addresses = '*'                    # Accept connections from all interfaces
port = 5432                              # Standard PostgreSQL port

# Connection limits
max_connections = 200                     # Increased for AI workload
superuser_reserved_connections = 3

#==============================================================================
# RESOURCE USAGE (MEMORY)
#==============================================================================

# Memory configuration for AI/ML workload
shared_buffers = 2GB                     # 25% of total RAM (assuming 8GB+ system)
huge_pages = try                         # Use huge pages if available
temp_buffers = 32MB                      # Temporary buffer size

# Work memory for complex queries (vector operations)
work_mem = 256MB                         # Increased for vector operations
maintenance_work_mem = 512MB             # For index building and maintenance
autovacuum_work_mem = 512MB             # Vacuum operations

# Effective cache size (should be ~75% of total system RAM)
effective_cache_size = 6GB              # Adjust based on actual system RAM

#==============================================================================
# QUERY TUNING
#==============================================================================

# Cost-based optimizer settings for large datasets
random_page_cost = 1.1                  # SSD-optimized value
effective_io_concurrency = 200          # For SSD storage

# Parallelism for complex queries
max_worker_processes = 8                 # Match CPU cores
max_parallel_workers_per_gather = 4     # Parallel query execution
max_parallel_workers = 8                # Total parallel workers
max_parallel_maintenance_workers = 4    # For maintenance tasks

#==============================================================================
# WRITE AHEAD LOG (WAL)
#==============================================================================

# WAL configuration for performance and reliability
wal_level = replica                      # Enable replication capability
wal_buffers = 64MB                      # WAL buffer size
checkpoint_timeout = 15min              # Checkpoint frequency
checkpoint_completion_target = 0.9      # Spread checkpoints over time
max_wal_size = 4GB                      # Maximum WAL size
min_wal_size = 1GB                      # Minimum WAL size

# WAL archiving (for backup/recovery)
archive_mode = on                        # Enable WAL archiving
archive_command = 'test ! -f /var/lib/postgresql/17/archive/%f && cp %p /var/lib/postgresql/17/archive/%f'

#==============================================================================
# LOGGING
#==============================================================================

# Comprehensive logging for monitoring
logging_collector = on
log_destination = 'stderr'
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-17-main.log'
log_rotation_age = 1d
log_rotation_size = 100MB

# Query logging for performance analysis
log_min_duration_statement = 1000       # Log queries > 1 second
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on                     # Log lock waits for debugging
log_statement = 'ddl'                  # Log DDL statements
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

#==============================================================================
# AUTOVACUUM (Critical for AI workload)
#==============================================================================

# Aggressive autovacuum for heavy insert/update workload
autovacuum = on
autovacuum_max_workers = 6              # More workers for large tables
autovacuum_naptime = 30s                # Check more frequently
autovacuum_vacuum_threshold = 1000      # Start vacuum earlier
autovacuum_vacuum_scale_factor = 0.1    # More aggressive vacuum
autovacuum_analyze_threshold = 500      # Analyze threshold
autovacuum_analyze_scale_factor = 0.05  # More frequent analysis

#==============================================================================
# EXTENSIONS AND CUSTOM SETTINGS
#==============================================================================

# Shared libraries for extensions
shared_preload_libraries = 'pg_stat_statements,auto_explain,pgvector'

# pg_stat_statements for query performance monitoring
pg_stat_statements.max = 10000
pg_stat_statements.track = all

# Auto explain for slow queries
auto_explain.log_min_duration = 2000    # Log explain plans for queries > 2s
auto_explain.log_analyze = on
auto_explain.log_verbose = on
auto_explain.log_nested_statements = on
```

### 2.2 Client Authentication: `/etc/postgresql/17/main/pg_hba.conf`

```bash
# Backup original
sudo cp /etc/postgresql/17/main/pg_hba.conf /etc/postgresql/17/main/pg_hba.conf.backup

# Edit authentication rules
sudo nano /etc/postgresql/17/main/pg_hba.conf
```

**Authentication Configuration:**

```
# PostgreSQL Client Authentication Configuration File
# ===================================================

# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             postgres                                peer
local   all             all                                     peer

# IPv4 local connections:
host    all             all             127.0.0.1/32            scram-sha-256

# IPv4 connections for sejm-whiz application
# Allow connections from local network
host    sejm_whiz       sejm_whiz_user  192.168.0.0/24         scram-sha-256

# Allow specific client machine (x230)
host    sejm_whiz       sejm_whiz_user  192.168.0.0/24         scram-sha-256

# For development/testing - restrict to specific IPs in production
host    all             all             192.168.0.0/24         scram-sha-256

# IPv6 local connections:
host    all             all             ::1/128                 scram-sha-256

# Reject all other connections
host    all             all             0.0.0.0/0              reject
```

## 3. Database and User Setup

### 3.1 Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql

-- Create database user
CREATE USER sejm_whiz_user WITH PASSWORD 'sejm_whiz_password';

-- Create database
CREATE DATABASE sejm_whiz
    OWNER sejm_whiz_user
    ENCODING 'UTF8'
    LC_COLLATE='en_US.UTF-8'
    LC_CTYPE='en_US.UTF-8'
    TEMPLATE template0;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE sejm_whiz TO sejm_whiz_user;

-- Connect to sejm_whiz database
\c sejm_whiz;

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant usage on extension
GRANT ALL ON SCHEMA public TO sejm_whiz_user;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3.2 Performance Optimization Settings

```sql
-- Connect as superuser to sejm_whiz database
\c sejm_whiz postgres;

-- Optimize for vector operations
ALTER DATABASE sejm_whiz SET effective_cache_size = '6GB';
ALTER DATABASE sejm_whiz SET work_mem = '256MB';
ALTER DATABASE sejm_whiz SET maintenance_work_mem = '512MB';
ALTER DATABASE sejm_whiz SET shared_buffers = '2GB';

-- Optimize for AI workload patterns
ALTER DATABASE sejm_whiz SET random_page_cost = 1.1;
ALTER DATABASE sejm_whiz SET seq_page_cost = 1.0;
ALTER DATABASE sejm_whiz SET default_statistics_target = 500;  -- Better statistics for large tables
```

## 4. System Configuration

### 4.1 Firewall Configuration

```bash
# Check current firewall status
sudo ufw status

# Allow PostgreSQL from local network
sudo ufw allow from 192.168.0.0/24 to any port 5432

# Or use iptables if ufw is not used
sudo iptables -A INPUT -p tcp -s 192.168.0.0/24 --dport 5432 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### 4.2 System Kernel Parameters

```bash
# Edit kernel parameters for PostgreSQL optimization
sudo nano /etc/sysctl.conf

# Add these parameters:
```

```ini
# PostgreSQL optimization
# Shared memory settings
kernel.shmmax = 8589934592        # 8GB - adjust based on system RAM
kernel.shmall = 2097152          # 8GB in pages (8GB/4KB)

# Semaphore settings
kernel.sem = 250 512000 100 2048

# Network settings
net.ipv4.tcp_keepalive_time = 7200
net.ipv4.tcp_keepalive_probes = 9
net.ipv4.tcp_keepalive_intvl = 75

# Memory overcommit (important for large queries)
vm.overcommit_memory = 2
vm.overcommit_ratio = 80

# Hugepages (if you have sufficient RAM)
vm.nr_hugepages = 1024           # 2GB of hugepages (2MB each)
```

```bash
# Apply kernel parameters
sudo sysctl -p
```

### 4.3 PostgreSQL Service Limits

```bash
# Create systemd override directory
sudo mkdir -p /etc/systemd/system/postgresql@17-main.service.d

# Create limits override
sudo nano /etc/systemd/system/postgresql@17-main.service.d/limits.conf
```

```ini
[Service]
# Increase file limits for large datasets
LimitNOFILE=65536
LimitNPROC=32768

# Memory limits (set to system RAM - 1GB for OS)
LimitAS=infinity

# Core dump settings for debugging
LimitCORE=0
```

```bash
# Reload systemd and restart PostgreSQL
sudo systemctl daemon-reload
sudo systemctl restart postgresql@17-main
```

## 5. pgvector Extension Setup

### 5.1 Install pgvector

```bash
# Check if pgvector is already installed
sudo -u postgres psql -d sejm_whiz -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# If not installed, install it
sudo apt update
sudo apt install postgresql-17-pgvector

# Connect to database and create extension
sudo -u postgres psql -d sejm_whiz -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5.2 Vector Index Optimization

```sql
-- Connect to sejm_whiz database
\c sejm_whiz;

-- Set vector-specific parameters
SET ivfflat.probes = 10;        -- Default probes for IVFFlat index
SET hnsw.ef_search = 40;        -- HNSW search parameter

-- For better vector performance on large datasets
ALTER DATABASE sejm_whiz SET shared_buffers = '3GB';  -- Increased for vector operations
ALTER DATABASE sejm_whiz SET work_mem = '512MB';      -- Larger work_mem for vector operations
```

## 6. Monitoring and Maintenance

### 6.1 Performance Monitoring Setup

```bash
# Create archive directory for WAL files
sudo mkdir -p /var/lib/postgresql/17/archive
sudo chown postgres:postgres /var/lib/postgresql/17/archive
sudo chmod 755 /var/lib/postgresql/17/archive

# Create backup directory
sudo mkdir -p /var/backups/postgresql
sudo chown postgres:postgres /var/backups/postgresql

# Install monitoring extensions
sudo -u postgres psql -d sejm_whiz -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
```

### 6.2 Maintenance Scripts

```bash
# Create maintenance script
sudo nano /usr/local/bin/sejm-whiz-db-maintenance.sh
```

```bash
#!/bin/bash
# PostgreSQL maintenance script for sejm-whiz

LOG_FILE="/var/log/postgresql/maintenance.log"
DATE=$(date)

echo "[$DATE] Starting maintenance tasks" >> $LOG_FILE

# Vacuum and analyze critical tables
sudo -u postgres psql -d sejm_whiz << EOF
-- Analyze table statistics
ANALYZE legal_documents;
ANALYZE document_embeddings;

-- Reindex vector indexes if needed
REINDEX INDEX CONCURRENTLY idx_embeddings_vector;

-- Clean up old statistics
SELECT pg_stat_reset();

-- Check database size
SELECT
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
WHERE datname = 'sejm_whiz';
EOF

echo "[$DATE] Maintenance completed" >> $LOG_FILE
```

```bash
# Make script executable
sudo chmod +x /usr/local/bin/sejm-whiz-db-maintenance.sh

# Add to cron (run weekly)
sudo crontab -e
# Add line: 0 2 * * 0 /usr/local/bin/sejm-whiz-db-maintenance.sh
```

## 7. Apply Configuration and Restart

### 7.1 Configuration Validation and Restart

```bash
# Validate configuration syntax
sudo -u postgres pg_ctl configtest -D /var/lib/postgresql/17/main

# Restart PostgreSQL to apply all changes
sudo systemctl restart postgresql@17-main

# Verify service is running
sudo systemctl status postgresql@17-main

# Check listening interfaces (should now show 0.0.0.0:5432)
ss -tlnp | grep :5432

# Test local connection
sudo -u postgres psql -d sejm_whiz -c "SELECT version();"

# Check extensions
sudo -u postgres psql -d sejm_whiz -c "SELECT * FROM pg_extension;"
```

### 7.2 Connection Testing

```bash
# Test connection from remote host (x230)
psql -h p7 -U sejm_whiz_user -d sejm_whiz -c "SELECT current_database(), version();"

# Test vector extension
psql -h p7 -U sejm_whiz_user -d sejm_whiz -c "SELECT vector_dims(vector '[1,2,3]');"
```

## 8. Security Hardening

### 8.1 SSL Configuration (Recommended for Production)

```bash
# Generate self-signed certificate (or use proper CA-signed cert)
sudo -u postgres openssl req -new -x509 -days 365 -nodes -text \
  -out /var/lib/postgresql/17/main/server.crt \
  -keyout /var/lib/postgresql/17/main/server.key \
  -subj "/CN=p7"

# Set proper permissions
sudo -u postgres chmod 600 /var/lib/postgresql/17/main/server.key
sudo -u postgres chmod 644 /var/lib/postgresql/17/main/server.crt
```

Add to `postgresql.conf`:

```ini
# SSL Configuration
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

### 8.2 Connection Security

Update `pg_hba.conf` for SSL-only connections:

```
# Require SSL for remote connections
hostssl sejm_whiz       sejm_whiz_user  192.168.0.0/24         scram-sha-256
```

## 9. Troubleshooting Commands

```bash
# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-17-main.log

# Check connection limits
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check database performance
sudo -u postgres psql -d sejm_whiz -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check table sizes
sudo -u postgres psql -d sejm_whiz -c "
SELECT schemaname,tablename,attname,n_distinct,correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY tablename;
"

# Monitor vector operations
sudo -u postgres psql -d sejm_whiz -c "
SELECT count(*) as vector_embeddings_count
FROM document_embeddings
WHERE embedding IS NOT NULL;
"
```

## 10. Performance Validation

After configuration, run these tests to validate performance:

```sql
-- Connect to database
\c sejm_whiz;

-- Test vector operations performance
\timing on

-- Create test vector table
CREATE TABLE test_vectors (
    id SERIAL PRIMARY KEY,
    embedding vector(768)
);

-- Insert test vectors
INSERT INTO test_vectors (embedding)
SELECT random()::text::vector(768)
FROM generate_series(1, 1000);

-- Test vector similarity search
SELECT id, embedding <-> '[0.1,0.2,0.3,...]'::vector AS distance
FROM test_vectors
ORDER BY distance
LIMIT 10;

-- Clean up test
DROP TABLE test_vectors;
```

## Summary Checklist

- ✅ Configure `listen_addresses = '*'` in postgresql.conf
- ✅ Optimize memory settings for AI workload
- ✅ Configure pg_hba.conf for network access
- ✅ Create sejm_whiz database and user
- ✅ Install and configure pgvector extension
- ✅ Configure firewall rules
- ✅ Set system kernel parameters
- ✅ Restart PostgreSQL service
- ✅ Test remote connections
- ✅ Set up monitoring and maintenance

**Expected Result**: PostgreSQL will accept connections from the network on port 5432 with optimized performance for the sejm-whiz AI workload including vector embeddings.
