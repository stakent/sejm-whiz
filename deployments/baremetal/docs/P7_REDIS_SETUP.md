# Redis 7.0 Configuration for Debian 12 - Sejm-Whiz Infrastructure

> **Part of**: [Sejm-Whiz Baremetal Deployment](../README.md)
> **Target OS**: Debian 12 (Bookworm)
> **Redis Version**: 7.0.x

## Overview

This document provides the complete Redis 7.0 configuration for Debian 12 hosts to handle the sejm-whiz AI workload with high-performance caching, job queues, and external network connections.

**Target System**: Debian 12 server with Redis 7.0+
**Use Case**: AI document processing cache, job queues, session storage
**Expected Workload**: High-throughput caching, vector search results, document metadata
**Network Requirements**: Accept connections from sejm-whiz application hosts

## 1. Current Status Assessment

```bash
# Service status - ❌ NOT RUNNING
systemctl status redis-server
# Current: inactive (dead), disabled

# Installation status - ✅ INSTALLED
redis-server --version
# Redis server v=7.0.15

# Network access - ❌ NOT ACCESSIBLE
ss -tlnp | grep :6379
# Expected: No output (not running)
```

## 2. Core Configuration Files

### 2.1 Main Configuration: `/etc/redis/redis.conf`

```bash
# Backup original configuration
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup

# Edit configuration
sudo nano /etc/redis/redis.conf
```

**Critical Settings for Network Access and AI Workload:**

```ini
################################## NETWORK #####################################

# Accept connections from all interfaces (CRITICAL for network access)
bind 0.0.0.0 ::
# Default: bind 127.0.0.1 ::1 (localhost only)
# Change to: bind 0.0.0.0 :: (all interfaces)

# Standard Redis port
port 6379

# TCP listen backlog
tcp-backlog 511

# Close connection after client is idle for N seconds (0 = disable)
timeout 300

# TCP keepalive
tcp-keepalive 300

################################# TLS/SSL ######################################

# Disable TLS for internal network (can enable later if needed)
# tls-port 0
# port 6379

################################# GENERAL #####################################

# Run as daemon
daemonize yes

# Set server verbosity to 'notice'
loglevel notice

# Log file location
logfile /var/log/redis/redis-server.log

# Set the number of databases (default 16)
databases 16

# Show Redis logo at startup
always-show-logo yes

################################ SNAPSHOTTING  ################################

# Save snapshots for persistence
# Format: save <seconds> <changes>
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Compress string objects in RDB snapshots
rdbcompression yes

# Checksum RDB files
rdbchecksum yes

# RDB file name
dbfilename dump.rdb

# RDB and AOF file directory
dir /var/lib/redis

################################# REPLICATION #################################

# Replication settings (for future scaling)
# replica-serve-stale-data yes
# replica-read-only yes

################################## SECURITY ###################################

# Require password for connections
# Uncomment and set strong password for production
# requirepass your_strong_password_here

# Command renaming for security (optional)
# rename-command FLUSHDB ""
# rename-command FLUSHALL ""

################################### CLIENTS ####################################

# Max number of connected clients
maxclients 10000

############################## MEMORY MANAGEMENT #############################

# Maximum memory usage (set to ~80% of available RAM for Redis)
# For 8GB system: set to ~2GB for Redis (rest for OS, PostgreSQL, etc.)
maxmemory 2gb

# Memory policy when max memory reached
# allkeys-lru: Remove any key using LRU algorithm (good for cache)
# volatile-lru: Remove keys with TTL using LRU (good for mixed workload)
maxmemory-policy allkeys-lru

# LRU sampling size
maxmemory-samples 5

############################# LAZY FREEING ####################################

# Lazy freeing for better performance
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes

############################ KERNEL OOM CONTROL ##############################

# OOM score adjustment
oom-score-adj no

############################ KERNEL TRANSPARENT HUGEPAGE #################

# Disable transparent huge pages for better latency
disable-thp yes

############################## APPEND ONLY FILE ###############################

# Enable AOF for better durability
appendonly yes

# AOF filename
appendfilename "appendonly.aof"

# AOF sync policy
# always: Sync after every write (slow, very durable)
# everysec: Sync every second (balance of performance and durability)
# no: Let OS decide when to sync (fast, less durable)
appendfsync everysec

# Don't sync AOF during rewrite
no-appendfsync-on-rewrite no

# Auto AOF rewrite
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Load truncated AOF on startup
aof-load-truncated yes

# AOF and RDB mixed format
aof-use-rdb-preamble yes

################################ LUA SCRIPTING  ###############################

# Lua time limit in milliseconds
lua-time-limit 5000

################################## SLOW LOG ###################################

# Slow query log settings
slowlog-log-slower-than 10000  # Log queries slower than 10ms
slowlog-max-len 128            # Keep last 128 slow queries

################################ LATENCY MONITOR ##############################

# Latency monitoring
latency-monitor-threshold 100   # Log events slower than 100ms

################################ EVENT NOTIFICATION ######################

# Keyspace notifications (useful for monitoring)
# K: Keyspace events
# E: Keyevent events
# g: Generic commands (DEL, EXPIRE, RENAME, etc.)
# x: Expired events
# A: Alias for "g$lshzxe"
notify-keyspace-events "Egx"

############################### ADVANCED CONFIG ###########################

# Hash table rehashing
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

# List compression
list-max-ziplist-size -2
list-compress-depth 0

# Set optimization
set-max-intset-entries 512

# Sorted set optimization
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# HyperLogLog sparse representation
hll-sparse-max-bytes 3000

# Streams optimization
stream-node-max-bytes 4096
stream-node-max-entries 100

# Active rehashing
activerehashing yes

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Client query buffer limit
client-query-buffer-limit 1gb

# Protocol buffer limit
proto-max-bulk-len 512mb

# Frequency of background tasks
hz 10

# Dynamic hz adjustment
dynamic-hz yes

# AOF rewrite incremental fsync
aof-rewrite-incremental-fsync yes

# RDB save incremental fsync
rdb-save-incremental-fsync yes
```

### 2.2 System Service Configuration

```bash
# Create systemd override directory
sudo mkdir -p /etc/systemd/system/redis-server.service.d

# Create performance override
sudo nano /etc/systemd/system/redis-server.service.d/override.conf
```

```ini
[Unit]
Description=Advanced key-value store for sejm-whiz AI workload
After=network.target

[Service]
# User and group
User=redis
Group=redis

# Process limits for high-performance workload
LimitNOFILE=65536
LimitNPROC=32768
LimitCORE=0

# Memory limits
LimitAS=4G

# Security settings
NoNewPrivileges=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/redis /var/log/redis

# Performance settings
IOSchedulingClass=1
IOSchedulingPriority=4
CPUSchedulingPolicy=2

# Restart policy
Restart=always
RestartSec=3
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
```

## 3. System Optimization

### 3.1 Kernel Parameters for Redis

```bash
# Edit kernel parameters
sudo nano /etc/sysctl.conf

# Add Redis-specific optimizations:
```

```ini
# Redis kernel optimizations
# Memory overcommit (important for Redis fork operations)
vm.overcommit_memory = 1

# Swappiness (reduce swapping for better performance)
vm.swappiness = 1

# TCP settings for high-concurrency Redis
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# File descriptor limits
fs.file-max = 2097152

# Network buffer sizes
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216

# TCP buffer sizes
net.ipv4.tcp_rmem = 4096 12288 16777216
net.ipv4.tcp_wmem = 4096 12288 16777216

# Disable transparent hugepages (recommended for Redis)
# This will be handled by Redis config, but can also be set globally
```

```bash
# Apply kernel parameters
sudo sysctl -p

# Disable transparent hugepages immediately
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Make transparent hugepage disable persistent
echo 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' | sudo tee -a /etc/rc.local
echo 'echo never > /sys/kernel/mm/transparent_hugepage/defrag' | sudo tee -a /etc/rc.local
sudo chmod +x /etc/rc.local
```

### 3.2 User Limits Configuration

```bash
# Edit user limits for redis user
sudo nano /etc/security/limits.conf

# Add Redis user limits:
```

```
# Redis user limits
redis soft nofile 65536
redis hard nofile 65536
redis soft nproc 32768
redis hard nproc 32768
redis soft memlock unlimited
redis hard memlock unlimited
```

### 3.3 Directory and Log Setup

```bash
# Ensure Redis directories exist with correct permissions
sudo mkdir -p /var/lib/redis
sudo mkdir -p /var/log/redis
sudo chown redis:redis /var/lib/redis
sudo chown redis:redis /var/log/redis
sudo chmod 755 /var/lib/redis
sudo chmod 755 /var/log/redis

# Create log rotation for Redis logs
sudo nano /etc/logrotate.d/redis-server
```

```
/var/log/redis/redis-server.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload redis-server
    endscript
}
```

## 4. Firewall Configuration

### 4.1 Allow Redis Port from Local Network

```bash
# Check current firewall status
sudo ufw status

# Allow Redis from local network
sudo ufw allow from 192.168.0.0/24 to any port 6379

# Or use iptables if ufw is not used
sudo iptables -A INPUT -p tcp -s 192.168.0.0/24 --dport 6379 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

## 5. Performance Tuning for AI Workload

### 5.1 Memory Management Strategy

Based on sejm-whiz workload patterns:

```bash
# Connect to Redis and configure runtime settings
redis-cli << 'EOF'
# Set memory policy for cache workload
CONFIG SET maxmemory-policy allkeys-lru

# Optimize for mixed workload (cache + job queue)
CONFIG SET maxmemory 2147483648  # 2GB

# Optimize hash tables for document metadata
CONFIG SET hash-max-ziplist-entries 512
CONFIG SET hash-max-ziplist-value 64

# Optimize lists for job queues
CONFIG SET list-max-ziplist-size -2

# Save configuration
CONFIG REWRITE
EOF
```

### 5.2 Monitoring Configuration

```bash
# Create Redis monitoring script
sudo nano /usr/local/bin/redis-monitor.sh
```

```bash
#!/bin/bash
# Redis monitoring script for sejm-whiz

LOG_FILE="/var/log/redis/monitor.log"
DATE=$(date)

echo "[$DATE] Redis monitoring check" >> $LOG_FILE

# Check Redis status
redis-cli ping >> $LOG_FILE 2>&1

# Memory usage
redis-cli info memory | grep -E 'used_memory_human|used_memory_peak_human|mem_fragmentation_ratio' >> $LOG_FILE

# Connection stats
redis-cli info clients | grep -E 'connected_clients|blocked_clients' >> $LOG_FILE

# Command stats
redis-cli info commandstats | head -10 >> $LOG_FILE

# Slow log
echo "Recent slow queries:" >> $LOG_FILE
redis-cli slowlog get 5 >> $LOG_FILE

echo "[$DATE] Monitoring complete" >> $LOG_FILE
echo "---" >> $LOG_FILE
```

```bash
# Make script executable
sudo chmod +x /usr/local/bin/redis-monitor.sh

# Add to cron (run every 5 minutes)
echo "*/5 * * * * /usr/local/bin/redis-monitor.sh" | sudo crontab -
```

## 6. Sejm-Whiz Specific Configuration

### 6.1 Database Structure Setup

```bash
# Connect to Redis and set up databases for different purposes
redis-cli << 'EOF'
# Database 0: General cache (default)
SELECT 0
INFO keyspace

# Database 1: Session storage
SELECT 1

# Database 2: Job queue data
SELECT 2

# Database 3: Vector search cache
SELECT 3

# Database 4: Document metadata cache
SELECT 4

# Return to default database
SELECT 0
EOF
```

### 6.2 Key Naming Conventions

Document the key naming strategy for the sejm-whiz application:

```
# Cache keys
sejm:cache:documents:{doc_id}
sejm:cache:embeddings:{doc_id}
sejm:cache:search:{query_hash}

# Job queue keys
sejm:jobs:pending
sejm:jobs:processing
sejm:jobs:completed
sejm:jobs:failed

# Session keys
sejm:session:{session_id}

# Stats and metrics
sejm:stats:daily:{date}
sejm:stats:hourly:{date}_{hour}
```

## 7. Security Configuration

### 7.1 Basic Authentication (Optional for internal network)

```bash
# If you want to add password authentication, uncomment in redis.conf:
# requirepass your_strong_password_here

# Generate strong password
openssl rand -base64 32
```

### 7.2 Command Restrictions (Production Hardening)

For production, consider disabling dangerous commands:

```ini
# In redis.conf, rename or disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
rename-command CONFIG "CONFIG_a8f5e3b2"  # Rename with random suffix
```

## 8. Start and Enable Redis Service

### 8.1 Service Management

```bash
# Reload systemd to pick up changes
sudo systemctl daemon-reload

# Enable Redis to start on boot
sudo systemctl enable redis-server

# Start Redis service
sudo systemctl start redis-server

# Check service status
sudo systemctl status redis-server

# Verify Redis is listening on all interfaces
ss -tlnp | grep :6379
# Expected: 0.0.0.0:6379

# Test local connection
redis-cli ping
# Expected: PONG

# Test remote connection (from x230)
redis-cli -h p7 ping
# Expected: PONG
```

## 9. Performance Validation and Benchmarks

### 9.1 Basic Performance Tests

```bash
# Run Redis benchmark for AI workload patterns
redis-benchmark -h p7 -t SET,GET,LPUSH,LPOP,HSET,HGET -n 100000 -c 50

# Test specific workload patterns
redis-benchmark -h p7 -t SET -n 10000 -d 1024  # 1KB values (document metadata)
redis-benchmark -h p7 -t SET -n 1000 -d 10240  # 10KB values (embeddings)
```

### 9.2 Memory Usage Validation

```bash
# Connect and check memory configuration
redis-cli -h p7 << 'EOF'
INFO memory
CONFIG GET maxmemory
CONFIG GET maxmemory-policy
EOF
```

## 10. Backup and Maintenance

### 10.1 Backup Strategy

```bash
# Create backup script
sudo nano /usr/local/bin/redis-backup.sh
```

```bash
#!/bin/bash
# Redis backup script for sejm-whiz

BACKUP_DIR="/var/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/redis/backup.log"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create RDB snapshot
redis-cli BGSAVE

# Wait for background save to complete
while [ $(redis-cli LASTSAVE) -eq $(redis-cli LASTSAVE) ]; do
    sleep 1
done

# Copy RDB file with timestamp
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump_$DATE.rdb

# Compress backup
gzip $BACKUP_DIR/dump_$DATE.rdb

# Keep only last 7 days of backups
find $BACKUP_DIR -name "dump_*.rdb.gz" -mtime +7 -delete

echo "[$DATE] Redis backup completed" >> $LOG_FILE
```

```bash
# Make executable and add to cron
sudo chmod +x /usr/local/bin/redis-backup.sh
echo "0 2 * * * /usr/local/bin/redis-backup.sh" | sudo crontab -
```

## 11. Troubleshooting Commands

```bash
# Check Redis process
ps aux | grep redis

# Check listening ports
ss -tlnp | grep 6379

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log

# Monitor Redis in real-time
redis-cli monitor

# Check slow queries
redis-cli slowlog get 10

# Check memory usage
redis-cli info memory

# Check connected clients
redis-cli client list

# Test connection and performance
redis-cli --latency -h p7
redis-cli --latency-history -h p7

# Check configuration
redis-cli config get "*"
```

## 12. Monitoring and Alerting

### 12.1 Key Metrics to Monitor

```bash
# Memory usage
redis-cli info memory | grep used_memory_human

# Connection count
redis-cli info clients | grep connected_clients

# Operations per second
redis-cli info stats | grep instantaneous_ops_per_sec

# Hit ratio (for cache effectiveness)
redis-cli info stats | grep -E 'keyspace_hits|keyspace_misses'

# Slow queries
redis-cli slowlog len

# Persistence status
redis-cli info persistence
```

## Summary Checklist

- ✅ Configure `bind 0.0.0.0 ::` in redis.conf
- ✅ Optimize memory settings for AI workload
- ✅ Set up proper logging and persistence
- ✅ Configure system limits and kernel parameters
- ✅ Set up firewall rules
- ✅ Enable and start Redis service
- ✅ Test local and remote connections
- ✅ Set up monitoring and backup scripts
- ✅ Validate performance with benchmarks

**Expected Result**: Redis will accept connections from the network on port 6379 with optimized performance for the sejm-whiz AI workload including caching and job queues.
