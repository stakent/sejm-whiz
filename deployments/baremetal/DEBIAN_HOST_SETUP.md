# Debian Host Setup for Sejm-Whiz Baremetal Deployment

This document provides complete steps to prepare a Debian host for running Sejm-Whiz on baremetal.

## Prerequisites

- Debian 12 (Bookworm) or compatible system
- Root access
- Internet connection
- At least 4GB RAM (8GB+ recommended)
- At least 20GB free disk space

## Step 1: System Update

```bash
apt-get update
apt-get upgrade -y
```

## Step 2: Install Basic Dependencies

```bash
# Install essential packages
apt-get install -y \
    curl \
    wget \
    git \
    rsync \
    htop \
    vim \
    tmux \
    build-essential \
    python3.12 \
    python3.12-venv \
    python3.12-dev
```

## Step 3: Install uv (Python Package Manager)

```bash
# Install uv system-wide
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
ln -sf ~/.cargo/bin/uv /usr/local/bin/uv
```

## Step 4: Install PostgreSQL 17 with pgvector

### Add PostgreSQL APT Repository

```bash
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
echo 'deb http://apt.postgresql.org/pub/repos/apt/ bookworm-pgdg main' > /etc/apt/sources.list.d/pgdg.list
apt-get update
```

### Install PostgreSQL 17

```bash
apt-get install -y postgresql-17 postgresql-client-17 postgresql-contrib-17
```

### Install pgvector 0.8.0 Extension

```bash
# Install development dependencies
apt-get install -y postgresql-server-dev-17 build-essential git

# Clone and build pgvector 0.8.0
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git /tmp/pgvector
cd /tmp/pgvector
make clean
make OPTFLAGS=''
make install
```

### Start and Configure PostgreSQL

```bash
# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE sejm_whiz;"
sudo -u postgres psql -c "CREATE USER sejm_whiz_user WITH PASSWORD 'sejm_whiz_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sejm_whiz TO sejm_whiz_user;"

# Grant schema permissions
sudo -u postgres psql -d sejm_whiz -c 'GRANT USAGE ON SCHEMA public TO sejm_whiz_user;'
sudo -u postgres psql -d sejm_whiz -c 'GRANT CREATE ON SCHEMA public TO sejm_whiz_user;'
sudo -u postgres psql -d sejm_whiz -c 'ALTER SCHEMA public OWNER TO sejm_whiz_user;'
sudo -u postgres psql -c 'ALTER USER sejm_whiz_user CREATEDB;'

# Enable pgvector extension
sudo -u postgres psql -d sejm_whiz -c 'CREATE EXTENSION vector;'
```

### Verify PostgreSQL Installation

```bash
# Test connection and extension
sudo -u postgres psql -d sejm_whiz -c "SELECT version();"
sudo -u postgres psql -d sejm_whiz -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

## Step 5: Install Redis

```bash
# Install Redis
apt-get install -y redis-server

# Start and enable Redis
systemctl start redis-server
systemctl enable redis-server

# Verify Redis is running
systemctl status redis-server
redis-cli ping  # Should return "PONG"
```

## Step 6: Create System User for Sejm-Whiz

```bash
# Create system user and group
groupadd --system sejm-whiz
useradd --system --gid sejm-whiz --home-dir /opt/sejm-whiz --shell /bin/bash sejm-whiz

# Create directories
mkdir -p /opt/sejm-whiz /etc/sejm-whiz /var/log/sejm-whiz /var/lib/sejm-whiz
chown -R sejm-whiz:sejm-whiz /opt/sejm-whiz /var/log/sejm-whiz /var/lib/sejm-whiz
chown root:sejm-whiz /etc/sejm-whiz
chmod 750 /etc/sejm-whiz
```

## Step 7: Install uv for sejm-whiz User

```bash
# Install uv for the sejm-whiz user
sudo -u sejm-whiz bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Create symlink for easier access
ln -sf /opt/sejm-whiz/.local/bin/uv /usr/local/bin/uv-sejm
```

## Step 8: Configure Firewall (Optional)

```bash
# If using ufw
if command -v ufw &> /dev/null; then
    ufw allow 8001/tcp comment "Sejm-Whiz API"
    ufw allow 8002/tcp comment "Sejm-Whiz Web UI"
fi
```

## Step 9: Setup Log Rotation

```bash
cat > /etc/logrotate.d/sejm-whiz << 'EOF'
/var/log/sejm-whiz/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su sejm-whiz sejm-whiz
}
EOF
```

## Step 10: Deploy Sejm-Whiz Application

### Copy Application Files

```bash
# From development machine, copy files to p7
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  /path/to/sejm-whiz-dev/ root@p7:/opt/sejm-whiz/

# Set ownership
chown -R sejm-whiz:sejm-whiz /opt/sejm-whiz/
```

### Install Python Dependencies

```bash
# Install dependencies as sejm-whiz user
cd /opt/sejm-whiz
echo '3.12' > .python-version
chown sejm-whiz:sejm-whiz .python-version
sudo -u sejm-whiz /opt/sejm-whiz/.local/bin/uv python install 3.12
sudo -u sejm-whiz /opt/sejm-whiz/.local/bin/uv sync --dev
```

### Configure Environment

```bash
# Copy configuration template
cp deployments/baremetal/config/environment.env /etc/sejm-whiz/
chown root:sejm-whiz /etc/sejm-whiz/environment.env
chmod 640 /etc/sejm-whiz/environment.env

# Edit configuration as needed
vim /etc/sejm-whiz/environment.env
```

### Run Database Migrations

```bash
cd /opt/sejm-whiz
sudo -u sejm-whiz /opt/sejm-whiz/.local/bin/uv run alembic upgrade head
```

## Step 11: Test Installation

### Test Database Connectivity

```bash
cd /opt/sejm-whiz
sudo -u sejm-whiz /opt/sejm-whiz/.local/bin/uv run python -c "
from sejm_whiz.database import DocumentOperations
from sejm_whiz.vector_db import VectorDBOperations
print('Testing database connections...')
db = DocumentOperations()
print('✅ Document operations initialized')
vector_db = VectorDBOperations()
print('✅ Vector DB operations initialized')
print('✅ All database connections successful')
"
```

### Test Redis Connectivity

```bash
redis-cli ping  # Should return "PONG"
```

## Verification Checklist

- [ ] PostgreSQL 17 is running with pgvector 0.8.0 extension
- [ ] Redis is running and accessible
- [ ] sejm-whiz user created with proper permissions
- [ ] Python 3.12 virtual environment set up with all dependencies
- [ ] Database migrations completed successfully
- [ ] Environment configuration file in place
- [ ] Log rotation configured
- [ ] Firewall rules configured (if applicable)

## Service Ports

- **PostgreSQL**: 5432 (internal)
- **Redis**: 6379 (internal) 
- **API Server**: 8001 (external)
- **Web UI**: 8002 (external)

## System Requirements Met

- ✅ **PostgreSQL 17** with pgvector 0.8.0
- ✅ **Redis 7.0+** for caching and queues
- ✅ **Python 3.12** with uv package manager
- ✅ **System user isolation** for security
- ✅ **Proper permissions** and directory structure
- ✅ **Log management** and rotation
- ✅ **Database schema** initialized

## Next Steps

After completing these steps, you can:

1. **Deploy services**: Use the deployment scripts in `deployments/baremetal/scripts/`
2. **Test pipeline**: Run the data processing pipeline
3. **Set up monitoring**: Configure systemd services for automatic startup
4. **Production hardening**: Additional security configurations as needed

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure sejm-whiz user owns application files
2. **Database connection**: Check PostgreSQL is running and user has correct permissions
3. **Redis connection**: Verify Redis service is active
4. **Python dependencies**: Ensure uv is properly installed for sejm-whiz user
5. **Port conflicts**: Check no other services are using ports 8001/8002

### Log Locations

- **Application logs**: `/var/log/sejm-whiz/`
- **PostgreSQL logs**: `/var/log/postgresql/`
- **Redis logs**: `/var/log/redis/`
- **System logs**: `journalctl -u <service-name>`

## Security Notes

- The sejm-whiz user runs with minimal privileges
- Database credentials should be changed from defaults in production
- Consider enabling SSL/TLS for database connections in production
- Firewall should be configured to restrict access to necessary ports only