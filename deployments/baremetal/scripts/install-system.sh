#!/bin/bash
# System installation script for Sejm-Whiz baremetal deployment on p7

set -euo pipefail

# Configuration
INSTALL_DIR="/opt/sejm-whiz"
CONFIG_DIR="/etc/sejm-whiz"
LOG_DIR="/var/log/sejm-whiz"
DATA_DIR="/var/lib/sejm-whiz"
SERVICE_USER="sejm-whiz"
SERVICE_GROUP="sejm-whiz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
   exit 1
fi

log "Starting Sejm-Whiz system installation on p7..."

# Create system user and group
log "Creating system user and group..."
if ! getent group $SERVICE_GROUP > /dev/null 2>&1; then
    groupadd --system $SERVICE_GROUP
    log "Created group: $SERVICE_GROUP"
fi

if ! getent passwd $SERVICE_USER > /dev/null 2>&1; then
    useradd --system --gid $SERVICE_GROUP --home-dir $INSTALL_DIR --shell /bin/bash $SERVICE_USER
    log "Created user: $SERVICE_USER"
fi

# Create directories
log "Creating directories..."
mkdir -p $INSTALL_DIR $CONFIG_DIR $LOG_DIR $DATA_DIR
chown -R $SERVICE_USER:$SERVICE_GROUP $INSTALL_DIR $LOG_DIR $DATA_DIR
chown root:$SERVICE_GROUP $CONFIG_DIR
chmod 750 $CONFIG_DIR

# Install system dependencies
log "Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    curl \
    wget \
    git \
    rsync \
    postgresql-client \
    redis-tools \
    htop \
    vim \
    tmux

# Install uv (Python package manager)
log "Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    ln -sf $HOME/.cargo/bin/uv /usr/local/bin/uv
    log "Installed uv"
else
    log "uv already installed"
fi

# Copy source code
log "Copying source code to $INSTALL_DIR..."
if [[ -d "/root/tmp/sejm-whiz-baremetal" ]]; then
    # Copy from deployment staging directory
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        /root/tmp/sejm-whiz-baremetal/ $INSTALL_DIR/
elif [[ -d "/home/d/project/sejm-whiz/sejm-whiz-dev" ]]; then
    # Fallback for local development
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        /home/d/project/sejm-whiz/sejm-whiz-dev/ $INSTALL_DIR/
else
    error "Source directory not found. Expected /root/tmp/sejm-whiz-baremetal or /home/d/project/sejm-whiz/sejm-whiz-dev"
    exit 1
fi

# Set ownership
chown -R $SERVICE_USER:$SERVICE_GROUP $INSTALL_DIR

# Install Python dependencies
log "Installing Python dependencies..."
cd $INSTALL_DIR
sudo -u $SERVICE_USER uv sync --dev

# Copy configuration
log "Setting up configuration..."
cp deployments/baremetal/config/environment.env $CONFIG_DIR/
chown root:$SERVICE_GROUP $CONFIG_DIR/environment.env
chmod 640 $CONFIG_DIR/environment.env

# Create log rotation configuration
log "Setting up log rotation..."
cat > /etc/logrotate.d/sejm-whiz << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $SERVICE_USER $SERVICE_GROUP
}
EOF

# Setup firewall rules (if ufw is available)
if command -v ufw &> /dev/null; then
    log "Configuring firewall..."
    ufw allow 8001/tcp comment "Sejm-Whiz API"
    ufw allow 8002/tcp comment "Sejm-Whiz Web UI"
fi

log "✅ System installation completed!"
info "Next steps:"
info "1. Configure PostgreSQL and Redis connections in $CONFIG_DIR/environment.env"
info "2. Run database migrations: cd $INSTALL_DIR && sudo -u $SERVICE_USER uv run alembic upgrade head"
info "3. Deploy services:"
info "   - ./deployments/baremetal/scripts/deploy-api-server.sh"
info "   - ./deployments/baremetal/scripts/deploy-data-processor.sh"  
info "   - ./deployments/baremetal/scripts/deploy-web-ui.sh"
info "4. Check service status: systemctl status sejm-whiz-*"