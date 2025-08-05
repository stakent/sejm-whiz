#!/bin/bash
# Deploy API Server to p7 baremetal

set -euo pipefail

# Configuration
INSTALL_DIR="/opt/sejm-whiz"
CONFIG_DIR="/etc/sejm-whiz"
LOG_DIR="/var/log/sejm-whiz"
SERVICE_USER="sejm-whiz"
SERVICE_NAME="sejm-whiz-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
   exit 1
fi

log "Starting API Server deployment on p7..."

# Stop existing service if running
if systemctl is-active --quiet $SERVICE_NAME; then
    log "Stopping existing $SERVICE_NAME service..."
    systemctl stop $SERVICE_NAME
fi

# Copy API server files
log "Copying API server files to $INSTALL_DIR..."
rsync -av --delete projects/api_server/ $INSTALL_DIR/projects/api_server/

# Install/update dependencies
log "Installing Python dependencies..."
cd $INSTALL_DIR
uv sync --dev

# Create/update systemd service
log "Creating systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Sejm-Whiz API Server
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR/components:$INSTALL_DIR/bases
EnvironmentFile=$CONFIG_DIR/environment.env
ExecStart=/usr/local/bin/uv run python projects/api_server/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Resource limits
LimitNOFILE=65536
MemoryMax=2G

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# Start service
log "Starting $SERVICE_NAME service..."
systemctl start $SERVICE_NAME

# Check service status
sleep 3
if systemctl is-active --quiet $SERVICE_NAME; then
    log "✅ API Server deployed successfully!"
    log "Service status:"
    systemctl status $SERVICE_NAME --no-pager
    log "API Server should be available at: http://p7:8001"
    log "API Documentation: http://p7:8001/docs"
else
    error "❌ API Server failed to start!"
    systemctl status $SERVICE_NAME --no-pager
    exit 1
fi