#!/bin/bash
# Deploy Web UI to p7 baremetal

set -euo pipefail

# Configuration
INSTALL_DIR="/opt/sejm-whiz"
CONFIG_DIR="/etc/sejm-whiz"
LOG_DIR="/var/log/sejm-whiz"
SERVICE_USER="sejm-whiz"
SERVICE_NAME="sejm-whiz-web-ui"

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

log "Starting Web UI deployment on p7..."

# Stop existing service if running
if systemctl is-active --quiet $SERVICE_NAME; then
    log "Stopping existing $SERVICE_NAME service..."
    systemctl stop $SERVICE_NAME
fi

# Copy web UI files
log "Copying web UI files to $INSTALL_DIR..."
rsync -av --delete projects/web_ui/ $INSTALL_DIR/projects/web_ui/

# Install/update dependencies
log "Installing Python dependencies..."
cd $INSTALL_DIR
uv sync --dev

# Create/update systemd service
log "Creating systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Sejm-Whiz Web UI
After=network.target
Wants=sejm-whiz-api.service sejm-whiz-processor.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR/components:$INSTALL_DIR/bases
EnvironmentFile=$CONFIG_DIR/environment.env
ExecStart=/usr/local/bin/uv run python projects/web_ui/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Resource limits
LimitNOFILE=65536
MemoryMax=1G

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
    log "✅ Web UI deployed successfully!"
    log "Service status:"
    systemctl status $SERVICE_NAME --no-pager
    log "Web UI should be available at: http://p7:8002"
    log "Dashboard: http://p7:8002/dashboard"
    log "Home: http://p7:8002/home"
else
    error "❌ Web UI failed to start!"
    systemctl status $SERVICE_NAME --no-pager
    exit 1
fi