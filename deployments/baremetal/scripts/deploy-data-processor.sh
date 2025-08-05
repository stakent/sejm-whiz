#!/bin/bash
# Deploy Data Processor to p7 baremetal

set -euo pipefail

# Configuration
INSTALL_DIR="/opt/sejm-whiz"
CONFIG_DIR="/etc/sejm-whiz"
LOG_DIR="/var/log/sejm-whiz"
SERVICE_USER="sejm-whiz"
SERVICE_NAME="sejm-whiz-processor"

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

log "Starting Data Processor deployment on p7..."

# Stop existing service if running
if systemctl is-active --quiet $SERVICE_NAME; then
    log "Stopping existing $SERVICE_NAME service..."
    systemctl stop $SERVICE_NAME
fi

# Copy data processor files
log "Copying data processor files to $INSTALL_DIR..."
rsync -av --delete projects/data_processor/ $INSTALL_DIR/projects/data_processor/

# Install/update dependencies
log "Installing Python dependencies..."
cd $INSTALL_DIR
uv sync --dev

# Create/update systemd service
log "Creating systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Sejm-Whiz Data Processor
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=oneshot
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR/components:$INSTALL_DIR/bases
EnvironmentFile=$CONFIG_DIR/environment.env
ExecStart=/usr/local/bin/uv run python projects/data_processor/main.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Resource limits
LimitNOFILE=65536
MemoryMax=4G

[Install]
WantedBy=multi-user.target
EOF

# Create timer for periodic execution
log "Creating systemd timer..."
cat > /etc/systemd/system/$SERVICE_NAME.timer << EOF
[Unit]
Description=Run Sejm-Whiz Data Processor periodically
Requires=$SERVICE_NAME.service

[Timer]
# Run every 6 hours
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# Reload systemd and enable services
systemctl daemon-reload
systemctl enable $SERVICE_NAME.timer

# Start timer
log "Starting $SERVICE_NAME timer..."
systemctl start $SERVICE_NAME.timer

# Test run the service once
log "Running initial data processor execution..."
systemctl start $SERVICE_NAME

# Wait for completion and check status
sleep 5
if systemctl is-failed --quiet $SERVICE_NAME; then
    error "❌ Data Processor initial run failed!"
    journalctl -u $SERVICE_NAME --no-pager -n 50
    exit 1
else
    log "✅ Data Processor deployed successfully!"
    log "Timer status:"
    systemctl status $SERVICE_NAME.timer --no-pager
    log "Last execution logs:"
    journalctl -u $SERVICE_NAME --no-pager -n 20
    log "Data processor will run every 6 hours automatically"
fi