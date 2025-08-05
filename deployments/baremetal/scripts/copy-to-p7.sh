#!/bin/bash
# Copy Sejm-Whiz project to p7 server for baremetal deployment

set -euo pipefail

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

# Configuration
LOCAL_DIR="/home/d/project/sejm-whiz/sejm-whiz-dev"
REMOTE_HOST="root@p7"
REMOTE_DIR="/root/tmp/sejm-whiz-baremetal"

# Check if local directory exists
if [[ ! -d "$LOCAL_DIR" ]]; then
    error "Local directory not found: $LOCAL_DIR"
    exit 1
fi

log "Copying Sejm-Whiz project to p7..."

# Test SSH connection
if ! ssh -q -o ConnectTimeout=5 $REMOTE_HOST exit; then
    error "Cannot connect to $REMOTE_HOST"
    exit 1
fi

# Create remote directory
log "Creating remote directory: $REMOTE_DIR"
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Copy files with rsync
log "Syncing files to $REMOTE_HOST:$REMOTE_DIR..."
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='node_modules' \
    --exclude='.DS_Store' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

log "✅ Files copied successfully!"
info "Next steps:"
info "1. SSH to p7: ssh $REMOTE_HOST"
info "2. Go to directory: cd $REMOTE_DIR"
info "3. Run installation: ./deployments/baremetal/scripts/install-system.sh"
info "4. Configure environment: vim /etc/sejm-whiz/environment.env"
info "5. Deploy services: ./deployments/baremetal/scripts/deploy-all.sh"