#!/bin/bash
# Deploy all Sejm-Whiz services to p7 baremetal

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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
   exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log "Starting complete Sejm-Whiz deployment on p7..."

# Deploy in order: API Server -> Data Processor -> Web UI
SERVICES=("api-server" "data-processor" "web-ui")
FAILED_SERVICES=()

for service in "${SERVICES[@]}"; do
    log "Deploying $service..."
    if ! "$SCRIPT_DIR/deploy-$service.sh"; then
        error "Failed to deploy $service"
        FAILED_SERVICES+=("$service")
    else
        log "✅ $service deployed successfully"
    fi
    echo
done

# Summary
log "Deployment Summary:"
if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    log "✅ All services deployed successfully!"
    info "Services status:"
    systemctl status sejm-whiz-api --no-pager -l
    echo
    systemctl status sejm-whiz-processor.timer --no-pager -l  
    echo
    systemctl status sejm-whiz-web-ui --no-pager -l
    echo
    info "Access URLs:"
    info "- API Server: http://p7:8001"
    info "- API Docs: http://p7:8001/docs"
    info "- Web UI: http://p7:8002"
    info "- Dashboard: http://p7:8002/dashboard"
else
    error "❌ Failed to deploy: ${FAILED_SERVICES[*]}"
    exit 1
fi