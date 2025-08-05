#!/bin/bash
# Manage Sejm-Whiz services on p7 baremetal

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

# Service names
SERVICES=("sejm-whiz-api" "sejm-whiz-processor" "sejm-whiz-web-ui")
TIMERS=("sejm-whiz-processor.timer")

show_usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
    echo ""
    echo "Commands:"
    echo "  start    - Start all services"
    echo "  stop     - Stop all services"
    echo "  restart  - Restart all services"
    echo "  status   - Show status of all services"
    echo "  logs     - Show recent logs for all services"
    echo "  enable   - Enable all services to start on boot"
    echo "  disable  - Disable all services from starting on boot"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 restart"
    echo "  $0 logs"
}

service_action() {
    local action=$1
    local success_count=0
    local total_count=0
    
    case $action in
        "start"|"stop"|"restart")
            for service in "${SERVICES[@]}"; do
                total_count=$((total_count + 1))
                log "${action^}ing $service..."
                if systemctl $action $service; then
                    success_count=$((success_count + 1))
                else
                    error "Failed to $action $service"
                fi
            done
            
            # Handle timers separately for start/stop
            if [[ $action == "start" || $action == "restart" ]]; then
                for timer in "${TIMERS[@]}"; do
                    total_count=$((total_count + 1))
                    log "Starting $timer..."
                    if systemctl start $timer; then
                        success_count=$((success_count + 1))
                    else
                        error "Failed to start $timer"
                    fi
                done
            elif [[ $action == "stop" ]]; then
                for timer in "${TIMERS[@]}"; do
                    total_count=$((total_count + 1))
                    log "Stopping $timer..."
                    if systemctl stop $timer; then
                        success_count=$((success_count + 1))
                    else
                        error "Failed to stop $timer"
                    fi
                done
            fi
            ;;
            
        "enable"|"disable")
            for service in "${SERVICES[@]}"; do
                total_count=$((total_count + 1))
                log "${action^}ing $service..."
                if systemctl $action $service; then
                    success_count=$((success_count + 1))
                else
                    error "Failed to $action $service"
                fi
            done
            
            for timer in "${TIMERS[@]}"; do
                total_count=$((total_count + 1))
                log "${action^}ing $timer..."
                if systemctl $action $timer; then
                    success_count=$((success_count + 1))
                else
                    error "Failed to $action $timer"
                fi
            done
            ;;
    esac
    
    log "Action completed: $success_count/$total_count succeeded"
}

show_status() {
    log "Sejm-Whiz Services Status:"
    echo
    
    for service in "${SERVICES[@]}"; do
        info "=== $service ==="
        systemctl status $service --no-pager -l || true
        echo
    done
    
    for timer in "${TIMERS[@]}"; do
        info "=== $timer ==="
        systemctl status $timer --no-pager -l || true
        echo
    done
    
    # Show listening ports
    info "=== Listening Ports ==="
    ss -tlnp | grep -E ':(800[12])\s' || echo "No Sejm-Whiz ports found"
    echo
}

show_logs() {
    log "Recent logs for all services:"
    echo
    
    for service in "${SERVICES[@]}"; do
        info "=== $service logs ==="
        journalctl -u $service --no-pager -n 20 || true
        echo
    done
}

# Main script logic
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

# Check if running as root for most operations
if [[ $EUID -ne 0 && "$1" != "status" && "$1" != "logs" ]]; then
   error "This script must be run as root for $1 operations"
   exit 1
fi

case $1 in
    "start"|"stop"|"restart"|"enable"|"disable")
        service_action $1
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    *)
        error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac