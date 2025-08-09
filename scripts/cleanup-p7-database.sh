#!/bin/bash
set -e

# P7 Database cleanup script for sejm-whiz system
# Drops database and user specifically from p7 server
# Use with caution - this destroys all p7 data!

# Configuration (p7-specific)
DB_HOST=p7
DB_NAME=sejm_whiz
DB_USER=sejm_whiz_user
SSH_HOST=root@p7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Confirm destructive action
confirm_cleanup() {
    echo "⚠️  WARNING: This will permanently delete:"
    echo "   - Database: $DB_NAME on $DB_HOST"
    echo "   - User: $DB_USER"
    echo "   - ALL DATA will be lost!"
    echo
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation

    if [ "$confirmation" != "yes" ]; then
        log_info "Operation cancelled"
        exit 0
    fi
}

# Drop database
drop_database() {
    log_info "Dropping database '$DB_NAME'..."

    if ssh $SSH_HOST "sudo -u postgres psql -c \"DROP DATABASE IF EXISTS $DB_NAME;\"" > /dev/null 2>&1; then
        log_success "Database '$DB_NAME' dropped successfully"
    else
        log_warning "Database '$DB_NAME' may not exist or failed to drop"
    fi
}

# Drop user
drop_user() {
    log_info "Dropping user '$DB_USER'..."

    if ssh $SSH_HOST "sudo -u postgres psql -c \"DROP USER IF EXISTS $DB_USER;\"" > /dev/null 2>&1; then
        log_success "User '$DB_USER' dropped successfully"
    else
        log_warning "User '$DB_USER' may not exist or failed to drop"
    fi
}

# Verify cleanup
verify_cleanup() {
    log_info "Verifying cleanup..."

    # Check database doesn't exist
    if ! ssh $SSH_HOST "sudo -u postgres psql -lqt" | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log_success "Database '$DB_NAME' successfully removed"
    else
        log_error "Database '$DB_NAME' still exists!"
        exit 1
    fi

    # Check user doesn't exist
    if ! ssh $SSH_HOST "sudo -u postgres psql -t -c \"SELECT 1 FROM pg_roles WHERE rolname='$DB_USER';\"" | grep -q 1; then
        log_success "User '$DB_USER' successfully removed"
    else
        log_error "User '$DB_USER' still exists!"
        exit 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "🗑️  SEJM-WHIZ P7 DATABASE CLEANUP SCRIPT"
    echo "=========================================="
    echo
    log_info "Target: $DB_HOST"
    log_info "Database: $DB_NAME"
    log_info "User: $DB_USER"
    echo

    confirm_cleanup
    drop_database
    drop_user
    verify_cleanup

    echo
    echo "=========================================="
    log_success "🧹 P7 DATABASE CLEANUP COMPLETED!"
    echo "=========================================="
    echo
    log_info "You can now run setup-p7-database.sh to create a fresh p7 database"
    echo
}

# Run main function
main "$@"
