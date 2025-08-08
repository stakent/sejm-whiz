#!/bin/bash
set -e

# P7 Database setup script for sejm-whiz system
# Creates fresh database specifically on p7 server with proper schema and extensions
# Exits if database already exists on p7

# Configuration (p7-specific)
DB_HOST=p7
DB_NAME=sejm_whiz
DB_USER=sejm_whiz_user
DB_PASSWORD=sejm_whiz_password
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

# Check if database already exists
check_database_exists() {
    log_info "Checking if database '$DB_NAME' exists on $DB_HOST..."

    if ssh $SSH_HOST "sudo -u postgres psql -lqt" | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log_error "Database '$DB_NAME' already exists on $DB_HOST!"
        log_error "To recreate the database, use the cleanup script:"
        log_error "  ./scripts/cleanup-p7-database.sh"
        log_error ""
        log_error "Or drop manually:"
        log_error "  ssh $SSH_HOST \"sudo -u postgres psql -c \\\"DROP DATABASE IF EXISTS $DB_NAME;\\\"\""
        log_error "  ssh $SSH_HOST \"sudo -u postgres psql -c \\\"DROP USER IF EXISTS $DB_USER;\\\"\""
        exit 1
    fi

    log_success "Database '$DB_NAME' does not exist. Proceeding with setup..."
}

# Test PostgreSQL connection
test_postgresql() {
    log_info "Testing PostgreSQL connection to $DB_HOST..."

    if ! ssh $SSH_HOST "sudo -u postgres psql -c 'SELECT version();'" > /dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL on $DB_HOST"
        log_error "Ensure PostgreSQL is running and accessible via SSH"
        exit 1
    fi

    local pg_version=$(ssh $SSH_HOST "sudo -u postgres psql -c 'SELECT version();'" | grep PostgreSQL | head -1)
    log_success "PostgreSQL connection successful"
    log_info "Version: $pg_version"
}

# Create database
create_database() {
    log_info "Creating database '$DB_NAME'..."

    if ssh $SSH_HOST "sudo -u postgres psql -c \"CREATE DATABASE $DB_NAME;\"" > /dev/null 2>&1; then
        log_success "Database '$DB_NAME' created successfully"
    else
        log_error "Failed to create database '$DB_NAME'"
        exit 1
    fi
}

# Create database user
create_user() {
    log_info "Creating database user '$DB_USER'..."

    # Create user (ignore if already exists)
    ssh $SSH_HOST "sudo -u postgres psql -c \"CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';\"" > /dev/null 2>&1 || true

    log_success "Database user '$DB_USER' ready"
}

# Grant permissions
grant_permissions() {
    log_info "Granting permissions..."

    # Grant database ownership
    ssh $SSH_HOST "sudo -u postgres psql -c \"ALTER DATABASE $DB_NAME OWNER TO $DB_USER;\""

    # Grant schema permissions
    ssh $SSH_HOST "sudo -u postgres psql -d $DB_NAME -c \"GRANT ALL ON SCHEMA public TO $DB_USER;\""

    log_success "Permissions granted successfully"
}

# Install pgvector extension
install_pgvector() {
    log_info "Installing pgvector extension..."

    if ssh $SSH_HOST "sudo -u postgres psql -d $DB_NAME -c \"CREATE EXTENSION IF NOT EXISTS vector;\"" > /dev/null 2>&1; then
        log_success "pgvector extension installed successfully"
    else
        log_error "Failed to install pgvector extension"
        exit 1
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    cd "$(dirname "$0")/.."

    if cd components/sejm_whiz/database && DEPLOYMENT_ENV=p7 PYTHONPATH="../.." uv run alembic upgrade head; then
        log_success "Database migrations completed successfully"
        cd - > /dev/null
    else
        log_error "Database migrations failed"
        cd - > /dev/null
        exit 1
    fi
}

# Verify database setup
verify_setup() {
    log_info "Verifying database setup..."

    # Test connection with new user
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null 2>&1; then
        log_success "Database connection with new user successful"
    else
        log_error "Cannot connect to database with new user"
        exit 1
    fi

    # Check tables exist
    local table_count=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';" | tr -d ' ')

    if [ "$table_count" -gt 0 ]; then
        log_success "Database schema created successfully ($table_count tables)"
    else
        log_error "No tables found in database schema"
        exit 1
    fi

    # Test pgvector
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT vector_dims(vector '[1,2,3]');" > /dev/null 2>&1; then
        log_success "pgvector extension working correctly"
    else
        log_error "pgvector extension not working"
        exit 1
    fi

    # Show empty tables (as expected for new database)
    log_info "Verifying database is empty (as expected for new setup):"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
        SELECT
            schemaname,
            relname as tablename,
            n_tup_ins as row_count
        FROM pg_stat_user_tables
        ORDER BY relname;
    "
}

# Main execution
main() {
    echo "=========================================="
    echo "🗄️  SEJM-WHIZ P7 DATABASE SETUP SCRIPT"
    echo "=========================================="
    echo
    log_info "Target: $DB_HOST"
    log_info "Database: $DB_NAME"
    log_info "User: $DB_USER"
    echo

    check_database_exists
    test_postgresql
    create_database
    create_user
    grant_permissions
    install_pgvector
    run_migrations
    verify_setup

    echo
    echo "=========================================="
    log_success "🎉 P7 DATABASE SETUP COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo
    log_info "Next steps:"
    echo "  1. Ingest data: DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --limit 10"
    echo "  2. Test search:  DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py search status"
    echo
    log_info "Connection details:"
    echo "  Host: $DB_HOST"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo "  Password: $DB_PASSWORD"
    echo
}

# Run main function
main "$@"
