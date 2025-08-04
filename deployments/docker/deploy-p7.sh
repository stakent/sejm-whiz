#!/bin/bash

# Deploy Sejm Whiz Docker Compose Dev Environment to p7 Server
# Usage: ./deploy-p7.sh

set -e

# Configuration
P7_HOST="root@p7"
REMOTE_DIR="/tmp/sejm-whiz"
LOCAL_PROJECT_ROOT="../../"
COMPOSE_FILE="docker-compose.dev.yml"

echo "🚀 Deploying Sejm Whiz Dev Environment to p7..."

# Function to run commands on p7
run_on_p7() {
    ssh $P7_HOST "$@"
}

# Function to copy files to p7
copy_to_p7() {
    rsync -avz --progress --delete "$@" $P7_HOST:$REMOTE_DIR/
}

echo "📁 Preparing remote directory..."
run_on_p7 "mkdir -p $REMOTE_DIR"

echo "📋 Copying project files to p7..."
# Copy essential files only to avoid large transfers
rsync -avz --progress --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='models/' \
    --exclude='.venv' \
    --exclude='*.log' \
    $LOCAL_PROJECT_ROOT $P7_HOST:$REMOTE_DIR/

echo "🐳 Setting up Docker environment on p7..."
run_on_p7 "cd $REMOTE_DIR && docker --version"

echo "🛑 Stopping existing containers..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE down --remove-orphans || true"

echo "🏗️ Building Docker images..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE build --no-cache"

echo "📦 Pulling base images..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE pull"

echo "🚀 Starting services..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE up -d"

echo "⏳ Waiting for services to start..."
sleep 10

echo "📊 Checking service status..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE ps"

echo "🏥 Health check..."
echo "Testing API server health..."
run_on_p7 "curl -f http://localhost:8001/health || echo 'API not ready yet'"

echo "✅ Deployment completed!"
echo ""
echo "🌐 Dashboard URL: http://p7:8001/dashboard"
echo "📡 API Server: http://p7:8001"
echo "🗄️ PostgreSQL: p7:5433"
echo "📦 Redis: p7:6379"
echo ""
echo "📋 To check logs:"
echo "  ssh root@p7 'cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE logs -f'"
echo ""
echo "🛑 To stop:"
echo "  ssh root@p7 'cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE down'"
