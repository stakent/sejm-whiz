#!/bin/bash

# Deploy Sejm Whiz Simple Docker Compose Dev Environment to p7 Server
# Usage: ./deploy-p7-simple.sh

set -e

# Configuration
P7_HOST="root@p7"
REMOTE_DIR="/tmp/sejm-whiz"
LOCAL_PROJECT_ROOT="../../"
COMPOSE_FILE="docker-compose.dev-p7.yml"

echo "🚀 Deploying Sejm Whiz Simple Dev Environment to p7..."

# Function to run commands on p7
run_on_p7() {
    ssh $P7_HOST "$@"
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
    --exclude='.mypy_cache' \
    $LOCAL_PROJECT_ROOT $P7_HOST:$REMOTE_DIR/

echo "🐳 Setting up Docker environment on p7..."
run_on_p7 "cd $REMOTE_DIR && docker --version && docker compose version"

echo "🛑 Stopping existing containers..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE down --remove-orphans || true"

echo "🧹 Cleaning up old containers and images..."
run_on_p7 "docker system prune -f || true"

echo "📦 Pulling base images..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE pull"

echo "🚀 Starting services..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE up -d"

echo "⏳ Waiting for services to start..."
sleep 15

echo "📊 Checking service status..."
run_on_p7 "cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE ps"

echo "🏥 Health check..."
echo "Testing API server health (may take a moment for first startup)..."
for i in {1..6}; do
    if run_on_p7 "curl -f -s http://localhost:8001/health"; then
        echo "✅ API server is healthy!"
        break
    else
        echo "⏳ Attempt $i/6: API not ready yet, waiting 10s..."
        sleep 10
    fi
done

echo "✅ Deployment completed!"
echo ""
echo "🌐 Dashboard URL: http://p7:8001/dashboard"
echo "📡 API Server: http://p7:8001"
echo "🗄️ PostgreSQL: p7:5433"
echo "📦 Redis: p7:6379"
echo ""
echo "📋 To check logs:"
echo "  ssh root@p7 'cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE logs -f api-server'"
echo ""
echo "🛑 To stop:"
echo "  ssh root@p7 'cd $REMOTE_DIR/deployments/docker && docker compose -f $COMPOSE_FILE down'"
