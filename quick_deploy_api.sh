#!/bin/bash

# Quick deployment script to test search endpoint
echo "🚀 Quick Deploy API Server with Search Endpoint"

# Copy files and start API server on p7
rsync -av --delete bases/ root@p7:/root/tmp/sejm-whiz/bases/
rsync -av --delete projects/ root@p7:/root/tmp/sejm-whiz/projects/

# Start database and Redis first
ssh root@p7 "cd /root/tmp/sejm-whiz && docker run -d --rm --name postgres-test -p 5433:5432 -e POSTGRES_DB=sejm_whiz -e POSTGRES_USER=sejm_whiz_user -e POSTGRES_PASSWORD=sejm_whiz_password pgvector/pgvector:pg17"

ssh root@p7 "cd /root/tmp/sejm-whiz && docker run -d --rm --name redis-test -p 6379:6379 redis:7.2-alpine"

sleep 5

# Start API server
ssh root@p7 "cd /root/tmp/sejm-whiz && docker run -d --rm --name api-test -p 8001:8000 --link postgres-test --link redis-test -v /root/tmp/sejm-whiz:/app -w /app -e DATABASE_HOST=postgres-test -e DATABASE_PORT=5432 -e DATABASE_NAME=sejm_whiz -e DATABASE_USER=sejm_whiz_user -e DATABASE_PASSWORD=sejm_whiz_password -e DATABASE_SSL_MODE=disable -e REDIS_HOST=redis-test -e REDIS_PORT=6379 python:3.12-slim-bookworm bash -c 'pip install --no-cache-dir fastapi uvicorn pydantic jinja2 pydantic-settings && cd projects/api_server && PYTHONPATH=/app/components:/app/bases python main.py'"

echo "✅ Deployment complete!"
echo "🔍 Test the search endpoint:"
echo "curl 'http://p7:8001/api/v1/search?q=konstytucja&limit=3'"
