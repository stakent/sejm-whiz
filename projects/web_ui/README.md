# Web UI Project

Web interface for monitoring and interacting with the Sejm Whiz data processing pipeline.

## Overview

This project provides a comprehensive web interface with multiple pages for monitoring the data processing pipeline, viewing system status, and accessing API documentation.

## Features

- **🏠 Home Page**: Project overview with feature descriptions
- **📊 Dashboard**: Real-time monitoring with live log streaming
- **📚 API Documentation**: Interactive FastAPI/Swagger docs integration
- **❤️ Health Check**: System health status monitoring
- **Fixed Navigation**: Top navigation menu across all pages

## Architecture

Uses the `web_api` base from the Polylith workspace to provide:
- FastAPI application framework
- CORS middleware configuration  
- Error handling and structured responses
- Health check endpoints

## Development

### Running Locally

```bash
# From workspace root
uv run python projects/web_ui/main.py

# Or with uvicorn for development
uv run uvicorn projects.web_ui.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Build

```bash
# Build the Docker image
docker build -t sejm-whiz-web-ui:latest -f projects/web_ui/Dockerfile .

# Run the container
docker run -p 8000:8000 sejm-whiz-web-ui:latest
```

## API Endpoints

- `GET /` - Root endpoint (redirects to home)
- `GET /home` - Landing page with project overview
- `GET /dashboard` - Real-time monitoring dashboard
- `GET /api/logs/stream` - Server-Sent Events for live log streaming
- `GET /health` - Health check with JSON response
- `GET /docs` - FastAPI auto-generated documentation

## Technology Stack

- **Backend**: FastAPI with embedded HTML templates
- **Frontend**: Vanilla JavaScript with Server-Sent Events (SSE)
- **Styling**: Modern CSS with gradients and responsive design
- **Deployment**: Multi-stage Docker build with uv package manager

## Dependencies

- FastAPI >= 0.116.1
- Uvicorn >= 0.32.1
- Jinja2 >= 3.1.6
- Pydantic >= 2.10.5
- httpx >= 0.28.1

## Deployment

This project is designed for deployment to k3s using the standard deployment manifests in `deployments/k3s/manifests/`. See the k3s deployment documentation for complete setup instructions.