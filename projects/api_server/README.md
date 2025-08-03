# API Server Project

This is the main web API server for the Sejm Whiz application, built using the Polylith architecture.

## Architecture

- **Base**: `web_api` - Provides FastAPI foundation with CORS, error handling, and basic routes
- **Components**: None currently - components will be added as needed for specific functionality

## Running the Server

From the workspace root directory:

```bash
# Run directly with Python
uv run python projects/api_server/main.py

# Run with uvicorn for development (with reload)
uv run uvicorn projects.api_server.main:app --host 0.0.0.0 --port 8000 --reload

# Run with uvicorn for production
uv run uvicorn projects.api_server.main:app --host 0.0.0.0 --port 8000
```

## Available Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Development

The server uses the `web_api` base which provides:
- FastAPI application setup
- CORS middleware configuration
- Error handling for HTTP exceptions and validation errors
- Basic health check and root endpoints
- Automatic API documentation

Additional API routes and functionality should be added by incorporating relevant components into this project.
