#!/usr/bin/env python3
"""
Standalone dashboard application for Sejm Whiz monitoring
This runs the web dashboard without requiring the full component system
"""

import subprocess
import logging
import asyncio
from datetime import datetime, UTC
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sejm_whiz import __version__

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = __version__


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


# Create FastAPI application
app = FastAPI(
    title="Sejm Whiz Dashboard",
    description="Real-time monitoring dashboard for Sejm Whiz data processing pipeline",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates_dir = Path("/app/bases/sejm_whiz/web_api/templates")
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    # Fallback: create a simple template
    templates_dir = Path("/tmp")
    templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", timestamp=datetime.now(UTC).isoformat())


@app.get("/")
async def root():
    return {
        "message": "Sejm Whiz Dashboard",
        "version": __version__,
        "status": "running",
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the monitoring dashboard."""
    try:
        if Path("/app/bases/sejm_whiz/web_api/templates/dashboard.html").exists():
            return templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": "Sejm Whiz Data Processor Monitor"},
            )
        else:
            # Return a simple fallback dashboard
            return HTMLResponse(
                content="""
<!DOCTYPE html>
<html>
<head>
    <title>Sejm Whiz Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status { padding: 20px; margin: 20px 0; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Sejm Whiz Dashboard</h1>
        <div class="status success">
            <h3>✅ System Status</h3>
            <p>Dashboard is running successfully on p7 server!</p>
        </div>
        <div class="status info">
            <h3>📊 Deployment Details</h3>
            <ul>
                <li><strong>Environment:</strong> Docker Compose Development</li>
                <li><strong>Server:</strong> p7</li>
                <li><strong>API Endpoint:</strong> /api/</li>
                <li><strong>Database:</strong> PostgreSQL (port 5433)</li>
                <li><strong>Cache:</strong> Redis (port 6379)</li>
            </ul>
        </div>
        <div class="status info">
            <h3>🔗 Available Endpoints</h3>
            <ul>
                <li><a href="/health">/health</a> - System health check</li>
                <li><a href="/api/services/status">/api/services/status</a> - Services status</li>
                <li><a href="/api/processor/status">/api/processor/status</a> - Processor status</li>
            </ul>
        </div>
    </div>
</body>
</html>
            """
            )
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return HTMLResponse(
            content=f"<h1>Dashboard Error</h1><p>{e}</p>", status_code=500
        )


@app.get("/api/services/status")
async def services_status():
    """Get the status of all Docker Compose services."""
    try:
        # Try to get Docker container information
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=sejm-whiz",
                "--format",
                "{{.Names}}\t{{.Status}}\t{{.Image}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        services = []
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3:
                    name = parts[0]
                    status = parts[1]
                    image = parts[2]

                    # Determine service type
                    service_type = "unknown"
                    if "postgres" in name.lower():
                        service_type = "database"
                    elif "redis" in name.lower():
                        service_type = "cache"
                    elif "api" in name.lower():
                        service_type = "api"

                    # Parse status
                    running = "Up" in status

                    services.append(
                        {
                            "name": name,
                            "status": "running" if running else "stopped",
                            "service_type": service_type,
                            "image": image,
                            "status_detail": status,
                        }
                    )

        return {
            "services": services,
            "total_services": len(services),
            "running_services": len([s for s in services if s["status"] == "running"]),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting services status: {e}")
        return {
            "services": [],
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@app.get("/api/processor/status")
async def processor_status():
    """Get the current status of the data processor."""
    return {
        "status": "not_deployed",
        "message": "Data processor not yet deployed in this minimal setup",
        "environment": "docker-compose-minimal",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/logs/stream")
async def stream_logs():
    """Stream real Docker container logs."""

    async def log_generator() -> AsyncGenerator[str, None]:
        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Starting real-time log streaming\n\n"

        # Get list of containers to monitor
        containers_to_monitor = [
            "sejm-whiz-api-server-dev",
            "sejm-whiz-processor-dev",
            "sejm-whiz-postgres-dev",
            "sejm-whiz-redis-dev",
        ]

        # Try to stream from each container
        for container in containers_to_monitor:
            try:
                # Check if container exists and is running
                check_result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "--filter",
                        f"name={container}",
                        "--format",
                        "{{.Names}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if check_result.returncode == 0 and check_result.stdout.strip():
                    yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Found container: {container}\n\n"

                    # Stream recent logs from this container
                    try:
                        logs_result = subprocess.run(
                            ["docker", "logs", "--tail=10", container],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )

                        if logs_result.returncode == 0:
                            recent_logs = logs_result.stdout.strip()
                            if recent_logs:
                                for log_line in recent_logs.split("\n"):
                                    if log_line.strip():
                                        yield f"data: {log_line.strip()}\n\n"
                            else:
                                yield f"data: {datetime.now(UTC).isoformat()} - {container} - INFO - No recent logs available\n\n"
                        else:
                            yield f"data: {datetime.now(UTC).isoformat()} - dashboard - WARN - Could not read logs from {container}\n\n"
                    except Exception as e:
                        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - ERROR - Log streaming error for {container}: {str(e)}\n\n"

            except Exception as e:
                yield f"data: {datetime.now(UTC).isoformat()} - dashboard - ERROR - Container check failed for {container}: {str(e)}\n\n"

        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Initial log scan complete\n\n"

        # Now start live streaming from containers
        active_containers = []
        for container in containers_to_monitor:
            try:
                check_result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "--filter",
                        f"name={container}",
                        "--format",
                        "{{.Names}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if check_result.returncode == 0 and check_result.stdout.strip():
                    active_containers.append(container)
            except Exception:
                continue

        if active_containers:
            yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Starting live streaming from {len(active_containers)} containers\n\n"

            # For now, we'll poll for new logs every 10 seconds
            # In a full implementation, we'd use docker logs -f with async subprocess
            last_log_time = datetime.now(UTC)

            while True:
                await asyncio.sleep(10)

                for container in active_containers:
                    try:
                        # Get logs since last check
                        since_time = last_log_time.strftime("%Y-%m-%dT%H:%M:%S")
                        logs_result = subprocess.run(
                            ["docker", "logs", "--since", since_time, container],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )

                        if logs_result.returncode == 0 and logs_result.stdout.strip():
                            new_logs = logs_result.stdout.strip()
                            for log_line in new_logs.split("\n"):
                                if log_line.strip():
                                    yield f"data: {log_line.strip()}\n\n"
                    except Exception as e:
                        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - ERROR - Live streaming error for {container}: {str(e)}\n\n"

                last_log_time = datetime.now(UTC)
                yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Log scan cycle complete\n\n"
        else:
            # Fallback to demo logs if no containers found
            while True:
                await asyncio.sleep(15)
                yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - No active containers found, showing status updates\n\n"

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
