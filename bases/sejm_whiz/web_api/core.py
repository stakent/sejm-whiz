from datetime import datetime
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sejm Whiz API",
        description="AI-driven legal prediction system using Polish Parliament data",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    configure_cors(app)
    configure_error_handlers(app)
    configure_static_files(app)
    configure_routes(app)

    return app


def configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_static_files(app: FastAPI) -> None:
    """Configure static file serving and templates."""
    # Create directories if they don't exist
    base_dir = Path(__file__).parent
    static_dir = base_dir / "static"
    templates_dir = base_dir / "templates"

    static_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def configure_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                detail=str(exc.detail),
                timestamp=datetime.utcnow().isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                detail=str(exc),
                timestamp=datetime.utcnow().isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                detail="An internal server error occurred",
                timestamp=datetime.utcnow().isoformat(),
            ).model_dump(),
        )


def configure_routes(app: FastAPI) -> None:
    # Set up templates
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(status="healthy", timestamp=datetime.utcnow().isoformat())

    @app.get("/")
    async def root():
        return {"message": "Sejm Whiz API", "version": "0.1.0", "docs": "/docs"}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Serve the monitoring dashboard."""
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "title": "Sejm Whiz Data Processor Monitor"},
        )

    @app.get("/api/logs/stream")
    async def stream_logs() -> StreamingResponse:
        """Stream data processor logs in real-time."""

        async def log_generator() -> AsyncGenerator[str, None]:
            kubectl_available = False
            pod_selector = None

            # Try GPU processor first, then CPU processor
            processor_labels = [
                "app=sejm-whiz-processor-gpu",
                "app=sejm-whiz-processor-cpu",
                "app=data-processor",
            ]

            # Check which processor pods are available
            for label in processor_labels:
                try:
                    check_process = await asyncio.create_subprocess_exec(
                        "kubectl",
                        "get",
                        "pods",
                        "-n",
                        "sejm-whiz",
                        "-l",
                        label,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await check_process.communicate()
                    if (
                        check_process.returncode == 0
                        and b"No resources found" not in stdout
                        and stdout.strip()
                    ):
                        kubectl_available = True
                        pod_selector = label
                        yield f"data: {datetime.utcnow().isoformat()} - dashboard - INFO - Found processor pods with label: {label}\n\n"
                        break
                except (FileNotFoundError, OSError):
                    continue

            if kubectl_available and pod_selector:
                try:
                    # Stream logs from Kubernetes pod
                    process = await asyncio.create_subprocess_exec(
                        "kubectl",
                        "logs",
                        "-f",
                        "-n",
                        "sejm-whiz",
                        "-l",
                        pod_selector,
                        "--tail=50",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    if process.stdout:
                        async for line in process.stdout:
                            yield f"data: {line.decode('utf-8')}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Failed to stream from kubectl: {e}")
                    yield f"data: {datetime.utcnow().isoformat()} - dashboard - ERROR - Failed to stream logs: {e}\n\n"

            # Check for local log file
            log_file = Path("/var/log/sejm-whiz/data-processor.log")
            if log_file.exists():
                try:
                    # Tail the log file
                    process = await asyncio.create_subprocess_exec(
                        "tail",
                        "-f",
                        str(log_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    if process.stdout:
                        async for line in process.stdout:
                            yield f"data: {line.decode('utf-8')}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Failed to tail log file: {e}")

            # Fallback: Generate sample/demo logs for development
            yield f"data: {datetime.utcnow().isoformat()} - dashboard - INFO - No live logs available, showing demo logs\n\n"

            sample_logs = [
                f"{datetime.utcnow().isoformat()} - data_processor - INFO - Starting data processor",
                f"{datetime.utcnow().isoformat()} - data_processor - INFO - Initializing pipeline components",
                f"{datetime.utcnow().isoformat()} - sejm_ingestion - INFO - Fetching Sejm proceedings data",
                f"{datetime.utcnow().isoformat()} - sejm_ingestion - INFO - Retrieved 15 proceedings for session 10",
                f"{datetime.utcnow().isoformat()} - eli_ingestion - INFO - Fetching ELI legal documents",
                f"{datetime.utcnow().isoformat()} - eli_ingestion - INFO - Retrieved 8 legal documents",
                f"{datetime.utcnow().isoformat()} - text_processing - INFO - Processing text data",
                f"{datetime.utcnow().isoformat()} - text_processing - INFO - Cleaned 23 documents",
                f"{datetime.utcnow().isoformat()} - embedding_generation - INFO - Generating embeddings",
                f"{datetime.utcnow().isoformat()} - embedding_generation - INFO - Generated embeddings for 23 documents",
                f"{datetime.utcnow().isoformat()} - database_storage - INFO - Storing data in database",
                f"{datetime.utcnow().isoformat()} - database_storage - INFO - Stored 15 Sejm proceedings",
                f"{datetime.utcnow().isoformat()} - database_storage - INFO - Stored 8 ELI documents",
                f"{datetime.utcnow().isoformat()} - data_processor - INFO - Pipeline completed successfully",
            ]

            for log in sample_logs:
                yield f"data: {log}\n\n"
                await asyncio.sleep(0.5)

            # Continue with simulated real-time logs
            batch_num = 1
            while True:
                logs = [
                    f"{datetime.utcnow().isoformat()} - data_processor - INFO - Starting batch {batch_num}",
                    f"{datetime.utcnow().isoformat()} - sejm_ingestion - INFO - Fetching new proceedings",
                    f"{datetime.utcnow().isoformat()} - text_processing - INFO - Processing batch {batch_num}",
                    f"{datetime.utcnow().isoformat()} - embedding_generation - INFO - Generating embeddings for batch {batch_num}",
                    f"{datetime.utcnow().isoformat()} - database_storage - INFO - Storing batch {batch_num} results",
                    f"{datetime.utcnow().isoformat()} - data_processor - INFO - Batch {batch_num} completed",
                ]
                for log in logs:
                    yield f"data: {log}\n\n"
                    await asyncio.sleep(1)

                batch_num += 1
                await asyncio.sleep(3)

        return StreamingResponse(
            log_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/processor/status")
    async def processor_status():
        """Get the current status of the data processor."""
        try:
            # Try GPU processor first, then CPU processor
            processor_labels = [
                "app=sejm-whiz-processor-gpu",
                "app=sejm-whiz-processor-cpu",
                "app=data-processor",
            ]

            for label in processor_labels:
                result = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "pods",
                        "-n",
                        "sejm-whiz",
                        "-l",
                        label,
                        "-o",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    import json

                    pods_data = json.loads(result.stdout)
                    if pods_data.get("items"):
                        pod = pods_data["items"][0]
                        processor_type = (
                            "GPU"
                            if "gpu" in label
                            else "CPU"
                            if "cpu" in label
                            else "Unknown"
                        )
                        return {
                            "status": pod["status"]["phase"],
                            "processor_type": processor_type,
                            "pod_name": pod["metadata"]["name"],
                            "started_at": pod["status"].get("startTime"),
                            "container_statuses": pod["status"].get(
                                "containerStatuses", []
                            ),
                            "label_selector": label,
                        }
        except Exception as e:
            logger.error(f"Error getting processor status: {e}")

        # Fallback status
        return {
            "status": "unknown",
            "message": "Unable to determine processor status",
            "timestamp": datetime.utcnow().isoformat(),
        }


def get_app() -> FastAPI:
    return create_app()
