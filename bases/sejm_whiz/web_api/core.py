from datetime import datetime
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

# Import components for API functionality
try:
    from sejm_whiz.semantic_search import SemanticSearchEngine
    from sejm_whiz.prediction_models import (
        PredictionInput,
        create_default_similarity_predictor,
        create_default_ensemble,
    )

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.5
    document_type: Optional[str] = None


class SearchResult(BaseModel):
    document_id: str
    title: str
    content: str
    document_type: str
    similarity_score: float
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    query: str
    processing_time_ms: float


class PredictionRequest(BaseModel):
    document_text: str
    context_documents: Optional[List[str]] = None
    model_type: str = "similarity"


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    prediction_type: str
    processing_time_ms: float


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
            docker_available = False
            container_name = None

            # Try to find the processor container
            processor_containers = [
                "sejm-whiz-processor-dev",
                "sejm-whiz-processor",
                "processor",
            ]

            # Check which processor containers are available
            for container in processor_containers:
                try:
                    check_process = await asyncio.create_subprocess_exec(
                        "docker",
                        "ps",
                        "--filter",
                        f"name={container}",
                        "--format",
                        "{{.Names}}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await check_process.communicate()
                    if check_process.returncode == 0 and stdout.strip():
                        docker_available = True
                        container_name = container
                        yield f"data: {datetime.utcnow().isoformat()} - dashboard - INFO - Found processor container: {container}\n\n"
                        break
                except (FileNotFoundError, OSError):
                    continue

            if docker_available and container_name:
                try:
                    # Stream logs from Docker container
                    process = await asyncio.create_subprocess_exec(
                        "docker",
                        "logs",
                        "-f",
                        "--tail=50",
                        container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )

                    if process.stdout:
                        async for line in process.stdout:
                            decoded_line = line.decode(
                                "utf-8", errors="replace"
                            ).strip()
                            if decoded_line:
                                yield f"data: {decoded_line}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Failed to stream from docker: {e}")
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
            # Try to find processor containers
            processor_containers = [
                "sejm-whiz-processor-dev",
                "sejm-whiz-processor",
                "processor",
            ]

            for container_name in processor_containers:
                result = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        container_name,
                        "--format",
                        "{{json .}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    import json

                    container_data = json.loads(result.stdout)
                    state = container_data.get("State", {})
                    config = container_data.get("Config", {})

                    # Determine processor type from environment variables
                    env_vars = config.get("Env", [])
                    processor_type = "CPU"  # Default
                    for env_var in env_vars:
                        if env_var.startswith("EMBEDDING_DEVICE="):
                            device = env_var.split("=", 1)[1]
                            if device.lower() in ["cuda", "gpu"]:
                                processor_type = "GPU"
                            break

                    # Map Docker state to readable status
                    if state.get("Running"):
                        status = "Running"
                    elif state.get("Paused"):
                        status = "Paused"
                    elif state.get("Restarting"):
                        status = "Restarting"
                    elif state.get("Dead"):
                        status = "Dead"
                    else:
                        status = "Stopped"

                    return {
                        "status": status,
                        "processor_type": processor_type,
                        "container_name": container_name,
                        "started_at": state.get("StartedAt"),
                        "finished_at": state.get("FinishedAt"),
                        "exit_code": state.get("ExitCode"),
                        "pid": state.get("Pid"),
                        "health": state.get("Health", {}).get("Status", "unknown"),
                        "environment": "docker-compose",
                    }
        except Exception as e:
            logger.error(f"Error getting processor status: {e}")

        # Fallback status
        return {
            "status": "unknown",
            "message": "Unable to determine processor status",
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/api/services/status")
    async def services_status():
        """Get the status of all Docker Compose services."""
        try:
            # Get all containers related to sejm-whiz
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    "name=sejm-whiz",
                    "--format",
                    "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}\t{{.CreatedAt}}",
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
                    if len(parts) >= 4:
                        name = parts[0]
                        status = parts[1]
                        image = parts[2]
                        ports = parts[3]
                        created = parts[4] if len(parts) > 4 else "Unknown"

                        # Determine service type
                        service_type = "unknown"
                        if "postgres" in name.lower():
                            service_type = "database"
                        elif "redis" in name.lower():
                            service_type = "cache"
                        elif "processor" in name.lower():
                            service_type = "processor"
                        elif "api" in name.lower():
                            service_type = "api"
                        elif "web" in name.lower():
                            service_type = "web"

                        # Parse status
                        running = "Up" in status

                        services.append(
                            {
                                "name": name,
                                "status": "running" if running else "stopped",
                                "service_type": service_type,
                                "image": image,
                                "ports": ports,
                                "created_at": created,
                                "status_detail": status,
                            }
                        )

            # Also check Docker Compose status if available
            compose_status = {}
            try:
                compose_result = subprocess.run(
                    ["docker", "compose", "ps", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd="/home/d/project/sejm-whiz/sejm-whiz-dev/deployments/docker",
                )
                if compose_result.returncode == 0:
                    import json

                    compose_data = json.loads(compose_result.stdout)
                    if isinstance(compose_data, list):
                        for service in compose_data:
                            compose_status[service.get("Name", "")] = {
                                "service": service.get("Service", ""),
                                "state": service.get("State", ""),
                                "health": service.get("Health", ""),
                            }
            except Exception as e:
                logger.debug(f"Could not get docker compose status: {e}")

            return {
                "services": services,
                "compose_status": compose_status,
                "total_services": len(services),
                "running_services": len(
                    [s for s in services if s["status"] == "running"]
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting services status: {e}")
            return {
                "services": [],
                "compose_status": {},
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # Semantic Search API endpoints
    if COMPONENTS_AVAILABLE:

        @app.post("/api/v1/search", response_model=SearchResponse)
        async def semantic_search(request: SearchRequest):
            """Perform semantic search on legal documents."""
            import time

            start_time = time.time()

            try:
                # Initialize search engine
                search_engine = SemanticSearchEngine()

                # Perform search
                results = search_engine.search(
                    query=request.query,
                    limit=request.limit,
                    document_type=request.document_type,
                    similarity_threshold=request.threshold,
                )

                # Convert results to API format
                search_results = []
                for result in results:
                    content = result.document.content
                    truncated_content = (
                        content[:500] + "..." if len(content) > 500 else content
                    )

                    search_results.append(
                        SearchResult(
                            document_id=str(result.document.id),
                            title=result.document.title,
                            content=truncated_content,
                            document_type=result.document.document_type,
                            similarity_score=result.similarity_score,
                            metadata=result.search_metadata,
                        )
                    )

                processing_time = (time.time() - start_time) * 1000

                return SearchResponse(
                    results=search_results,
                    total_results=len(search_results),
                    query=request.query,
                    processing_time_ms=processing_time,
                )

            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @app.get("/api/v1/search")
        async def semantic_search_get(
            q: str = Query(..., description="Search query"),
            limit: int = Query(10, description="Maximum number of results"),
            threshold: float = Query(0.5, description="Similarity threshold"),
            document_type: Optional[str] = Query(
                None, description="Filter by document type"
            ),
        ):
            """Perform semantic search via GET request."""
            request = SearchRequest(
                query=q, limit=limit, threshold=threshold, document_type=document_type
            )
            return await semantic_search(request)

        @app.post("/api/v1/predict", response_model=PredictionResponse)
        async def predict_law_changes(request: PredictionRequest):
            """Predict potential law changes based on document analysis."""
            import time

            start_time = time.time()

            try:
                # Initialize predictor based on model type
                if request.model_type == "similarity":
                    predictor = create_default_similarity_predictor()
                elif request.model_type == "ensemble":
                    predictor = create_default_ensemble()
                else:
                    raise ValueError(f"Unknown model type: {request.model_type}")

                # Create prediction input
                prediction_input = PredictionInput(
                    document_text=request.document_text,
                    context_documents=request.context_documents or [],
                )

                # Generate prediction
                result = await predictor.predict(prediction_input)

                processing_time = (time.time() - start_time) * 1000

                return PredictionResponse(
                    prediction=result.prediction,
                    confidence=result.confidence,
                    prediction_type=request.model_type,
                    processing_time_ms=processing_time,
                )

            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

    else:

        @app.get("/api/v1/search")
        async def search_unavailable():
            raise HTTPException(
                status_code=503,
                detail="Search functionality unavailable - components not loaded",
            )

        @app.post("/api/v1/predict")
        async def predict_unavailable():
            raise HTTPException(
                status_code=503,
                detail="Prediction functionality unavailable - components not loaded",
            )


def get_app() -> FastAPI:
    return create_app()
