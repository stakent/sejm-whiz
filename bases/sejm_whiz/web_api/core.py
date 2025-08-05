from datetime import datetime, UTC
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from enum import StrEnum

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from sejm_whiz import __version__

# Import components for API functionality
try:
    from sejm_whiz.semantic_search import (
        get_semantic_search_service,
        process_search_query,
        get_production_search_config,
    )
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


class ProcessorStatus(StrEnum):
    """Docker container status values."""

    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    DEAD = "dead"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = __version__


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.5
    document_type: Optional[str] = None
    enable_query_expansion: bool = True
    enable_cross_register: bool = True
    search_mode: str = "hybrid"  # semantic_only, cross_register, hybrid, legal_focused


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
    processed_query: Optional[dict] = None  # ProcessedQuery information
    search_metadata: Optional[dict] = None


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
        version=__version__,
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
                timestamp=datetime.now(UTC).isoformat(),
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
                timestamp=datetime.now(UTC).isoformat(),
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
                timestamp=datetime.now(UTC).isoformat(),
            ).model_dump(),
        )


def configure_routes(app: FastAPI) -> None:
    # Set up templates
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(status="healthy", timestamp=datetime.now(UTC).isoformat())

    @app.get("/")
    async def root():
        return {"message": "Sejm Whiz API", "version": __version__, "docs": "/docs"}

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

            # Check for the processor container
            container_name = "sejm-whiz-processor"
            try:
                check_process = await asyncio.create_subprocess_exec(
                    "docker",
                    "ps",
                    "--filter",
                    f"name={container_name}",
                    "--format",
                    "{{.Names}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await check_process.communicate()
                if check_process.returncode == 0 and stdout.strip():
                    docker_available = True
                    yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Found processor container: {container_name}\n\n"
            except (FileNotFoundError, OSError):
                pass

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
                    yield f"data: {datetime.now(UTC).isoformat()} - dashboard - ERROR - Failed to stream logs: {e}\n\n"

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
            yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - No live logs available, showing demo logs\n\n"

            sample_logs = [
                f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Starting data processor",
                f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Initializing pipeline components",
                f"{datetime.now(UTC).isoformat()} - sejm_ingestion - INFO - [SAMPLE] Fetching Sejm proceedings data",
                f"{datetime.now(UTC).isoformat()} - sejm_ingestion - INFO - [SAMPLE] Retrieved 15 proceedings for session 10",
                f"{datetime.now(UTC).isoformat()} - eli_ingestion - INFO - [SAMPLE] Fetching ELI legal documents",
                f"{datetime.now(UTC).isoformat()} - eli_ingestion - INFO - [SAMPLE] Retrieved 8 legal documents",
                f"{datetime.now(UTC).isoformat()} - text_processing - INFO - [SAMPLE] Processing text data",
                f"{datetime.now(UTC).isoformat()} - text_processing - INFO - [SAMPLE] Cleaned 23 documents",
                f"{datetime.now(UTC).isoformat()} - embedding_generation - INFO - [SAMPLE] Generating embeddings",
                f"{datetime.now(UTC).isoformat()} - embedding_generation - INFO - [SAMPLE] Generated embeddings for 23 documents",
                f"{datetime.now(UTC).isoformat()} - database_storage - INFO - [SAMPLE] Storing data in database",
                f"{datetime.now(UTC).isoformat()} - database_storage - INFO - [SAMPLE] Stored 15 Sejm proceedings",
                f"{datetime.now(UTC).isoformat()} - database_storage - INFO - [SAMPLE] Stored 8 ELI documents",
                f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Pipeline completed successfully",
            ]

            for log in sample_logs:
                yield f"data: {log}\n\n"
                await asyncio.sleep(0.5)

            # Continue with simulated real-time logs
            batch_num = 1
            while True:
                logs = [
                    f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Starting batch {batch_num}",
                    f"{datetime.now(UTC).isoformat()} - sejm_ingestion - INFO - [SAMPLE] Fetching new proceedings",
                    f"{datetime.now(UTC).isoformat()} - text_processing - INFO - [SAMPLE] Processing batch {batch_num}",
                    f"{datetime.now(UTC).isoformat()} - embedding_generation - INFO - [SAMPLE] Generating embeddings for batch {batch_num}",
                    f"{datetime.now(UTC).isoformat()} - database_storage - INFO - [SAMPLE] Storing batch {batch_num} results",
                    f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Batch {batch_num} completed",
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
            # Check for the processor container
            container_name = "sejm-whiz-processor"
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
                    status = ProcessorStatus.RUNNING
                elif state.get("Paused"):
                    status = ProcessorStatus.PAUSED
                elif state.get("Restarting"):
                    status = ProcessorStatus.RESTARTING
                elif state.get("Dead"):
                    status = ProcessorStatus.DEAD
                else:
                    status = ProcessorStatus.STOPPED

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
            "status": ProcessorStatus.UNKNOWN,
            "message": "Unable to determine processor status",
            "timestamp": datetime.now(UTC).isoformat(),
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
                                "status": ProcessorStatus.RUNNING
                                if running
                                else ProcessorStatus.STOPPED,
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
                    [s for s in services if s["status"] == ProcessorStatus.RUNNING]
                ),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting services status: {e}")
            return {
                "services": [],
                "compose_status": {},
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # Semantic Search API endpoints
    if COMPONENTS_AVAILABLE:

        @app.post("/api/v1/search", response_model=SearchResponse)
        async def semantic_search(request: SearchRequest):
            """Perform enhanced semantic search on legal documents with query processing."""
            import time

            start_time = time.time()

            try:
                # Process the query with legal term normalization
                processed_query = process_search_query(
                    request.query, expand_terms=request.enable_query_expansion
                )

                # Get production search configuration
                search_config = get_production_search_config()

                # Override config based on request parameters
                search_config.max_results = request.limit
                search_config.similarity_threshold = request.threshold
                search_config.enable_cross_register = request.enable_cross_register
                search_config.enable_query_expansion = request.enable_query_expansion

                # Set search mode from request
                from sejm_whiz.semantic_search.config import SearchMode

                if request.search_mode == "semantic_only":
                    search_config.search_mode = SearchMode.SEMANTIC_ONLY
                elif request.search_mode == "cross_register":
                    search_config.search_mode = SearchMode.CROSS_REGISTER
                elif request.search_mode == "legal_focused":
                    search_config.search_mode = SearchMode.LEGAL_FOCUSED
                else:
                    search_config.search_mode = SearchMode.HYBRID

                # Initialize enhanced search service
                search_service = get_semantic_search_service()

                # Use the processed query's normalized text for search
                search_query = processed_query.normalized_query

                # Include expanded terms if available
                if processed_query.expanded_terms:
                    search_query += " " + " ".join(
                        processed_query.expanded_terms[:3]
                    )  # Limit to 3 expanded terms

                # Perform enhanced search using the processed query
                results = search_service.search_documents(
                    query=search_query,
                    limit=search_config.max_results,
                    document_type=request.document_type,
                    similarity_threshold=search_config.similarity_threshold,
                    include_cross_register=search_config.enable_cross_register,
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
                    processed_query=processed_query.to_dict(),
                    search_metadata={
                        "search_mode": request.search_mode,
                        "query_expansion_enabled": request.enable_query_expansion,
                        "cross_register_enabled": request.enable_cross_register,
                        "similarity_threshold": request.threshold,
                    },
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
            expand_query: bool = Query(
                True, description="Enable query expansion with synonyms"
            ),
            cross_register: bool = Query(
                True, description="Enable cross-register matching"
            ),
            search_mode: str = Query(
                "hybrid",
                description="Search mode: semantic_only, cross_register, hybrid, legal_focused",
            ),
        ):
            """Perform enhanced semantic search via GET request."""
            request = SearchRequest(
                query=q,
                limit=limit,
                threshold=threshold,
                document_type=document_type,
                enable_query_expansion=expand_query,
                enable_cross_register=cross_register,
                search_mode=search_mode,
            )
            return await semantic_search(request)

        @app.post("/api/v1/query/analyze")
        async def analyze_query(request: dict):
            """Analyze and process a search query to show legal term extraction and normalization."""
            try:
                query = request.get("query", "")
                if not query:
                    raise HTTPException(status_code=400, detail="Query is required")

                expand_terms = request.get("expand_terms", True)

                # Process the query
                processed_query = process_search_query(query, expand_terms=expand_terms)

                return {
                    "query_analysis": processed_query.to_dict(),
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error analyzing query: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Query analysis failed: {str(e)}"
                )

        @app.get("/api/v1/query/analyze")
        async def analyze_query_get(
            q: str = Query(..., description="Query to analyze"),
            expand_terms: bool = Query(True, description="Enable term expansion"),
        ):
            """Analyze a query via GET request."""
            return await analyze_query({"query": q, "expand_terms": expand_terms})

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
