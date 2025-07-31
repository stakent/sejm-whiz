from datetime import datetime
from typing import Dict, Any
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sejm Whiz API",
        description="AI-driven legal prediction system using Polish Parliament data",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    configure_cors(app)
    configure_error_handlers(app)
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


def configure_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                detail=str(exc.detail),
                timestamp=datetime.utcnow()
            ).model_dump()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                detail=str(exc),
                timestamp=datetime.utcnow()
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                detail="An internal server error occurred",
                timestamp=datetime.utcnow()
            ).model_dump()
        )


def configure_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow()
        )
    
    @app.get("/")
    async def root():
        return {
            "message": "Sejm Whiz API",
            "version": "0.1.0",
            "docs": "/docs"
        }


def get_app() -> FastAPI:
    return create_app()