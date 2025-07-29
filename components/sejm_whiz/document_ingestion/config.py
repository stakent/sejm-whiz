"""Document ingestion configuration."""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class DocumentIngestionConfig(BaseSettings):
    """Document ingestion configuration with environment variable support."""
    
    # ELI API settings
    eli_api_base_url: str = Field(default="https://eli.gov.pl/api", env="ELI_API_BASE_URL")
    eli_api_timeout: int = Field(default=30, env="ELI_API_TIMEOUT")
    eli_api_max_retries: int = Field(default=3, env="ELI_API_MAX_RETRIES")
    eli_api_rate_limit: int = Field(default=10, env="ELI_API_RATE_LIMIT")  # requests per second
    
    # Document processing settings
    max_document_size: int = Field(default=10485760, env="MAX_DOCUMENT_SIZE")  # 10MB
    supported_formats: List[str] = Field(default=["text/plain", "text/html", "application/pdf"], env="SUPPORTED_FORMATS")
    
    # Text processing settings
    min_text_length: int = Field(default=100, env="MIN_TEXT_LENGTH")
    max_text_length: int = Field(default=1000000, env="MAX_TEXT_LENGTH")  # 1MB
    text_encoding: str = Field(default="utf-8", env="TEXT_ENCODING")
    
    # Content extraction settings
    extract_metadata: bool = Field(default=True, env="EXTRACT_METADATA")
    extract_structure: bool = Field(default=True, env="EXTRACT_STRUCTURE")
    extract_references: bool = Field(default=True, env="EXTRACT_REFERENCES")
    
    # Legal document types to process
    legal_document_types: List[str] = Field(
        default=["ustawa", "kodeks", "rozporządzenie", "konstytucja", "dekret"],
        env="LEGAL_DOCUMENT_TYPES"
    )
    
    # Amendment processing settings
    process_amendments: bool = Field(default=True, env="PROCESS_AMENDMENTS")
    track_omnibus_bills: bool = Field(default=True, env="TRACK_OMNIBUS_BILLS")
    
    # Batch processing settings
    batch_size: int = Field(default=50, env="INGESTION_BATCH_SIZE")
    parallel_workers: int = Field(default=4, env="INGESTION_PARALLEL_WORKERS")
    
    # Storage settings
    temp_dir: str = Field(default="/tmp/sejm_whiz_ingestion", env="INGESTION_TEMP_DIR")
    backup_processed_documents: bool = Field(default=True, env="BACKUP_PROCESSED_DOCUMENTS")
    
    # Quality control settings
    validate_legal_structure: bool = Field(default=True, env="VALIDATE_LEGAL_STRUCTURE")
    require_eli_identifier: bool = Field(default=True, env="REQUIRE_ELI_IDENTIFIER")
    min_quality_score: float = Field(default=0.7, env="MIN_QUALITY_SCORE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @classmethod
    def for_production(cls) -> "DocumentIngestionConfig":
        """Create configuration for production environment."""
        return cls(
            eli_api_timeout=60,
            eli_api_max_retries=5,
            eli_api_rate_limit=5,  # More conservative in production
            batch_size=25,
            parallel_workers=2,
            validate_legal_structure=True,
            require_eli_identifier=True,
            min_quality_score=0.8
        )
    
    @classmethod  
    def for_development(cls) -> "DocumentIngestionConfig":
        """Create configuration for development environment."""
        return cls(
            eli_api_timeout=30,
            eli_api_max_retries=3,
            eli_api_rate_limit=10,
            batch_size=10,
            parallel_workers=1,
            validate_legal_structure=False,
            require_eli_identifier=False,
            min_quality_score=0.5
        )


def get_ingestion_config() -> DocumentIngestionConfig:
    """Get document ingestion configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return DocumentIngestionConfig.for_production()
    else:
        return DocumentIngestionConfig.for_development()