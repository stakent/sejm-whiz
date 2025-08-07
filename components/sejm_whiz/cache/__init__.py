"""Comprehensive caching system for API responses, documents, and processed data."""

from .config import CacheConfig, get_cache_config
from .manager import CacheManager, get_cache_manager
from .document_cache import DocumentContentCache, get_document_cache
from .processed_text_cache import ProcessedTextCache, get_processed_text_cache
from .integration import (
    cached_api_call,
    cached_processing,
    CachedSejmApiClient,
    CachedEliApiClient,
    CacheAwareEmbeddingProcessor,
    CacheAwareDocumentProcessor,
    create_cache_aware_clients,
    get_comprehensive_cache_status,
)
from .management import CacheMaintenanceManager, get_maintenance_manager

__all__ = [
    # Configuration
    "CacheConfig",
    "get_cache_config",
    # Core cache managers
    "CacheManager",
    "get_cache_manager",
    "DocumentContentCache",
    "get_document_cache",
    "ProcessedTextCache",
    "get_processed_text_cache",
    # Decorators and utilities
    "cached_api_call",
    "cached_processing",
    # Cache-aware clients
    "CachedSejmApiClient",
    "CachedEliApiClient",
    "CacheAwareEmbeddingProcessor",
    "CacheAwareDocumentProcessor",
    # Factory functions
    "create_cache_aware_clients",
    "get_comprehensive_cache_status",
    # Cache management
    "CacheMaintenanceManager",
    "get_maintenance_manager",
]
