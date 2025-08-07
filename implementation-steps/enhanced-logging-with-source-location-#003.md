IMPLEMENTATION BRIEF #003
Priority: Medium
Category: Infrastructure/Logging

Business Requirement:

- Enhance error logging across the codebase to include Python file name and line number information
- Improve debugging efficiency by making it easier to correlate log error messages with originating code
- Provide consistent, detailed error context for production troubleshooting and development

Technical Scope:

- Update logging configuration to automatically include source location (filename:line_number)
- Enhance exception handling throughout the codebase with structured error information
- Implement centralized logging utilities with source location tracking
- Update existing log statements to provide better context and traceability
- Add caller information to critical error paths (API clients, data processing, database operations)

Acceptance Criteria:

- [ ] All log messages start with clearly visible timestamps in YYYY-MM-DD HH:MM:SS format
- [ ] All error log messages include Python file name and line number
- [ ] Log format consistently shows: `YYYY-MM-DD HH:MM:SS | LEVEL | relative/file/path.py:line_number | message`
- [ ] Exception handling preserves stack trace information with source locations
- [ ] Critical error paths (ELI API, Sejm API, database, embeddings) have enhanced logging
- [ ] Validation errors include source context for easier debugging
- [ ] Log configuration is centralized and easily configurable
- [ ] Performance impact is minimal (logging overhead < 5% in benchmarks)
- [ ] Log messages remain readable and don't become overly verbose
- [ ] Integration with existing Rich console output for CLI commands

Implementation Strategy:

### Phase 1: Centralized Logging Configuration (30 minutes)

**1.1 Create Enhanced Logging Component**

- Create `components/sejm_whiz/logging/` component directory
- Implement `enhanced_logger.py` with source location tracking
- Add `SourceLocationFormatter` class for filename:line_number display
- Add `EnhancedLogger` class with automatic caller context for errors
- Create `setup_enhanced_logging()` function to replace scattered `basicConfig()` calls

**1.2 Update Existing Entry Points (8 files identified)**

- `enhanced_data_processor.py:551` - Replace basicConfig with setup_enhanced_logging
- `simple_full_ingestion.py:245` - Update logging configuration
- `projects/data_processor/main.py:374` - Centralize logging setup
- `full_2025_ingestion.py:429` - Update to use enhanced logging
- Other ingestion scripts: `ingest_february_2024.py`, `ingest_january_2024.py`
- Benchmark scripts: `benchmark_full_pipeline.py`, `benchmark_pipeline_simple.py`
- Test scripts with logging: `test_cache_system.py`, `test_cache_performance_january2025.py`

### Phase 2: Enhanced Exception Handling (45 minutes)

**2.1 ELI API Client Enhancement (client.py)**

- Line 288: `logger.warning(f"Failed to parse document: {e}")` → Add document context
- Line 330: `logger.error(f"Failed to fetch document {eli_id}: {e}")` → Add caller info
- Line 407: `logger.error(f"Failed to fetch amendments for {eli_id}: {e}")` → Enhance
- Line 435: `logger.error(f"Failed to fetch recent {doc_type} documents: {e}")` → Context
- Line 536: `logger.error(f"Failed to fetch document {eli_id} in batch: {e}")` → Batch info

**2.2 Sejm API Client Enhancement (sejm_api/client.py)**

- Line 769: `logger.error(f"Health check failed: {e}")` → Add URL context
- Add source location to rate limiting warnings and request errors
- Enhance validation error messages with parameter context

**2.3 Database Operations Enhancement**

- `database/operations.py` - Add source location to SQL errors
- `database/connection.py` - Enhance connection failure messages
- `vector_db/operations.py` - Add context to vector operation failures
- Include transaction IDs and operation context in error messages

### Phase 3: Validation and Model Error Enhancement (30 minutes)

**3.1 Pydantic Model Validation**

- Enhance validation error handling in data models
- Add source location context to model validation failures
- Improve error messages for ELI document parsing errors
- Add context about input data that caused validation failures

**3.2 Configuration and Setup Errors**

- Enhance configuration loading error messages
- Add source location to environment setup failures
- Improve error context for missing dependencies or configurations

### Phase 4: Integration and Testing (15 minutes)

**4.1 CLI Integration**

- Ensure enhanced logging works with Rich console output
- Maintain readable error messages in CLI commands
- Add debug mode with extended source location information

**4.2 Performance Validation**

- Benchmark logging overhead with source location tracking
- Ensure performance impact is acceptable
- Add configuration option to disable source location in production if needed

Technical Implementation Details:

### Current Logging Analysis

**Current Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
**Problem Example**: Line 288 in `client.py` logs: `"Failed to parse document: {e}"` without source context
**Timestamp Issue**: Need to ensure timestamps are clearly visible at the start of every log message

**Logging Usage Patterns Found**:

- 60+ files use `logger = logging.getLogger(__name__)`
- Multiple `logging.basicConfig()` configurations across entry points
- Error logging in critical paths: ELI API (line 288), Sejm API, database operations
- Validation errors from Pydantic models lack source context

### Enhanced Logger Implementation

```python
# components/sejm_whiz/logging/enhanced_logger.py
import logging
import inspect
import os
from typing import Optional

class SourceLocationFormatter(logging.Formatter):
    """Custom formatter that adds relative file path and line number to log records."""

    def format(self, record):
        # Convert absolute path to relative path from project root
        if hasattr(record, 'pathname'):
            # Find the project root (where pyproject.toml or .git exists)
            current_path = os.path.abspath(record.pathname)
            project_root = self._find_project_root(current_path)

            if project_root:
                # Get relative path from project root
                try:
                    record.filepath = os.path.relpath(current_path, project_root)
                except ValueError:
                    # Fallback if paths are on different drives (Windows)
                    record.filepath = os.path.basename(current_path)
            else:
                record.filepath = os.path.basename(current_path)
        else:
            record.filepath = 'unknown'

        return super().format(record)

    def _find_project_root(self, start_path):
        """Find project root by looking for pyproject.toml or .git directory."""
        current_path = os.path.dirname(start_path)
        while current_path != os.path.dirname(current_path):  # Not at filesystem root
            if (os.path.exists(os.path.join(current_path, 'pyproject.toml')) or
                os.path.exists(os.path.join(current_path, '.git'))):
                return current_path
            current_path = os.path.dirname(current_path)
        return None

class EnhancedLogger(logging.Logger):
    """Logger that automatically includes caller context in error messages."""

    def error(self, msg, *args, **kwargs):
        if not kwargs.get('exc_info') and not kwargs.get('stack_info'):
            # Add caller context for error messages
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            enhanced_msg = f"{msg} (at {filename}:{lineno})"
            super().error(enhanced_msg, *args, **kwargs)
        else:
            super().error(msg, *args, **kwargs)

def setup_enhanced_logging(level=logging.INFO, include_source=True, timestamp_format='%Y-%m-%d %H:%M:%S'):
    """Setup enhanced logging configuration with prominent timestamps and file paths."""
    if include_source:
        format_string = "%(asctime)s | %(levelname)s | %(filepath)s:%(lineno)d | %(message)s"
    else:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=timestamp_format,
        handlers=[logging.StreamHandler()]
    )

    # Set our custom formatter for all handlers with timestamp format
    formatter = SourceLocationFormatter(format_string)
    formatter.datefmt = timestamp_format
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

def get_enhanced_logger(name: str) -> logging.Logger:
    """Get logger with enhanced error reporting capabilities."""
    logging.setLoggerClass(EnhancedLogger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)  # Reset to default
    return logger
```

### Exception Context Enhancement

```python
# Enhanced exception handling pattern
def enhanced_exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get current frame info
            frame = inspect.currentframe()
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno

            logger.error(
                f"Exception in {func.__name__} at {os.path.basename(filename)}:{line_number}: {e}",
                exc_info=True
            )
            raise
    return wrapper
```

### Validation Error Enhancement

```python
# Enhanced Pydantic model with better error context
class LegalDocument(BaseModel):
    eli_id: str

    @validator('eli_id')
    def validate_eli_id(cls, v):
        if not v or not v.strip():
            # Get caller information
            frame = inspect.currentframe().f_back
            caller_info = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
            raise ValueError(f"ELI ID cannot be empty (called from {caller_info})")
        return v
```

Example Enhanced Error Messages:

**Before (line 288 in client.py):**

```
2025-01-13 10:30:15 - sejm_whiz.eli_api.client - WARNING - Failed to parse document: 1 validation error for LegalDocument
eli_id
  Value error, ELI ID cannot be empty [type=value_error, input_value='', input_type=str]
```

**After with Enhanced Logging (file path format):**

```
2025-01-13 10:30:15 | WARNING | components/sejm_whiz/eli_api/client.py:288 | Failed to parse document: 1 validation error for LegalDocument (at client.py:288)
eli_id
  Value error, ELI ID cannot be empty [type=value_error, input_value='', input_type=str]
  Document index: 5, API response batch: DU/2025/, Total processed: 127
```

**Critical Path Examples:**

- `components/sejm_whiz/eli_api/client.py:288` - Document parsing failures in ELI API responses
- `components/sejm_whiz/eli_api/client.py:330` - Document fetch failures with ELI IDs
- `components/sejm_whiz/eli_api/client.py:407` - Amendment fetch errors
- `components/sejm_whiz/database/operations.py:XX` - Database operation failures
- `components/sejm_whiz/document_ingestion/ingestion_pipeline.py:XX` - Pipeline processing errors

Configuration Options:

```python
# Enhanced logging configuration
LOGGING_CONFIG = {
    'include_source_location': True,    # Add filename:line_number
    'include_caller_context': True,     # Add caller information in exceptions
    'timestamp_format': '%Y-%m-%d %H:%M:%S',  # Clear timestamp format
    'timestamp_first': True,            # Start log messages with timestamp
    'debug_mode': False,                # Extended debug information
    'performance_mode': False,          # Minimal logging overhead
    'use_pipe_separators': True,        # Use | instead of - for clarity
}

# Example usage
setup_enhanced_logging(
    level=logging.INFO,
    include_source=True,
    timestamp_format='%Y-%m-%d %H:%M:%S'
)
```

Constraints:

- Timeline: 2 hours for complete implementation
- Dependencies: Python inspect module, existing logging infrastructure
- Performance: Logging overhead must remain under 5% impact
- Compatibility: Must work with existing Rich console output and CLI interface
- Maintainability: Centralized configuration for easy management

Context: Error messages like the Pydantic validation error shown lack source location context, making debugging difficult in production. Enhanced logging will provide immediate visibility into where errors occur, reducing debugging time and improving development efficiency.

Benefits:

- **Faster Debugging**: Immediate identification of error source locations with full file paths
- **Better Production Support**: Clear error context for troubleshooting with precise file locations
- **Improved Development Experience**: Easier correlation between logs and code - can directly navigate to `components/sejm_whiz/eli_api/client.py:288`
- **Consistent Error Reporting**: Standardized error message format across codebase
- **Enhanced Observability**: Better insights into application behavior and failure patterns
- **IDE Integration**: File paths are clickable in many IDEs and log viewers
- **Eliminates Redundancy**: No duplicate module name and filename information
- **Project-Relative Paths**: Clear context within the project structure rather than absolute system paths

Success Metrics:

- All critical error paths include source location information
- Average debugging time reduced by identifying error sources quickly
- Log messages provide actionable information for developers
- No significant performance degradation in logging-intensive operations
- Consistent error message format across all components
