"""Enhanced logging utilities with file path and source location tracking."""

import logging
import inspect
import os


class SourceLocationFormatter(logging.Formatter):
    """Custom formatter that adds relative file path and line number to log records."""

    def format(self, record):
        # Convert absolute path to relative path from project root
        if hasattr(record, "pathname"):
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
            record.filepath = "unknown"

        return super().format(record)

    def _find_project_root(self, start_path):
        """Find project root by looking for pyproject.toml or .git directory."""
        current_path = os.path.dirname(start_path)
        while current_path != os.path.dirname(current_path):  # Not at filesystem root
            if os.path.exists(
                os.path.join(current_path, "pyproject.toml")
            ) or os.path.exists(os.path.join(current_path, ".git")):
                return current_path
            current_path = os.path.dirname(current_path)
        return None


class EnhancedLogger(logging.Logger):
    """Logger that automatically includes caller context in error messages."""

    def error(self, msg, *args, **kwargs):
        if not kwargs.get("exc_info") and not kwargs.get("stack_info"):
            # Add caller context for error messages
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            enhanced_msg = f"{msg} (at {filename}:{lineno})"
            super().error(enhanced_msg, *args, **kwargs)
        else:
            super().error(msg, *args, **kwargs)


def setup_enhanced_logging(
    level=logging.INFO, include_source=True, timestamp_format="%Y-%m-%d %H:%M:%S"
):
    """Setup enhanced logging configuration with prominent timestamps and file paths."""
    if include_source:
        format_string = (
            "%(asctime)s | %(levelname)s | %(filepath)s:%(lineno)d | %(message)s"
        )
    else:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    # Clear any existing handlers to avoid duplicates
    logging.root.handlers.clear()

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=timestamp_format,
        handlers=[logging.StreamHandler()],
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


# Context manager for adding extra context to log messages
class LogContext:
    """Context manager for adding extra context information to log messages."""

    def __init__(self, logger, **context):
        self.logger = logger
        self.context = context
        self.old_extra = getattr(logger, "_context", {})

    def __enter__(self):
        self.logger._context = {**self.old_extra, **self.context}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._context = self.old_extra


def add_context_to_message(logger, level_name, msg, **context):
    """Add context information to a log message."""
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        return f"{msg} | {context_str}"
    return msg
