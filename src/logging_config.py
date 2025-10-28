"""
Logging configuration for the web search application.

This module sets up enterprise-grade logging with:
- File rotation (prevents disk space issues)
- Structured logging (JSON format for log aggregation tools)
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Separate handlers for console and file output
- Context information (timestamp, module, function, line number)
"""

import logging
import logging.handlers
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import sys


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    This enables:
    - Easy parsing by log aggregation tools (ELK, Splunk, DataDog)
    - Structured querying of logs
    - Machine-readable log format
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
            
        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (created if doesn't exist)
        enable_console: Whether to log to console (stdout/stderr)
        enable_file: Whether to log to file
        json_format: Use JSON format for file logs (for log aggregation)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging(log_level="DEBUG", json_format=True)
        >>> logger.info("Application started")
        >>> logger.error("Something went wrong", exc_info=True)
    """
    # Create logger
    logger = logging.getLogger("websearch")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers (prevent duplicate logs)
    logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
    
    # Console Handler (for development and debugging)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Human-readable format for console
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File Handler with rotation (prevents disk space issues)
    if enable_file:
        # Main application log (rotates at 10MB, keeps 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=f"{log_dir}/app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Capture everything to file
        
        # Use JSON format for production, human-readable for development
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            format_str = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            file_format = logging.Formatter(
                format_str,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
        
        logger.addHandler(file_handler)
        
        # Error log (only errors and critical issues)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=f"{log_dir}/error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        if json_format:
            error_handler.setFormatter(JSONFormatter())
        else:
            error_handler.setFormatter(file_format)
        
        logger.addHandler(error_handler)
    
    # Log the initialization
    logger.info(
        f"Logging configured: level={log_level}, console={enable_console}, "
        f"file={enable_file}, json={json_format}"
    )
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing search request")
    """
    return logging.getLogger(f"websearch.{name}")


# Performance logging decorator
def log_performance(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> @log_performance(logger)
        ... def search_web(query: str):
        ...     # function implementation
        ...     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"{func.__name__} completed successfully",
                    extra={"duration_ms": duration_ms, "function": func.__name__}
                )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed: {str(e)}",
                    extra={"duration_ms": duration_ms, "function": func.__name__},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Context manager for logging blocks
class LogContext:
    """
    Context manager for logging operation blocks.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, "Searching web"):
        ...     results = search_api("AI news")
        ...     parse_results(results)
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            operation: Description of the operation
            **kwargs: Additional context fields
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Log operation start."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log operation completion."""
        import time
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation}",
                extra={**self.context, "duration_ms": duration_ms}
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} - {exc_val}",
                extra={**self.context, "duration_ms": duration_ms},
                exc_info=True
            )
        
        return False  # Don't suppress exceptions
