"""Error handling patterns for Shvayambhu project."""

import functools
import traceback
import logging
from typing import TypeVar, Callable, Optional, Any, Type, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ShvayambhuError(Exception):
    """Base exception for all Shvayambhu errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 details: Optional[dict] = None):
        super().__init__(message)
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class ConsciousnessError(ShvayambhuError):
    """Errors related to consciousness engine."""
    pass

class MemoryError(ShvayambhuError):
    """Errors related to memory constraints."""
    pass

class ModelError(ShvayambhuError):
    """Errors related to model operations."""
    pass

class CompressionError(ShvayambhuError):
    """Errors related to compression/decompression."""
    pass

class WebIntelligenceError(ShvayambhuError):
    """Errors related to web connectivity."""
    pass

@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    module: str
    additional_info: dict
    memory_state: Optional[dict] = None
    consciousness_state: Optional[dict] = None

def with_error_handling(
    exceptions: Union[Type[Exception], tuple] = Exception,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Callable:
    """
    Decorator for consistent error handling across the project.
    
    Args:
        exceptions: Exception types to catch
        default_return: Default value to return on error
        log_level: Logging level for errors
        reraise: Whether to re-raise the exception after logging
        severity: Error severity level
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                # Create error context
                context = ErrorContext(
                    operation=func.__name__,
                    module=func.__module__,
                    additional_info={
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200],
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "severity": severity.value
                    }
                )
                
                # Log the error
                log_func = getattr(logger, log_level)
                log_func(
                    f"Error in {context.operation}: {str(e)}",
                    extra={
                        "error_context": context.__dict__,
                        "traceback": traceback.format_exc()
                    }
                )
                
                # Handle specific error types
                if isinstance(e, MemoryError):
                    handle_memory_error(e, context)
                elif isinstance(e, ConsciousnessError):
                    handle_consciousness_error(e, context)
                
                if reraise:
                    raise
                    
                return default_return
                
        return wrapper
    return decorator

def handle_memory_error(error: MemoryError, context: ErrorContext):
    """Special handling for memory errors."""
    logger.critical(
        f"Memory error in {context.operation}: {error}",
        extra={"memory_state": context.memory_state}
    )
    # Could trigger memory cleanup here
    
def handle_consciousness_error(error: ConsciousnessError, context: ErrorContext):
    """Special handling for consciousness errors."""
    logger.error(
        f"Consciousness error in {context.operation}: {error}",
        extra={"consciousness_state": context.consciousness_state}
    )
    # Could trigger consciousness state recovery

class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self.is_open:
                if (datetime.utcnow() - self.last_failure_time).seconds > self.reset_timeout:
                    self.reset()
                else:
                    raise Exception(f"Circuit breaker is open for {func.__name__}")
                    
            try:
                result = func(*args, **kwargs)
                self.reset()
                return result
            except Exception as e:
                self.record_failure()
                raise
                
        return wrapper
        
    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                        
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, "
                        f"retrying in {wait_time} seconds: {str(e)}"
                    )
                    time.sleep(wait_time)
                    
        return wrapper
    return decorator

class ErrorAggregator:
    """Aggregate errors for batch processing."""
    
    def __init__(self):
        self.errors = []
        
    def add_error(self, error: Exception, context: Optional[dict] = None):
        """Add an error to the aggregator."""
        self.errors.append({
            "error": error,
            "context": context,
            "timestamp": datetime.utcnow()
        })
        
    def get_summary(self) -> dict:
        """Get a summary of aggregated errors."""
        if not self.errors:
            return {"error_count": 0}
            
        error_types = {}
        for error_info in self.errors:
            error_type = type(error_info["error"]).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            "error_count": len(self.errors),
            "error_types": error_types,
            "first_error": str(self.errors[0]["error"]),
            "last_error": str(self.errors[-1]["error"])
        }
        
    def clear(self):
        """Clear all aggregated errors."""
        self.errors = []

# Global error handler for uncaught exceptions
def setup_global_error_handler():
    """Set up global error handling for uncaught exceptions."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
    import sys
    sys.excepthook = handle_exception

# Context manager for error boundaries
class ErrorBoundary:
    """Context manager for creating error boundaries."""
    
    def __init__(self, error_handler: Optional[Callable] = None):
        self.error_handler = error_handler
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.error_handler:
                self.error_handler(exc_type, exc_val, exc_tb)
                return True  # Suppress the exception
            else:
                logger.error(
                    f"Error in error boundary: {exc_type.__name__}: {exc_val}",
                    exc_info=True
                )
        return False