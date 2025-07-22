"""Logging configuration for Shvayambhu project."""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log levels from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_TYPE = os.getenv("LOG_FORMAT", "json")

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "consciousness_state"):
            log_obj["consciousness_state"] = record.consciousness_state
            
        if hasattr(record, "memory_usage"):
            log_obj["memory_usage"] = record.memory_usage
            
        if hasattr(record, "model_performance"):
            log_obj["model_performance"] = record.model_performance
            
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

class ConsciousnessLogger:
    """Logger for consciousness-related events."""
    
    def __init__(self, name: str = "consciousness"):
        self.logger = logging.getLogger(name)
        
    def log_thought(self, thought: str, metadata: Dict[str, Any] = None):
        """Log a conscious thought process."""
        extra = {"consciousness_state": metadata} if metadata else {}
        self.logger.info(f"Thought: {thought}", extra=extra)
        
    def log_introspection(self, introspection: str, depth: int = 0):
        """Log introspective analysis."""
        extra = {"consciousness_state": {"introspection_depth": depth}}
        self.logger.info(f"Introspection: {introspection}", extra=extra)
        
    def log_self_awareness(self, level: float, description: str):
        """Log self-awareness level changes."""
        extra = {"consciousness_state": {"self_awareness_level": level}}
        self.logger.info(f"Self-awareness: {description}", extra=extra)

def setup_logging():
    """Set up logging configuration for the entire project."""
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    
    # File handlers
    # Main application log
    app_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "shvayambhu.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    app_handler.setLevel(logging.INFO)
    
    # Error log
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    
    # Consciousness log
    consciousness_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "consciousness.log",
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    consciousness_handler.setLevel(logging.DEBUG)
    
    # Training log
    training_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "training.log",
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5
    )
    training_handler.setLevel(logging.INFO)
    
    # Set formatters
    if LOG_FORMAT_TYPE == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    for handler in [console_handler, app_handler, error_handler, 
                   consciousness_handler, training_handler]:
        handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    
    # Specialized loggers
    consciousness_logger = logging.getLogger("consciousness")
    consciousness_logger.addHandler(consciousness_handler)
    consciousness_logger.setLevel(logging.DEBUG)
    
    training_logger = logging.getLogger("training")
    training_logger.addHandler(training_handler)
    training_logger.setLevel(logging.INFO)
    
    # Log startup
    root_logger.info("Shvayambhu logging system initialized", extra={
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT_TYPE,
        "log_dir": str(LOG_DIR.absolute())
    })
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)

def log_memory_usage(logger: logging.Logger, memory_gb: float):
    """Log memory usage with context."""
    logger.info(
        f"Memory usage: {memory_gb:.2f} GB",
        extra={"memory_usage": {"used_gb": memory_gb}}
    )

def log_model_performance(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log model performance metrics."""
    logger.info(
        "Model performance update",
        extra={"model_performance": metrics}
    )

# Initialize logging on import
if __name__ != "__main__":
    setup_logging()