#!/usr/bin/env python3

from contextlib import contextmanager
from scanpy._settings import settings
import logging
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

@contextmanager
def redirect_scanpy_logs_to_loguru(custom_caller: str):
    """
    Redirect logs from Scanpy's logger to Loguru with a custom caller string.
    
    Parameters
    ----------
    custom_caller : str
        A custom string to identify the source of the log message.
        For example: "sctram.infer._base:351->sc.pp.neighbors"
    """
    # Use Scanpy’s internal logger (Option B).
    scanpy_logger = settings._root_logger
    original_handlers = scanpy_logger.handlers[:]  # Copy the current handlers.
    
    # Define a custom intercept handler that attaches our custom caller.
    class CustomInterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno
            # Bind the custom caller string and forward the message.
            logger.bind(custom_caller=custom_caller).log(level, record.getMessage())
    
    # Replace Scanpy's handlers with our custom handler.
    scanpy_logger.handlers = [CustomInterceptHandler()]
    
    try:
        yield  # Run the code inside the with-block.
    finally:
        # Restore the original handlers.
        scanpy_logger.handlers = original_handlers
