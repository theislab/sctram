#!/usr/bin/env python3

from contextlib import contextmanager
from scanpy._settings import settings  # This is Scanpy's settings, including verbosity and _root_logger
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
    scanpy_level : bool, optional
        If True, use the log level coming from Scanpy; otherwise, force "INFO".
    """
    # Save the original verbosity level and set verbosity to 4.
    original_verbosity = settings.verbosity
    settings.verbosity = 4

    # Use Scanpy’s internal logger.
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
        # Restore the original handlers and verbosity level.
        scanpy_logger.handlers = original_handlers
        settings.verbosity = original_verbosity
