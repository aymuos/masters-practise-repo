import logging
import sys

def get_logger(name: str):
    """
    Create a process-wide logger with timestamp, level and message.
    Emits to stdout so DVC/CI can capture it.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Avoid double logging if imported multiple times
    logger.propagate = False
    return logger
