"""Helper functions for logging across the codebase."""

from src.logging.logger import Logger
from typing import Optional

# Global reference to avoid repeated lookups
_logger_instance: Optional[Logger] = None


def get_logger(rank: int) -> Logger:
    """
    Get the singleton logger instance for the given rank.

    This function can be called from any file - it will always return
    the same Logger instance for a given rank.

    Args:
        rank: GPU rank (usually from environment or distributed setup)

    Returns:
        Logger instance for the given rank
    """
    return Logger.get_instance(rank)


def logi(msg: str, rank: int):
    """
    Convenience function for logging info messages.

    Args:
        msg: Message to log
        rank: GPU rank
    """
    logger = get_logger(rank)
    logger.log(msg)


def log_once(msg: str, rank: int):
    """
    Log message only from rank 0.

    Args:
        msg: Message to log
        rank: GPU rank
    """
    logger = get_logger(rank)
    logger.log_only_one_gpu(msg)
