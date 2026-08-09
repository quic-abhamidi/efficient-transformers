#!/usr/bin/env python3
"""
GPU Memory Management Utilities

Provides helper functions for managing GPU memory in PyTorch applications.
"""

import gc
from typing import Optional

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


def cleanup_gpu_memory(verbose: bool = True) -> None:
    """
    Aggressively clean up GPU memory.
    
    This function performs the following operations:
    1. Runs Python garbage collection
    2. Empties PyTorch CUDA cache
    3. Synchronizes CUDA operations
    4. Optionally logs memory statistics
    
    Args:
        verbose: If True, logs memory statistics after cleanup
        
    Example:
        >>> from QEfficient.model_pruning.core.gpu_utils import cleanup_gpu_memory
        >>> # After model operations
        >>> del model, tokenizer
        >>> cleanup_gpu_memory()
    """
    try:
        import torch
    except ImportError:
        if verbose:
            logger.warning("PyTorch not available, skipping GPU cleanup")
        return
    
    # Run garbage collection
    gc.collect()
    
    # Clean up CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        
        if verbose:
            # Log memory statistics
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    elif verbose:
        logger.info("CUDA not available, performed CPU garbage collection only")


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary containing memory statistics:
        - allocated_gb: Currently allocated memory in GB
        - reserved_gb: Currently reserved memory in GB
        - max_allocated_gb: Maximum allocated memory in GB
        - max_reserved_gb: Maximum reserved memory in GB
        - available: Whether CUDA is available
        
    Example:
        >>> from QEfficient.model_pruning.core.gpu_utils import get_gpu_memory_stats
        >>> stats = get_gpu_memory_stats()
        >>> print(f"Using {stats['allocated_gb']:.2f}GB")
    """
    try:
        import torch
    except ImportError:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "available": False
        }
    
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "available": False
        }
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "available": True
    }


def reset_peak_memory_stats() -> None:
    """
    Reset peak memory statistics.
    
    Useful for measuring memory usage of specific operations.
    
    Example:
        >>> from QEfficient.model_pruning.core.gpu_utils import reset_peak_memory_stats, get_gpu_memory_stats
        >>> reset_peak_memory_stats()
        >>> # ... perform operations ...
        >>> stats = get_gpu_memory_stats()
        >>> print(f"Peak memory: {stats['max_allocated_gb']:.2f}GB")
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def log_gpu_memory_summary(prefix: str = "") -> None:
    """
    Log a summary of GPU memory usage.
    
    Args:
        prefix: Optional prefix for the log message
        
    Example:
        >>> from QEfficient.model_pruning.core.gpu_utils import log_gpu_memory_summary
        >>> log_gpu_memory_summary("After model loading")
    """
    stats = get_gpu_memory_stats()
    
    if not stats["available"]:
        logger.info(f"{prefix}CUDA not available")
        return
    
    message = f"{prefix}GPU Memory: " \
              f"Allocated={stats['allocated_gb']:.2f}GB, " \
              f"Reserved={stats['reserved_gb']:.2f}GB, " \
              f"Peak Allocated={stats['max_allocated_gb']:.2f}GB"
    
    logger.info(message)
