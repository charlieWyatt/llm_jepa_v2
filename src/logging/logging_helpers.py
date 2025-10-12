"""Helper functions for logging across the codebase."""

from src.logging.logger import Logger
from typing import Optional, Any, Dict, Union
import torch
import os

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


def training_loop_log(
    step: int,
    max_steps: int,
    loss: float,
    learning_rate: float,
    grad_norm: float,
    batch_size: int,
    num_patches: int,
    num_targets: int,
    rank: int
):
    """
    Log training loop metrics in a standardized format.

    Args:
        step: Current training step
        max_steps: Maximum number of training steps
        loss: Current loss value
        learning_rate: Current learning rate
        grad_norm: Gradient norm before clipping
        batch_size: Batch size
        num_patches: Number of patches in current batch
        num_targets: Number of target positions
        rank: GPU rank
    """
    logger = get_logger(rank)
    logger.log_only_one_gpu(
        f"Step {step}/{max_steps} | Loss: {loss:.6f} | "
        f"LR: {learning_rate:.2e} | Grad Norm: {grad_norm:.4f} | "
        f"Batch size: {batch_size} | Patches: {num_patches} | Targets: {num_targets}"
    )


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    engine: Any,
    optimizer: Any,
    loss: float,
    config: Union[Dict[str, Any], Any],
    rank: int,
    is_final: bool = False
):
    """
    Save training checkpoint (only on rank 0).

    Args:
        checkpoint_dir: Directory to save checkpoints
        step: Current training step
        engine: DeepSpeed engine
        optimizer: Optimizer instance
        loss: Current loss value
        config: Configuration dictionary or TypedDict
        rank: GPU rank
        is_final: Whether this is the final checkpoint
    """
    if rank != 0:
        return

    logger = get_logger(rank)

    checkpoint_name = "checkpoint_final.pt" if is_final else f"checkpoint_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    logger.log_only_one_gpu(f"Saving checkpoint to {checkpoint_path}")

    torch.save({
        'step': step,
        'model_state_dict': engine.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }, checkpoint_path)

    logger.log_only_one_gpu("Checkpoint saved successfully.")


def calculate_gradient_norm(model: Any) -> float:
    """
    Calculate the L2 norm of all gradients in a model.

    This is useful for:
    - Monitoring training stability (detecting exploding/vanishing gradients)
    - Understanding if gradient clipping is being triggered
    - General training diagnostics

    Args:
        model: PyTorch model or DeepSpeed engine module

    Returns:
        Total gradient norm (L2 norm across all parameters)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
