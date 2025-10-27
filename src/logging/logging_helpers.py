"""Helper functions for logging across the codebase."""

from src.logging.logger import Logger
from typing import Optional, Any, Dict, Union
import torch
import os
from datetime import datetime

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
    engine,  # This could be DeepSpeed engine or Accelerate-unwrapped model
    optimizer,
    loss: float,
    config: Dict[str, Any],
    rank: int,
    is_final: bool = False
):
    """Save checkpoint - works with both DeepSpeed and Accelerate."""
    if rank != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_name = "checkpoint_final.pt" if is_final else f"checkpoint_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    print(f"Saving checkpoint to {checkpoint_path}")
    
    # Handle both DeepSpeed engine and Accelerate unwrapped model
    if hasattr(engine, 'module'):
        # DeepSpeed engine
        model_state = engine.module.state_dict()
    else:
        # Accelerate unwrapped model or regular PyTorch model
        model_state = engine.state_dict()
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved successfully to {checkpoint_path}")



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
