
import torch
from typing import Tuple, Any
import torch.nn as nn


def create_patched_embeddings(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_layer: nn.Module,
    patcher: Any,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to create a stream of patches from tokens.

    This applies patching right after tokenization, creating patch embeddings
    that can be used for context/target creation and encoding.

    Args:
        input_ids: [B, L] - Token IDs
        attention_mask: [B, L] - Attention mask (1 for real tokens, 0 for padding)
        embedding_layer: Token embedding layer from the model
        patcher: Patcher instance (MeanPatcher or MaxPatcher)
        patch_size: Size of each patch

    Returns:
        patched_embeddings: [B, L//patch_size, D] - Patched token embeddings
        patched_attention_mask: [B, L//patch_size] - Patched attention mask
    """
    B, L = input_ids.shape

    # Get token embeddings
    token_embeddings = embedding_layer(input_ids)  # [B, L, D]

    # Apply patching to create patch embeddings
    if patch_size > 1:
        patched_embeddings = patcher.patch(
            token_embeddings)  # [B, L//patch_size, D]

        # Also patch attention mask to match
        truncated_length = (L // patch_size) * patch_size
        attention_mask_truncated = attention_mask[:, :truncated_length]
        attention_mask_reshaped = attention_mask_truncated.view(
            B, -1, patch_size)
        patched_attention_mask = attention_mask_reshaped.max(dim=2)[
            0]  # [B, L//patch_size]
    else:
        # patch_size=1: no patching
        patched_embeddings = token_embeddings
        patched_attention_mask = attention_mask

    return patched_embeddings, patched_attention_mask
