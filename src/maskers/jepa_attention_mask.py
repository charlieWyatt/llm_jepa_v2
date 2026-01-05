import torch


def create_jepa_attention_mask(
    patched_attention_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create attention mask that prevents context from attending to targets.
    
    This fixes the JEPA "leakage" problem where the context encoder
    could see target patches during self-attention.
    
    Args:
        patched_attention_mask: [B, L] - 1 for real patches, 0 for padding
        context_mask: [B, L] - 1 for context positions
        target_mask: [B, L] - 1 for target positions
    
    Returns:
        attention_mask: [B, 1, L, L] - additive mask for transformer
    """
    B, L = patched_attention_mask.shape
    device = patched_attention_mask.device
    
    # Start with causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()
    
    # Block attention TO target positions
    # target_mask: [B, L] -> [B, 1, 1, L]
    target_block = target_mask.unsqueeze(1).unsqueeze(2).float()
    
    # Zero out columns corresponding to target positions
    causal_mask = causal_mask * (1 - target_block)
    
    # Apply padding mask
    padding_mask = patched_attention_mask.unsqueeze(1).unsqueeze(2).float()
    causal_mask = causal_mask * padding_mask
    
    # Convert to additive mask (1 -> 0, 0 -> -inf)
    attention_mask = (1.0 - causal_mask) * -10000.0
    
    return attention_mask