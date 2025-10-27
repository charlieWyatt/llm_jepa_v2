import torch.nn as nn
import torch

class SimplePredictor(nn.Module):
    """
    Simple predictor that uses full context to predict target representations.

    Uses mean-pooling of context to predict each target position.
    In a more sophisticated version, you'd use cross-attention from targets to context.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(
        self, 
        context_repr: torch.Tensor, 
        target_mask: torch.Tensor, 
        context_mask: torch.Tensor,
        num_targets: int
    ) -> torch.Tensor:
        """
        Args:
            context_repr: [B, L, D] - context representations
            target_mask: [B, L] - 1 where to predict
            context_mask: [B, L] - 1 for valid context positions
            num_targets: Pre-computed global max targets across all ranks

        Returns:
            predictions: [B, num_targets, D] with gradients
        """
        B, L, D = context_repr.shape
        dtype = context_repr.dtype

        # Vectorized context mean computation
        context_sum = (context_repr * context_mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        context_count = context_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        context_mean = (context_sum / context_count).to(dtype=dtype)  # [B, D]
        
        # Project (maintains gradients)
        projected = self.proj(context_mean)  # [B, D]
        
        if num_targets == 0:
            # Empty tensor with computation graph
            return projected.unsqueeze(1)[:, :0, :]  # [B, 0, D]
        
        # Expand to target positions (maintains gradients)
        predictions = projected.unsqueeze(1).expand(B, num_targets, D).contiguous()
        
        return predictions