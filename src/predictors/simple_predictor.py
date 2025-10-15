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
        num_targets: int  # ← Accept pre-computed global max
    ) -> torch.Tensor:
        """
        Args:
            context_repr: [B, L, D] - context representations
            target_mask: [B, L] - 1 where to predict
            context_mask: [B, L] - 1 for valid context positions
            num_targets: Pre-computed global max targets across all ranks

        Returns:
            predictions: [B, num_targets, D]
        """
        B, L, D = context_repr.shape
        device = context_repr.device
        dtype = context_repr.dtype

        if num_targets == 0:
            return torch.zeros(B, 1, D, device=device, dtype=dtype)

        predictions = torch.zeros(B, num_targets, D, device=device, dtype=dtype)

        for i in range(B):
            num_tgt = int(target_mask[i].sum().item())
            num_context = context_mask[i].sum()

            # Calculate context mean
            if num_context > 0:
                context_sum = context_repr[i].sum(dim=0)
                context_mean = context_sum / num_context
            else:
                context_mean = torch.zeros(D, device=device, dtype=dtype)

            # ALWAYS call self.proj for synchronization
            pred = self.proj(context_mean.unsqueeze(0))  # [1, D]
            
            # Only fill if this sample has targets
            if num_tgt > 0:
                repeated_pred = pred.repeat(num_tgt, 1)
                predictions[i, :num_tgt, :] = repeated_pred

        return predictions