from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
import torch


class L2LossCalculator:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def __call__(self, predicted_embeddings, target_embeddings):
        """
        Args:
            predicted_embeddings: Tensor of shape (batch_size, num_targets, embedding_dim)
            target_embeddings: Tensor of shape (batch_size, num_targets, embedding_dim)
        Returns:
            Scalar loss with gradients
        """
        if predicted_embeddings is None or target_embeddings is None:
            raise ValueError("Embeddings cannot be None")
        
        # Handle empty tensor case (num_targets = 0)
        if predicted_embeddings.numel() == 0:
            if predicted_embeddings.requires_grad:
                return (predicted_embeddings ** 2).sum() * 0.0
            else:
                return torch.tensor(0.0, device=predicted_embeddings.device, 
                                   dtype=predicted_embeddings.dtype)
        
        return self.loss_fn(predicted_embeddings, target_embeddings)



class LossStrategy(Enum):
    l2 = L2LossCalculator


# Type for configuration
LossCalculatorType = Literal["l2"]


class loss_calculator_builder(EnumBuilder[LossStrategy]):
    def __init__(self, strategy) -> None:
        super().__init__(strategy, LossStrategy, label="Loss Strategy")
