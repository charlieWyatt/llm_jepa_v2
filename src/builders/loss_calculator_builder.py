from src.builders.base_builder import EnumBuilder
from enum import Enum
import torch


class L2LossCalculator:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def __call__(self, target_embeddings, predicted_embeddings):
        """
        Args:
            target_embeddings: Tensor of shape (batch_size, embedding_dim)
            predicted_embeddings: list of tensors, each of shape (batch_size, embedding_dim)
        Returns:
            Scalar loss
        """
        if not predicted_embeddings:
            raise ValueError("predicted_embeddings is empty")

        stacked_preds = torch.stack(predicted_embeddings, dim=0)
        avg_pred = stacked_preds.mean(dim=0)

        return self.loss_fn(avg_pred, target_embeddings)


class LossStrategy(Enum):
    l2 = L2LossCalculator


class loss_calculator_builder(EnumBuilder[LossStrategy]):
    def __init__(self, strategy) -> None:
        super().__init__(strategy, LossStrategy, label="Loss Strategy")
