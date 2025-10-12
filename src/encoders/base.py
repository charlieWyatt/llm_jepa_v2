from abc import ABC, abstractmethod
import torch.nn as nn


class ContextEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        pass

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """
        Return the input embedding layer.

        This is needed for patching operations where we need to convert
        token IDs to embeddings before applying patch aggregation.

        Returns:
            nn.Module: The embedding layer (e.g., model.embeddings.word_embeddings)
        """
        pass
