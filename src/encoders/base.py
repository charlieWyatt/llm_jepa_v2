from abc import ABC, abstractmethod
import torch.nn as nn


class ContextEncoder(nn.Module, ABC):
    """Base class for all context encoders."""
    
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension of the model."""
        pass
    
    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of layers in the model."""
        pass
    
    @property
    @abstractmethod
    def max_seq_length(self) -> int:
        """Return the maximum sequence length supported."""
        pass
    
    @property
    @abstractmethod
    def tokenizer(self):
        """Return the tokenizer."""
        pass
    
    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Return the input embedding layer."""
        pass