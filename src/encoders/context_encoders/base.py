from abc import ABC, abstractmethod
import torch.nn as nn


class ContextEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
