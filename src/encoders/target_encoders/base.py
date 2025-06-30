from abc import abstractmethod
from src.encoders.base import ContextEncoder


class TargetEncoder(ContextEncoder):
    # Target encoder is always derived from the context encoder
    def __init__(self, context_encoder: ContextEncoder | None):
        super().__init__()
        if context_encoder is None:
            raise ValueError("Context Encoder must be defined")
        self.context_encoder = context_encoder

    @property
    def tokenizer(self):
        return self.context_encoder.tokenizer

    @abstractmethod
    def update():
        pass
