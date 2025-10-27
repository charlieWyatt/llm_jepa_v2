import copy
import torch
import torch.nn as nn
from src.encoders.base import ContextEncoder
from src.encoders.target_encoders.base import TargetEncoder


class ema_target_encoder(TargetEncoder):
    def __init__(self, context_encoder: ContextEncoder, ema_decay=0.999):
        super().__init__(context_encoder)
        self.ema_decay = ema_decay
        # Deep copy the model (before it gets wrapped by Accelerate)
        self.model: ContextEncoder = copy.deepcopy(context_encoder)

        # Disable gradient updates on the EMA model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        """Delegate to the wrapped model's get_input_embeddings method."""
        return self.model.get_input_embeddings()

    def update(self, source_encoder: ContextEncoder):
        """
        Update EMA parameters from source encoder.
        
        Args:
            source_encoder: The unwrapped context encoder (pass accelerator.unwrap_model(context_encoder))
        """
        with torch.no_grad():
            ema_params = list(self.model.parameters())
            source_params = list(source_encoder.parameters())
            
            for ema_param, source_param in zip(ema_params, source_params):
                # Simple EMA update - works with any distributed setup
                ema_param.data.mul_(self.ema_decay).add_(
                    source_param.data, alpha=(1.0 - self.ema_decay)
                )