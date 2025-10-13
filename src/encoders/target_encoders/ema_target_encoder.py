import copy
import torch
import torch.nn as nn
from src.encoders.base import ContextEncoder
import deepspeed
from src.encoders.target_encoders.base import TargetEncoder


class ema_target_encoder(TargetEncoder):
    def __init__(self, context_encoder: ContextEncoder, ema_decay=0.999):
        super().__init__(context_encoder)
        self.ema_decay = ema_decay
        self.model: ContextEncoder = copy.deepcopy(context_encoder)

        # Disable gradient updates on the EMA model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        """Delegate to the wrapped model's get_input_embeddings method."""
        return self.model.get_input_embeddings()

    def update(self):
        """Update EMA parameters from context encoder (ZeRO-3 compatible)."""
        with torch.no_grad():
            ema_params = list(self.model.parameters())
            context_params = list(self.context_encoder.parameters())
            
            for ema_param, context_param in zip(ema_params, context_params):
                # Only rank 0 gathers and updates, then broadcasts to all ranks
                with deepspeed.zero.GatheredParameters([ema_param, context_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ema_param.data.mul_(self.ema_decay).add_(
                            context_param.data, alpha=(1.0 - self.ema_decay)
                        )
