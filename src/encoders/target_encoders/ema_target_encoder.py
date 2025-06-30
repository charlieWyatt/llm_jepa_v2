import copy
import torch
from src.encoders.base import ContextEncoder
from src.encoders.target_encoders.base import TargetEncoder


class ema_target_encoder(TargetEncoder):
    def __init__(self, context_encoder: ContextEncoder, ema_decay=0.999):
        super().__init__(context_encoder)
        self.ema_decay = ema_decay
        self.model = copy.deepcopy(context_encoder)

        # Disable gradient updates on the EMA model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update(self):
        with torch.no_grad():
            for ema_param, context_param in zip(
                self.model.parameters(), self.context_encoder.parameters()
            ):
                ema_param.data = (
                    self.ema_decay * ema_param.data
                    + (1.0 - self.ema_decay) * context_param.data
                )
