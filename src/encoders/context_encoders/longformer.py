from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from src.encoders.base import ContextEncoder
import torch


class Longformer(ContextEncoder):
    def __init__(self, model_id: str = "allenai/longformer-base-4096", config: dict | None = None):
        super().__init__()
        if not isinstance(model_id, str):
            raise TypeError(
                "model_id must be a Hugging Face repo id or path string")

        self._tokenizer = LongformerTokenizer.from_pretrained(model_id)

        if config:  # custom architecture → random init
            base = LongformerConfig.from_pretrained(model_id)
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            # random weights with overridden config
            self.model = LongformerModel(base)
        else:       # use pretrained checkpoint as-is
            self.model = LongformerModel.from_pretrained(model_id)

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, input_texts):
        inputs = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt", max_length=4096
        )

        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        return self.model(**inputs).last_hidden_state
