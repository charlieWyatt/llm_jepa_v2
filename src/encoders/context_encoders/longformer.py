from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from src.encoders.base import ContextEncoder
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List


class Longformer(ContextEncoder):
    LOCAL_LONGFORMER_PATH = Path(
        "/g/data/oy87/cw9909/hf_models/allenai_longformer-base-4096")

    def __init__(
        self,
        model_id: str | None = None,
        config: dict | None = None,
        **kwargs,
    ):
        """Load Longformer from local checkpoint."""
        super().__init__()
        local_dir = self.LOCAL_LONGFORMER_PATH

        if not local_dir.exists():
            raise FileNotFoundError(
                f"Local model directory not found: {local_dir}")

        self._tokenizer = LongformerTokenizer.from_pretrained(
            local_dir, local_files_only=True
        )

        if config:
            base = LongformerConfig.from_pretrained(
                local_dir, local_files_only=True)
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            self.model = LongformerModel(base)
        else:
            self.model = LongformerModel.from_pretrained(
                local_dir, local_files_only=True)

    @property
    def tokenizer(self):
        return self._tokenizer

    def get_input_embeddings(self) -> nn.Module:
        """Return the input embedding layer for patching."""
        return self.model.embeddings.word_embeddings

    def get_embeddings(self, input_data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Get word embeddings from text or token IDs."""
        device = next(self.model.parameters()).device

        token_ids: torch.Tensor
        if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            token_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(input_data)], device=device)
        elif isinstance(input_data, str):
            encoded = self.tokenizer.encode(input_data, return_tensors="pt")
            token_ids = encoded.to(device)
        else:
            # Already a tensor or convertible to tensor
            if isinstance(input_data, torch.Tensor):
                token_ids = input_data.to(device)
            else:
                token_ids = torch.tensor(input_data, device=device)

        return self.model.embeddings.word_embeddings(token_ids)

    def forward(self, 
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode text inputs and return representations."""
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
