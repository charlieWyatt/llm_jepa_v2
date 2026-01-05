from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from src.encoders.base import ContextEncoder
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List


class Longformer(ContextEncoder):
    LOCAL_LONGFORMER_PATH = Path(
        "/g/data/oy87/cw9909/hf_models/allenai_longformer-base-4096")

    # Model-specific constants
    HIDDEN_SIZE = 768
    NUM_LAYERS = 12
    MAX_SEQ_LENGTH = 4096

    def __init__(
        self,
        model_id: str | None = None,
        config: dict | None = None,
        load_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        local_dir = self.LOCAL_LONGFORMER_PATH

        if not local_dir.exists():
            raise FileNotFoundError(f"Local model directory not found: {local_dir}")

        self._tokenizer = LongformerTokenizer.from_pretrained(
            local_dir, local_files_only=True
        )

        base_config = LongformerConfig.from_pretrained(
            local_dir, local_files_only=True
        )
        
        if config:
            for k, v in config.items():
                if hasattr(base_config, k):
                    setattr(base_config, k, v)

        if load_pretrained:
            self.model = LongformerModel.from_pretrained(
                local_dir, 
                config=base_config,
                local_files_only=True
            )
            print("✓ Loaded Longformer with PRETRAINED weights")
        else:
            self.model = LongformerModel(base_config)
            print(f"✓ Initialized Longformer with RANDOM weights: {self.model.num_parameters():,} parameters")

    # === Properties from base class ===
    @property
    def hidden_size(self) -> int:
        return self.HIDDEN_SIZE
    
    @property
    def num_layers(self) -> int:
        return self.NUM_LAYERS
    
    @property
    def max_seq_length(self) -> int:
        return self.MAX_SEQ_LENGTH

    @property
    def tokenizer(self):
        return self._tokenizer

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings.word_embeddings

    def get_embeddings(self, input_data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device

        if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            token_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(input_data)], device=device)
        elif isinstance(input_data, str):
            encoded = self.tokenizer.encode(input_data, return_tensors="pt")
            token_ids = encoded.to(device)
        else:
            if isinstance(input_data, torch.Tensor):
                token_ids = input_data.to(device)
            else:
                token_ids = torch.tensor(input_data, device=device)

        return self.model.embeddings.word_embeddings(token_ids)

    def forward(self, 
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )