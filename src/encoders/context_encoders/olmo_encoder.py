from transformers import AutoConfig, AutoModel, AutoTokenizer
from src.encoders.base import ContextEncoder
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List


class OlmoEncoder(ContextEncoder):
    # Model-specific constants for OLMo-2-1B-Instruct
    HIDDEN_SIZE = 2048
    NUM_LAYERS = 16
    MAX_SEQ_LENGTH = 4096

    def __init__(
        self,
        model_id: str = "/g/data/oy87/cw9909/hf_models/OLMo-2-0425-1B-Instruct",
        config: dict | None = None,
        load_pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        local_dir = Path(model_id)

        if local_dir.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(
                local_dir, local_files_only=True, trust_remote_code=True
            )
            base_config = AutoConfig.from_pretrained(
                local_dir, local_files_only=True, trust_remote_code=True
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            base_config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if config:
            for k, v in config.items():
                if hasattr(base_config, k):
                    setattr(base_config, k, v)

        if load_pretrained:
            if local_dir.exists():
                self.model = AutoModel.from_pretrained(
                    local_dir,
                    config=base_config,
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    config=base_config,
                    trust_remote_code=True
                )
            print(f"✓ Loaded OLMo with PRETRAINED weights: {self.model.num_parameters():,} parameters")
        else:
            self.model = AutoModel.from_config(base_config, trust_remote_code=True)
            print(f"✓ Initialized OLMo with RANDOM weights: {self.model.num_parameters():,} parameters")

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
        return self.model.get_input_embeddings()

    def get_embeddings(self, input_data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        embed_layer = self.get_input_embeddings()

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

        return embed_layer(token_ids)

    def forward(self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )