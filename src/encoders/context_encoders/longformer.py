from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from src.encoders.base import ContextEncoder
import torch
from pathlib import Path

class Longformer(ContextEncoder):
    # Absolute path to your local checkpoint
    LOCAL_LONGFORMER_PATH = Path("/g/data/oy87/cw9909/hf_models/allenai_longformer-base-4096")

    def __init__(
        self,
        model_id: str | None = None,      # builder may still pass this
        config: dict | None = None,
        **kwargs,
    ):
        """
        If model_id is provided, ignore it and always load from LOCAL_LONGFORMER_PATH.
        """
        super().__init__()
        local_dir = self.LOCAL_LONGFORMER_PATH

        if not local_dir.exists():
            raise FileNotFoundError(f"Local model directory not found: {local_dir}")

        # Load tokenizer from local files only
        self._tokenizer = LongformerTokenizer.from_pretrained(
            local_dir, local_files_only=True
        )

        if config:
            base = LongformerConfig.from_pretrained(local_dir, local_files_only=True)
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            self.model = LongformerModel(base)
        else:
            self.model = LongformerModel.from_pretrained(local_dir, local_files_only=True)

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, input_texts):
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state