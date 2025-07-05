from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from src.encoders.base import ContextEncoder


class Longformer(ContextEncoder):
    def __init__(self, model_name="allenai/longformer-base-4096", config=None):
        super().__init__()

        config = config or {}
        self.hidden_size = config.get("hidden_size", 256)
        self.num_layers = config.get("num_layers", 4)
        self.attention_window = config.get("attention_window", 512)

        self._tokenizer = LongformerTokenizer.from_pretrained(model_name)
        config = LongformerConfig.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name, config=config)

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, input_texts):
        """
        input_texts: List[str] or str
        Returns: torch.Tensor (hidden states)
        """
        # Tokenize and move tensors to model device
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096,
        )
        return self.model(**inputs).last_hidden_state
