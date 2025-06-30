from src.builders.base_builder import EnumBuilder
from enum import Enum


class TokenPatcher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_patches(self, text: str):
        return self.tokenizer.tokenize(text)


class PatchStrategy(Enum):
    token = TokenPatcher


class patcher_builder(EnumBuilder[PatchStrategy]):
    def __init__(self, strategy: str | None) -> None:
        super().__init__(strategy, PatchStrategy, label="Patch Strategy")

    def build(self, context_encoder, target_encoder):
        context_tokenizer = context_encoder.tokenizer
        target_tokenizer = target_encoder.tokenizer

        if context_tokenizer != target_tokenizer:
            raise ValueError(
                "Context and Target encoder tokenizers do not match")

        patcher_class = self.get_class()
        return patcher_class(tokenizer=context_tokenizer)
