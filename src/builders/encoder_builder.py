from src.builders.base_builder import EnumBuilder
from enum import Enum


class Longformer:
    pass


class EncodingStrategy(Enum):
    longformer = Longformer


class encoder_builder(EnumBuilder[EncodingStrategy]):
    def __init__(self, encoder_strategy: str | None):
        super().__init__(encoder_strategy, EncodingStrategy, label="Encoding Strategy")
