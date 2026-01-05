from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
from src.encoders.context_encoders.longformer import Longformer
from src.encoders.context_encoders.olmo_encoder import OlmoEncoder


class EncodingStrategy(Enum):
    longformer = Longformer
    olmo = OlmoEncoder


# Type for configuration
EncoderType = Literal["longformer", "olmo"]


class encoder_builder(EnumBuilder[EncodingStrategy]):
    def __init__(self, encoder_strategy: str | None):
        super().__init__(encoder_strategy, EncodingStrategy, label="Encoding Strategy")
