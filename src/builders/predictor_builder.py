from src.builders.base_builder import EnumBuilder
from enum import Enum


class LongformerPredictor:
    pass


class PredictionStrategy(Enum):
    longformer = LongformerPredictor


class predictor_builder(EnumBuilder[PredictionStrategy]):
    def __init__(self, strategy: str | None) -> None:
        super().__init__(strategy, PredictionStrategy, label="Prediction Strategy")
