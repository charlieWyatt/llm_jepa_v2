from src.builders.base_builder import EnumBuilder
from enum import Enum


class L2LossCalculator:
    pass


class LossStrategy(Enum):
    l2 = L2LossCalculator


class loss_calculator_builder(EnumBuilder[LossStrategy]):
    def __init__(self, strategy) -> None:
        super().__init__(strategy, LossStrategy, label="Loss Strategy")
