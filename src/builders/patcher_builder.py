from src.builders.base_builder import EnumBuilder
from enum import Enum


class TokenPatcher:
    pass


class PatchStrategy(Enum):
    token = TokenPatcher


class patcher_builder(EnumBuilder[PatchStrategy]):
    def __init__(self, strategy: str | None) -> None:
        super().__init__(strategy, PatchStrategy, label="Patch Strategy")
