from src.builders.base_builder import EnumBuilder
from enum import Enum


class RecipeDataloader:
    pass


class Dataset(Enum):
    recipe = RecipeDataloader


class dataloader_builder(EnumBuilder[Dataset]):
    def __init__(self, dataset: str | None):
        super().__init__(dataset, Dataset, label="Dataset")
