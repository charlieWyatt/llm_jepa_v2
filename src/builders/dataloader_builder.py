from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
from src.dataloaders.recipe1m_dataloader import RecipeDataloader
from src.dataloaders.dolma_sample_dataloader import DolmaSampleDataloader


class Dataset(Enum):
    recipe = RecipeDataloader
    dolma_sample = DolmaSampleDataloader


# Type for configuration
DatasetType = Literal["recipe", "dolma_sample"]


class dataloader_builder(EnumBuilder[Dataset]):
    def __init__(self, dataset: str | None):
        super().__init__(dataset, Dataset, label="Dataset")

    def build(self, patcher, batch_size: int = 1, **kwargs):
        dataloader_cls = self.get_class()
        return dataloader_cls(patcher=patcher, batch_size=batch_size, **kwargs)
