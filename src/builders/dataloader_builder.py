from src.builders.base_builder import EnumBuilder
from enum import Enum
from src.dataloaders.recipe1m_dataloader import RecipeDataloader


class Dataset(Enum):
    recipe = RecipeDataloader


class dataloader_builder(EnumBuilder[Dataset]):
    def __init__(self, dataset: str | None):
        super().__init__(dataset, Dataset, label="Dataset")

    def build(self, patcher, batch_size: int = 1, **kwargs):
        dataloader_cls = self.get_class()
        return dataloader_cls(patcher=patcher, batch_size=batch_size, **kwargs)
