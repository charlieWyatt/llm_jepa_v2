from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
from src.dataloaders.recipe1m_dataloader import RecipeDataloader
from src.dataloaders.dolma_sample_dataloader import DolmaSampleDataloader
from src.dataloaders.nl_rx_synth_dataloader import NLRXSynthDataloader
from src.dataloaders.nl_rx_synth_flat_dataloader import NLRXSynthFlatDataloader


class Dataset(Enum):
    recipe = RecipeDataloader
    dolma_sample = DolmaSampleDataloader
    nl_rx_synth = NLRXSynthDataloader
    nl_rx_synth_flat = NLRXSynthFlatDataloader


# Type for configuration
DatasetType = Literal["recipe", "dolma_sample", "nl_rx_synth", "nl_rx_synth_flat"]


class dataloader_builder(EnumBuilder[Dataset]):
    def __init__(self, dataset: str | None):
        super().__init__(dataset, Dataset, label="Dataset")

    def build(self, patcher, batch_size: int = 1, **kwargs):
        dataloader_cls = self.get_class()
        return dataloader_cls(patcher=patcher, batch_size=batch_size, **kwargs)
