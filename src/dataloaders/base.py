from abc import ABC, abstractmethod

BASE_DATASET_PATH = "data"

class BaseDataloader(ABC):
    def __init__(self, patcher, batch_size: int = 1):
        self.patcher = patcher
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self):
        """Yield batches of patches"""
        pass
