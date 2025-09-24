from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DATASET_PATH = os.getenv('BASE_DATA_PATH')

class BaseDataloader(ABC):
    def __init__(self, patcher, batch_size: int = 1):
        self.patcher = patcher
        self.batch_size = batch_size

    @abstractmethod
    def __iter__(self):
        """Yield batches of patches"""
        pass
