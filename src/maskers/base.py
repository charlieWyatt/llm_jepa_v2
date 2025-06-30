from abc import ABC, abstractmethod
from typing import Any, List


class BaseMaskingStrategy(ABC):
    @abstractmethod
    def create_spans(self, patches: Any) -> List[Any]:
        pass
