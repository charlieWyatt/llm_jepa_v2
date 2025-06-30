from enum import Enum
from typing import Type, TypeVar, Generic, Optional

T = TypeVar("T", bound=Enum)


class EnumBuilder(Generic[T]):
    def __init__(self, value: Optional[str], enum_cls: Type[T], label: str = "Value"):
        if value is None:
            raise ValueError(
                f"{label} is not set (got None) for {enum_cls.__name__}\n"
                f"Available options: {[e.name for e in enum_cls]}")

        try:
            self.enum_value = enum_cls[value]
        except KeyError:
            raise ValueError(
                f"{label} '{value}' is not a valid option for {enum_cls.__name__}.\n"
                f"Available options: {[e.name for e in enum_cls]}")

    def get_class(self):
        return self.enum_value.value

    def build(self, *args, **kwargs):
        return self.get_class()(*args, **kwargs)
