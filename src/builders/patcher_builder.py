from pydantic import BaseModel


class PatchStrategy(BaseModel):
    token = "token"


class patcher_builder:

    def __init__(self, strategy) -> None:
        if not PatchStrategy(strategy):
            raise Exception(f"Dataset: {strategy} does not exist")
        pass
