from pydantic import BaseModel


class Dataset(BaseModel):
    receipe = "recipe"


class dataloader_builder:

    def __init__(self, dataset) -> None:
        if not Dataset(dataset):
            raise Exception(f"Dataset: {dataset} does not exist")
        pass
