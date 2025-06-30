from datasets import load_dataset
from textwrap import dedent
from src.dataloaders.base import BaseDataloader, BASE_DATASET_PATH

RECIPE_DATASET_PATH = f"{BASE_DATASET_PATH}/recipe1m/layer1.json"


class RecipeDataloader(BaseDataloader):
    def __init__(self, patcher, batch_size: int = 1, split="train", cache_dir=None):
        super().__init__(patcher, batch_size)

        self.dataset = load_dataset(
            "json",
            data_files=RECIPE_DATASET_PATH,
            split="train"
        )

    def format_recipe(self, example) -> str:
        title = example["title"]
        ingredients = "\n".join(
            f"- {i['text']}" for i in example["ingredients"])
        instructions = "\n".join(
            f"{idx+1}. {step['text']}" for idx, step in enumerate(example["instructions"])
        )

        return dedent(f"""
        {title}

        Ingredients:
        {ingredients}

        Instructions:
        {instructions}
        """).strip()

    def __iter__(self):
        batch = []

        for example in self.dataset:
            formatted_text = self.format_recipe(example)
            patches = self.patcher.create_patches(formatted_text)
            batch.append(patches)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
