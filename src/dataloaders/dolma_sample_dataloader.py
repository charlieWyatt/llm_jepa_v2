from datasets import load_dataset, DownloadConfig
from textwrap import dedent
from pathlib import Path

# Path to the local folder that holds all the .json files
LOCAL_GLOB = "/g/data/oy87/cw9909/dolma_sample/json/*.json"


class DolmaSampleDataloader:
    def __init__(self, patcher, batch_size: int = 1, streaming=False):
        self.batch_size = batch_size
        self.patcher = patcher
        # load all the local json files
        self.dataset = load_dataset(
            "json",
            data_files=LOCAL_GLOB,
            split="train",
            streaming=True,                              # no Arrow caching, no temp copies
            download_config=DownloadConfig(local_files_only=True),
        )

    def _format_example(self, ex) -> str:
        return ex.get("text", "")

    def __iter__(self):
        batch = []
        for ex in self.dataset:
            patches = self.patcher.create_patches(self._format_example(ex))
            batch.append(patches)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch