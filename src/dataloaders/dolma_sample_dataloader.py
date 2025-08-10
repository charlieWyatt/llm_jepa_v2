from datasets import load_dataset
from textwrap import dedent

GCS_GLOB = "gs://llm_jepa/pipeline-root/dolma/v1_6-sample/**/*.json.gz"


class DolmaSampleDataloader:
    def __init__(self, patcher, batch_size: int = 1, streaming=True):
        self.batch_size = batch_size
        self.patcher = patcher
        self.dataset = load_dataset(
            "json",
            data_files=GCS_GLOB,
            split="train",
            # stream directly from GCS (no full download)
            streaming=streaming,
        )

    def _format_example(self, ex) -> str:
        title = ex.get("metadata", {}).get(
            "title") or ex.get("title") or "(untitled)"
        text = ex.get("text", "")
        url = ex.get("id", "") or ex.get("metadata", {}).get("url", "")
        return dedent(f"""
        {title}

        Source: {url}

        {text}
        """).strip()

    def __iter__(self):
        batch = []
        for ex in self.dataset:
            formatted = self._format_example(ex)
            patches = self.patcher.create_patches(formatted)
            batch.append(patches)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
