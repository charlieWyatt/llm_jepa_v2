import json
from pathlib import Path

from src.dataloaders.nl_rx_synth_dataloader import LOCAL_DATA_PATH


class NLRXSynthFlatDataloader:
    """
    Flat (single-sequence) dataloader for NL-RX-Synth dataset.

    Concatenates each NL description + regex into a single string,
    tokenizes, and yields List[List[str]] batches — same format as
    DolmaSampleDataloader, so it plugs directly into train_accelerate.py.
    """

    def __init__(self, patcher, batch_size: int = 1, data_path: str | None = None):
        self.batch_size = batch_size
        self.patcher = patcher

        local_dir = Path(data_path) if data_path else Path(LOCAL_DATA_PATH)
        data_file = local_dir / "train.json"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_file}. "
                "Run scripts/download_nl_rx_synth.py first."
            )

        with open(data_file, "r") as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} NL-RX-Synth flat samples from {data_file}")

    def __iter__(self):
        batch = []
        for ex in self.samples:
            nl = ex.get("nl", "")
            regex = ex.get("regex", "")
            if not nl or not regex:
                continue

            concat_text = f"{nl} {regex}"
            patches = self.patcher.create_patches(concat_text)
            batch.append(patches)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
