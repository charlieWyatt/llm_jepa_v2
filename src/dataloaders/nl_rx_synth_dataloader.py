import json
from pathlib import Path

# Default local path for Gadi HPC (pre-downloaded)
LOCAL_DATA_PATH = "/g/data/oy87/cw9909/datasets/nl_rx_synth"


class NLRXSynthDataloader:
    """
    Paired (text, code) dataloader for NL-RX-Synth dataset.

    Source: https://github.com/nicholaslocascio/deep-regex

    Each sample contains:
      - "nl": natural language description
      - "regex": regular expression pattern

    Yields batches of:
      {
        "text_tokens": List[List[str]],
        "code_tokens": List[List[str]],
        "raw_text": List[str],
        "raw_code": List[str],
      }
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

        print(f"Loaded {len(self.samples)} NL-RX-Synth samples from {data_file}")

    def __iter__(self):
        batch_text_tokens = []
        batch_code_tokens = []
        batch_raw_text = []
        batch_raw_code = []

        for ex in self.samples:
            nl = ex.get("nl", "")
            regex = ex.get("regex", "")
            if not nl or not regex:
                continue

            text_tokens = self.patcher.create_patches(nl)
            code_tokens = self.patcher.create_patches(regex)

            batch_text_tokens.append(text_tokens)
            batch_code_tokens.append(code_tokens)
            batch_raw_text.append(nl)
            batch_raw_code.append(regex)

            if len(batch_text_tokens) == self.batch_size:
                yield {
                    "text_tokens": batch_text_tokens,
                    "code_tokens": batch_code_tokens,
                    "raw_text": batch_raw_text,
                    "raw_code": batch_raw_code,
                }
                batch_text_tokens = []
                batch_code_tokens = []
                batch_raw_text = []
                batch_raw_code = []

        if batch_text_tokens:
            yield {
                "text_tokens": batch_text_tokens,
                "code_tokens": batch_code_tokens,
                "raw_text": batch_raw_text,
                "raw_code": batch_raw_code,
            }
