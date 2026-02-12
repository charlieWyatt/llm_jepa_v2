from datasets import load_dataset, DownloadConfig
from pathlib import Path

# Default local path for Gadi HPC (pre-downloaded)
LOCAL_DATA_PATH = "/g/data/oy87/cw9909/datasets/nl_rx_synth"
HF_DATASET_ID = "CatherineYeh/NL-RX-Synth"


class NLRXSynthDataloader:
    """
    Paired (text, code) dataloader for NL-RX-Synth dataset.

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
        if local_dir.exists():
            self.dataset = load_dataset(
                "json",
                data_files=str(local_dir / "*.json"),
                split="train",
                download_config=DownloadConfig(local_files_only=True),
            )
        else:
            self.dataset = load_dataset(HF_DATASET_ID, split="train")

    def __iter__(self):
        batch_text_tokens = []
        batch_code_tokens = []
        batch_raw_text = []
        batch_raw_code = []

        for ex in self.dataset:
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
