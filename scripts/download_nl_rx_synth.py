"""
Download NL-RX-SYNTH dataset for offline use on Gadi HPC.

Source: https://github.com/nicholaslocascio/deep-regex
Paper: "Neural Generation of Regular Expressions from Natural Language" (EMNLP 2016)

Run this on a login node with internet access:
    python scripts/download_nl_rx_synth.py
"""

from pathlib import Path
import json
import urllib.request

OUTPUT_DIR = "/g/data/oy87/cw9909/datasets/nl_rx_synth"

BASE_URL = "https://raw.githubusercontent.com/nicholaslocascio/deep-regex/master/datasets/NL-RX-Synth"
SRC_URL = f"{BASE_URL}/src.txt"
TGT_URL = f"{BASE_URL}/targ.txt"


def download_text(url: str) -> list[str]:
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    lines = [line for line in text.strip().split("\n") if line.strip()]
    print(f"  Got {len(lines)} lines")
    return lines


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    nl_lines = download_text(SRC_URL)
    regex_lines = download_text(TGT_URL)

    assert len(nl_lines) == len(regex_lines), (
        f"Mismatch: {len(nl_lines)} NL lines vs {len(regex_lines)} regex lines"
    )

    records = [
        {"nl": nl, "regex": regex}
        for nl, regex in zip(nl_lines, regex_lines)
    ]

    out_file = output_path / "train.json"
    with open(out_file, "w") as f:
        json.dump(records, f)

    print(f"Done. Saved {len(records)} samples to {out_file}")


if __name__ == "__main__":
    main()
