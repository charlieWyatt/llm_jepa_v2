import argparse
from eval.config import EvalConfig
from eval.model_wrappers.longformer_wrapper import LongformerWrapper
from eval.model_wrappers.jepa_wrapper import JEPAWrapper
from run_glue_eval import run_glue
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["longformer", "jepa"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    args = parser.parse_args()

    config = EvalConfig(output_dir=args.output_dir)

    wrapper = (
        LongformerWrapper(args.checkpoint, config.device)
        if args.model_type == "longformer"
        else JEPAWrapper(args.checkpoint, config.device)
    )
    wrapper.load()

    results = run_glue(wrapper, config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{wrapper.get_model_name()}_glue_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()