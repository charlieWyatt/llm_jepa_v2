import argparse
from eval.config import EvalConfig
from eval.model_wrappers.longformer_wrapper import LongformerWrapper
from eval.model_wrappers.jepa_wrapper import JEPAWrapper
from run_glue_eval import run_glue
from run_longbench_eval import run_longbench
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["longformer", "jepa"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--eval-suite", choices=["glue", "longbench", "both"], default="both")
    parser.add_argument("--longbench-mode", 
                       choices=["overall", "by_domain", "by_difficulty", "by_length"],
                       default="overall",
                       help="Analysis mode for LongBench")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom name for output file (default: model name)")
    args = parser.parse_args()

    config = EvalConfig(output_dir=args.output_dir)

    wrapper = (
        LongformerWrapper(args.checkpoint, config.device)
        if args.model_type == "longformer"
        else JEPAWrapper(args.checkpoint, config.device)
    )
    wrapper.load()

    results = {}
    
    if args.eval_suite in ["glue", "both"]:
        print("\n" + "="*60)
        print("Running GLUE Evaluation")
        print("="*60)
        results["glue"] = run_glue(wrapper, config)
    
    if args.eval_suite in ["longbench", "both"]:
        print("\n" + "="*60)
        print("Running LongBench v2 Evaluation")
        print("="*60)
        results["longbench"] = run_longbench(wrapper, config, args.longbench_mode)

    # Use custom output name if provided, otherwise use model name
    output_name = args.output_name if args.output_name else wrapper.get_model_name()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    output_file = out / f"{output_name}_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()