from eval.tasks.longbench_dataset import LongBenchTask, LONGBENCH_DOMAINS
from eval.probes.linear_probe import train_probe, eval_probe
from typing import Dict
import numpy as np

def run_longbench(model_wrapper, config, analysis_mode: str = "overall") -> Dict:
    """
    Run LongBench v2 evaluation
    
    Args:
        model_wrapper: Model wrapper with extract_features method
        config: EvalConfig
        analysis_mode: "overall", "by_domain", "by_difficulty", or "by_length"
    """
    task = LongBenchTask()
    
    # Print statistics
    stats = task.get_stats()
    print("\n" + "="*60)
    print("LongBench v2 Statistics")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Avg context length: {stats['avg_context_length']:.0f} words")
    print(f"Max context length: {stats['max_context_length']:.0f} words")
    print("\nBy domain:")
    for domain, count in stats['by_domain'].items():
        print(f"  {domain}: {count}")
    print("\nBy difficulty:")
    for diff, count in stats['by_difficulty'].items():
        print(f"  {diff}: {count}")
    print("\nBy length:")
    for length, count in stats['by_length'].items():
        print(f"  {length}: {count}")
    
    results = {}
    
    if analysis_mode == "overall":
        print("\n" + "="*60)
        print("Overall LongBench v2 Evaluation")
        print("="*60)
        
        inputs, labels = task.get_all()
        results["overall"] = _evaluate_split(
            model_wrapper, config, inputs, labels, "Overall"
        )
    
    elif analysis_mode == "by_domain":
        print("\n" + "="*60)
        print("LongBench v2 Evaluation by Domain")
        print("="*60)
        
        for domain in LONGBENCH_DOMAINS:
            inputs, labels = task.get_by_domain(domain)
            if len(inputs) > 0:
                results[domain] = _evaluate_split(
                    model_wrapper, config, inputs, labels, domain
                )
    
    elif analysis_mode == "by_difficulty":
        print("\n" + "="*60)
        print("LongBench v2 Evaluation by Difficulty")
        print("="*60)
        
        for difficulty in ["easy", "hard"]:
            inputs, labels = task.get_by_difficulty(difficulty)
            if len(inputs) > 0:
                results[difficulty] = _evaluate_split(
                    model_wrapper, config, inputs, labels, difficulty.capitalize()
                )
    
    elif analysis_mode == "by_length":
        print("\n" + "="*60)
        print("LongBench v2 Evaluation by Length")
        print("="*60)
        
        for length in ["short", "medium", "long"]:
            inputs, labels = task.get_by_length(length)
            if len(inputs) > 0:
                results[length] = _evaluate_split(
                    model_wrapper, config, inputs, labels, length.capitalize()
                )
    
    return results

def _evaluate_split(model_wrapper, config, inputs, labels, split_name):
    """Helper to evaluate a data split"""
    print(f"\n{split_name}: {len(inputs)} samples")
    
    # Since LongBench has no train split, we'll use the data itself for training
    # This gives us an upper bound on performance
    # For a proper evaluation, you might want to use GLUE or another dataset for training
    
    print("Extracting features...")
    features = model_wrapper.extract_features(inputs, config)
    
    # 4-way classification (A/B/C/D)
    print("Training 4-way classifier...")
    probe = train_probe(features, labels, is_regression=False, max_iter=config.max_iter)
    
    print("Evaluating...")
    metrics = eval_probe(probe, features, labels, is_regression=False)
    
    print(f"Results: {metrics}")
    return metrics