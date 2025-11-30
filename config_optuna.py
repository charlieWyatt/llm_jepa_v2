import optuna
import subprocess
import json
import os
from optuna.pruners import MedianPruner
from pathlib import Path

# Path to JEPA training script and evaluation script
TRAIN_SCRIPT = "/home/561/cw9909/experiments/llm_jepa_v2/train_optuna.py"
EVAL_SCRIPT  = "/home/561/cw9909/experiments/llm_jepa_v2/run_glue_eval.py"
CHECKPOINT_DIR = "/g/data/oy87/cw9909/llm_jepa/checkpoints/llm_jepa"

def objective(trial: optuna.Trial):
    """Hyperparameter objective for JEPA training -> GLUE eval."""

    params = {
        "PATCH_SIZE": trial.suggest_categorical("PATCH_SIZE", [1, 2, 4, 8]),
        "CONTEXT_MASK_RATIO": trial.suggest_float("CONTEXT_MASK_RATIO", 0.3, 0.9),
        "TARGET_MASK_RATIO": trial.suggest_float("TARGET_MASK_RATIO", 0.10, 0.35),
        "NUM_TARGETS": trial.suggest_int("NUM_TARGETS", 1, 4),
    }

    # Save params to temp JSON
    tmp_cfg = f"/g/data/oy87/cw9909/tmp/optuna_params_{trial.number}.json"
    os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
    with open(tmp_cfg, "w") as f:
        json.dump(params, f)

    # Launch training with accelerate - use all 4 GPUs
    env = os.environ.copy()
    subprocess.run(
        [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", "4",
            "--mixed_precision", "bf16",
            TRAIN_SCRIPT,
            "--config-json", tmp_cfg
        ],
        check=True,
        env=env
    )

    # Find latest checkpoint for this trial
    trial_ckpt_dir = f"{CHECKPOINT_DIR}/trial_{trial.number}"
    ckpts = sorted(Path(trial_ckpt_dir).glob("**/*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoint found for trial {trial.number}")
    ckpt = ckpts[-1]

    # Run GLUE Evaluation
    os.makedirs("./optuna_results", exist_ok=True)
    eval_out = f"./optuna_results/optuna_eval_{trial.number}.json"
    subprocess.run([
        "python", EVAL_SCRIPT,
        "--model-type", "jepa",
        "--checkpoint", str(ckpt),
        "--output", eval_out
    ], check=True)

    # Load GLUE scores
    with open(eval_out, "r") as f:
        glue_scores = json.load(f)

    # Average score across tasks
    mean_score = sum(v["score"] for v in glue_scores.values()) / len(glue_scores)

    # Clean up checkpoint to save space (optional)
    # import shutil
    # shutil.rmtree(trial_ckpt_dir)

    return mean_score

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
    )
    study.optimize(objective, n_trials=20, timeout=86400)  # 24 hour timeout
    
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    
    # Save study
    import joblib
    joblib.dump(study, "optuna_study.pkl")