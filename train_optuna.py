import argparse, json, os
from copy import deepcopy
from config import STRATEGY_CONSTS
from train_accelerate import run_train_jepa

parser = argparse.ArgumentParser()
parser.add_argument("--config-json", type=str, required=False)
args, _ = parser.parse_known_args()

config = deepcopy(STRATEGY_CONSTS)

# Load Optuna overrides
if args.config_json and os.path.exists(args.config_json):
    with open(args.config_json) as f:
        overrides = json.load(f)
    
    # Extract trial number from filename
    trial_num = args.config_json.split("_")[-1].replace(".json", "")
    
    for k, v in overrides.items():
        print(f"[OPTUNA] Overriding {k} -> {v}")
        config[k] = v
    
    # Set trial-specific checkpoint directory
    config["CHECKPOINT_DIR"] = f"{config['CHECKPOINT_DIR']}/trial_{trial_num}"
    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)

    config["WANDB_RUN_NAME"] = f"optuna_trial_{trial_num}"

# Update global config
STRATEGY_CONSTS.update(config)

run_train_jepa()